import json
import logging
import os
import tempfile
from io import BytesIO
from typing import List, Optional, Literal

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

logger = logging.getLogger("predictive-maintenance-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    TF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - best effort fallback
    TF_AVAILABLE = False
    logger.warning("TensorFlow unavailable: %s. Falling back to simulated model.", exc)

FEATURE_NAMES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "type_l",
    "type_m",
]
SEQUENCE_LENGTH = 50
REMOTE_BASE_URL = "https://raw.githubusercontent.com/sidnei-almeida/manutencao_preditiva_lstm/main"


class ManualReading(BaseModel):
    air_temperature_k: float = Field(..., ge=250, le=400, description="Ambient temperature in Kelvin.")
    process_temperature_k: float = Field(..., ge=250, le=420, description="Process temperature in Kelvin.")
    rotational_speed_rpm: float = Field(..., ge=0, le=4000, description="Rotational speed in RPM.")
    torque_nm: float = Field(..., ge=0, le=200, description="Torque in Newton-meters.")
    tool_wear_min: float = Field(..., ge=0, le=500, description="Tool wear in minutes.")
    product_type: Literal["H", "L", "M"] = Field("M", description="Product type: H (baseline), L, or M.")


class PredictionRequest(BaseModel):
    reading: Optional[ManualReading] = Field(
        None, description="Single timestamp sensor reading. The API will broadcast it to a 50-step sequence."
    )
    sequence: Optional[List[List[float]]] = Field(
        None,
        description="Preprocessed sequence with shape [timesteps, 7]. Timesteps shorter than 50 will be padded using the last row.",
    )

    @root_validator
    def ensure_payload(cls, values: dict) -> dict:
        if not values.get("reading") and not values.get("sequence"):
            raise ValueError("Provide either `reading` or `sequence`.")
        return values


class PredictionResponse(BaseModel):
    probability: float
    predicted_label: int
    threshold: float = 0.5
    details: dict


class StatusResponse(BaseModel):
    model_loaded: bool
    data_loaded: bool
    training_loaded: bool
    tensorflow_available: bool


class MetadataResponse(BaseModel):
    project: str
    description: str
    version: str
    features: List[str]
    sequence_length: int
    dataset: dict
    training: dict


class SimulatedModel:
    """Simple heuristic model for environments without TensorFlow."""

    @staticmethod
    def predict(batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 3 or batch.shape[2] != len(FEATURE_NAMES):
            raise ValueError("Expected input shape (batch, timestep, features=7).")

        last_frame = batch[:, -1, :]
        risk_score = (
            np.clip(last_frame[:, 0] - 1.5, 0, None) * 0.20
            + np.clip(last_frame[:, 1] - 1.5, 0, None) * 0.20
            + np.clip(last_frame[:, 3] - 1.5, 0, None) * 0.15
            + np.clip(last_frame[:, 4] - 1.5, 0, None) * 0.35
            + np.random.random(len(last_frame)) * 0.10
        )
        probability = 1.0 / (1.0 + np.exp(-risk_score * 2.0))
        return probability.reshape(-1, 1).astype("float32")


def load_training_data() -> dict:
    local_path = "treinamento/training_summary.json"
    try:
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as local_file:
                return json.load(local_file)

        training_url = f"{REMOTE_BASE_URL}/treinamento/training_summary.json"
        response = requests.get(training_url, timeout=30, headers={"User-Agent": "predictive-maintenance-api/1.0"})
        if response.status_code == 200:
            return response.json()

        logger.warning("Unable to download training summary (status %s). Returning demo metadata.", response.status_code)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Training summary loading error: %s. Returning demo metadata.", exc)

    return {
        "model_architecture": "Demo LSTM",
        "training_parameters": {"epochs": 50, "batch_size": 64, "sequence_length": SEQUENCE_LENGTH},
        "dataset_info": {"training_samples": 8000, "testing_samples": 2000, "features_per_timestep": len(FEATURE_NAMES)},
        "final_evaluation": {"test_accuracy": 0.9518, "test_loss": 0.1234},
    }


def load_processed_data() -> tuple[np.ndarray, np.ndarray]:
    X_local_path = "dados/X_processed.npy"
    y_local_path = "dados/y_processed.npy"

    try:
        if os.path.exists(X_local_path) and os.path.exists(y_local_path):
            X = np.load(X_local_path, allow_pickle=True).astype("float32")
            y = np.load(y_local_path, allow_pickle=True).astype("float32")
            return X, y

        X_url = f"{REMOTE_BASE_URL}/dados/X_processed.npy"
        y_url = f"{REMOTE_BASE_URL}/dados/y_processed.npy"
        X_response = requests.get(X_url, timeout=60, headers={"User-Agent": "predictive-maintenance-api/1.0"})
        y_response = requests.get(y_url, timeout=60, headers={"User-Agent": "predictive-maintenance-api/1.0"})

        if X_response.status_code == 200 and y_response.status_code == 200:
            X = np.load(BytesIO(X_response.content), allow_pickle=True).astype("float32")
            y = np.load(BytesIO(y_response.content), allow_pickle=True).astype("float32")
            return X, y

        logger.warning(
            "Could not download processed data (X status %s, y status %s). Generating synthetic dataset.",
            X_response.status_code,
            y_response.status_code,
        )
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Processed data loading error: %s. Generating synthetic dataset.", exc)

    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=42)
    samples = 10_000

    air_temperature = np.clip(rng.normal(300, 15, samples), 280, 350)
    process_temperature = np.clip(air_temperature + rng.normal(10, 5, samples), 285, 360)
    rotational_speed = np.clip(rng.normal(2000, 300, samples), 1000, 3000)
    torque = np.clip(rng.normal(40, 15, samples), 0, 120)
    tool_wear = np.clip(rng.normal(150, 50, samples), 0, 350)
    type_l = rng.choice([0, 1], samples, p=[0.7, 0.3])
    type_m = rng.choice([0, 1], samples, p=[0.6, 0.4])

    X = np.column_stack([air_temperature, process_temperature, rotational_speed, torque, tool_wear, type_l, type_m]).astype(
        "float32"
    )
    scaler = StandardScaler()
    X[:, :5] = scaler.fit_transform(X[:, :5])

    failure_risk = (
        (X[:, 0] > 1.5).astype("float32") * 0.15
        + (X[:, 1] > 1.5).astype("float32") * 0.20
        + (X[:, 3] > 1.5).astype("float32") * 0.25
        + (X[:, 4] > 1.5).astype("float32") * 0.30
        + rng.random(samples) * 0.10
    )
    y = (failure_risk > 0.35).astype("float32")
    return X, y


def load_model():
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available. Using simulated model.")
        return SimulatedModel()

    local_model_path = "modelos/predictive_maintenance_model.keras"
    try:
        if os.path.exists(local_model_path):
            with tf.device("/CPU:0"):
                model = tf.keras.models.load_model(local_model_path, compile=False)
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                return model

        model_url = f"{REMOTE_BASE_URL}/modelos/predictive_maintenance_model.keras"
        response = requests.get(model_url, timeout=60, headers={"User-Agent": "predictive-maintenance-api/1.0"})
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            with tf.device("/CPU:0"):
                model = tf.keras.models.load_model(tmp_path, compile=False)
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            os.unlink(tmp_path)
            return model

        logger.warning("Unable to fetch remote model (status %s). Using demo model.", response.status_code)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Model loading error: %s. Using demo model.", exc)

    return create_demo_model()


def create_demo_model():
    if not TF_AVAILABLE:
        return SimulatedModel()

    with tf.device("/CPU:0"):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(64, input_shape=(SEQUENCE_LENGTH, len(FEATURE_NAMES))),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def manual_reading_to_array(reading: ManualReading) -> np.ndarray:
    type_l = 1.0 if reading.product_type == "L" else 0.0
    type_m = 1.0 if reading.product_type == "M" else 0.0
    return np.array(
        [
            reading.air_temperature_k,
            reading.process_temperature_k,
            reading.rotational_speed_rpm,
            reading.torque_nm,
            reading.tool_wear_min,
            type_l,
            type_m,
        ],
        dtype="float32",
    )


def ensure_sequence(payload: PredictionRequest) -> np.ndarray:
    if payload.sequence:
        array = np.asarray(payload.sequence, dtype="float32")
        if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES):
            raise HTTPException(status_code=400, detail=f"Sequence must be of shape [steps, {len(FEATURE_NAMES)}].")
        if array.shape[0] < SEQUENCE_LENGTH:
            padding = np.repeat(array[-1, :][None, :], SEQUENCE_LENGTH - array.shape[0], axis=0)
            array = np.vstack([array, padding])
        elif array.shape[0] > SEQUENCE_LENGTH:
            array = array[-SEQUENCE_LENGTH:, :]
        return array

    if payload.reading is None:
        raise HTTPException(status_code=400, detail="Provide either `sequence` or `reading`.")

    reading_array = manual_reading_to_array(payload.reading)
    sequence = np.tile(reading_array, (SEQUENCE_LENGTH, 1))
    return sequence


def run_inference(sequence: np.ndarray) -> float:
    batch = sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURE_NAMES)).astype("float32")

    if isinstance(MODEL, SimulatedModel):
        probability = float(MODEL.predict(batch)[0][0])
    else:
        probability = float(MODEL.predict(batch, verbose=0)[0][0])
    return float(np.clip(probability, 0.0, 1.0))


APP = FastAPI(
    title="Predictive Maintenance LSTM API",
    description="Backend API for predictive maintenance using an LSTM model. Optimized for Hugging Face Spaces deployments.",
    version="1.0.0",
)

TRAINING_DATA = load_training_data()
X_DATA, Y_DATA = load_processed_data()
MODEL = load_model()


@APP.get("/", response_model=dict)
def root():
    return {
        "message": "Predictive Maintenance LSTM API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@APP.get("/health", response_model=StatusResponse)
def health():
    return StatusResponse(
        model_loaded=MODEL is not None,
        data_loaded=X_DATA is not None and Y_DATA is not None,
        training_loaded=TRAINING_DATA is not None,
        tensorflow_available=TF_AVAILABLE,
    )


@APP.get("/metadata", response_model=MetadataResponse)
def metadata():
    dataset_info = TRAINING_DATA.get("dataset_info", {})
    training_info = TRAINING_DATA.get("training_parameters", {})
    final_eval = TRAINING_DATA.get("final_evaluation", {})

    return MetadataResponse(
        project="Predictive Maintenance LSTM API",
        description="REST API serving an LSTM model trained to detect potential machine failures.",
        version="1.0.0",
        features=FEATURE_NAMES,
        sequence_length=SEQUENCE_LENGTH,
        dataset={
            "samples": int(X_DATA.shape[0]) if X_DATA is not None else None,
            "failure_rate": float(Y_DATA.mean()) if Y_DATA is not None else None,
            "source": "Local files or GitHub artifacts",
        },
        training={
            "parameters": training_info,
            "evaluation": final_eval,
            "architecture": TRAINING_DATA.get("model_architecture", "Unknown"),
        },
    )


@APP.get("/sample", response_model=dict)
def sample():
    if X_DATA is None or Y_DATA is None:
        raise HTTPException(status_code=500, detail="Dataset unavailable.")

    idx = int(np.random.randint(0, len(X_DATA)))
    sample_features = X_DATA[idx].astype("float32").tolist()
    sample_label = int(Y_DATA[idx])

    probability = run_inference(np.tile(sample_features, (SEQUENCE_LENGTH, 1)))

    return {
        "index": idx,
        "features": dict(zip(FEATURE_NAMES, sample_features)),
        "label": sample_label,
        "predicted_probability": probability,
    }


@APP.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    sequence = ensure_sequence(payload)
    probability = run_inference(sequence)
    label = int(probability >= 0.5)

    details = {
        "sequence_steps": int(sequence.shape[0]),
        "features_order": FEATURE_NAMES,
        "uses_simulated_model": isinstance(MODEL, SimulatedModel),
    }

    if payload.reading:
        details["reading"] = payload.reading.dict()

    return PredictionResponse(probability=probability, predicted_label=label, details=details)


app = APP

