---
title: Predictive Maintenance LSTM API
emoji: 🔧
colorFrom: teal
colorTo: indigo
sdk: docker
license: mit
pinned: false
---

# Predictive Maintenance LSTM API

Production-ready REST API for predictive maintenance, backed by a Long Short-Term Memory (LSTM) neural network.  
The service is optimized for deployment on [Hugging Face Spaces](https://huggingface.co/spaces) as a Docker Space, and it exposes endpoints that your custom HTML/CSS/JS frontend can consume.

## Features

- 🚀 **Ready for Spaces**: Dockerfile and metadata prepared for instant deployment.  
- 🤖 **LSTM backend**: Uses the pre-trained `predictive_maintenance_model.keras` model (with graceful fallbacks).  
- 📈 **Rich metadata**: Access training stats, dataset details, and ready-to-consume health endpoints.  
- 🧠 **Fallback heuristics**: When TensorFlow is unavailable, the API switches to a simulated heuristic predictor so the Space remains interactive.  
- 📦 **Self-contained artifacts**: Loads local assets first and automatically falls back to the GitHub versions if needed.

## Project Structure

```
manutencao_preditiva_lstm/
├── app.py                      # FastAPI application (entry point for Spaces)
├── Dockerfile                  # Space Docker configuration
├── requirements.txt            # Python dependencies
├── dados/                      # Processed dataset (features/labels)
├── modelos/                    # Pre-trained Keras model
├── notebooks/                  # Data exploration and training notebooks
├── treinamento/                # Training summary and metrics
└── ...
```

## API Overview

| Method | Endpoint    | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/`         | Basic welcome payload with helpful links     |
| GET    | `/health`   | Component status (model/data/training)       |
| GET    | `/metadata` | Dataset, training, and model descriptors     |
| GET    | `/sample`   | Serves a random dataset sample + prediction  |
| POST   | `/predict`  | Run inference using manual or sequence data  |

### Prediction Payloads

You can either send a single reading (the API broadcasts it to a 50-step sequence) or a full preprocessed sequence.

#### Single Reading

```json
{
  "reading": {
    "air_temperature_k": 298.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 45.0,
    "tool_wear_min": 120.0,
    "product_type": "L"
  }
}
```

#### Preprocessed Sequence

```json
{
  "sequence": [
    [0.12, 0.32, -0.48, 0.21, 0.45, 0.0, 1.0],
    [0.10, 0.30, -0.50, 0.22, 0.46, 0.0, 1.0]
  ]
}
```

If the sequence contains fewer than 50 timesteps, the API pads it using the last frame; if it is longer, the most recent 50 timesteps are used.

### Response Example

```json
{
  "probability": 0.7421,
  "predicted_label": 1,
  "threshold": 0.5,
  "details": {
    "sequence_steps": 50,
    "features_order": [
      "air_temperature_k",
      "process_temperature_k",
      "rotational_speed_rpm",
      "torque_nm",
      "tool_wear_min",
      "type_l",
      "type_m"
    ],
    "uses_simulated_model": false,
    "reading": {
      "air_temperature_k": 298.0,
      "process_temperature_k": 310.0,
      "rotational_speed_rpm": 1500.0,
      "torque_nm": 45.0,
      "tool_wear_min": 120.0,
      "product_type": "L"
    }
  }
}
```

## Running Locally

```bash
git clone https://github.com/sidnei-almeida/manutencao_preditiva_lstm.git
cd manutencao_preditiva_lstm
python -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

The interactive docs will be available at `http://localhost:7860/docs`.

## Deploying to Hugging Face Spaces

1. Create a **Docker Space**.  
2. Push this repository to the Space (or connect it as a Git submodule).  
3. Spaces automatically detects the `Dockerfile`, installs dependencies, and launches `uvicorn app:app` on port 7860.

### Large File Storage (LFS)

Model and dataset artifacts are tracked with Git LFS via the `.gitattributes` file.  
Before committing locally, ensure LFS is installed and pull the tracked binaries:

```bash
git lfs install
git lfs pull
```

When pushing to the Space, Hugging Face will store these large files efficiently ([Spaces guide](https://huggingface.co/spaces/salmeida/predictive-maintenance-lstm/)).

### Environment Notes

- The image defaults to CPU execution (`tensorflow-cpu`); GPU layers are disabled in code.  
- If TensorFlow cannot load, a heuristic model keeps the API responsive (flagged in responses via `uses_simulated_model`).  
- `packages.txt` provides runtime libraries required by TensorFlow (GL, OpenMP).

## Frontend Integration

You can build any UI stack (HTML/CSS/JS, React, etc.) and call the API from the same Space or from an external frontend. Suggested flow:

1. Fetch `/metadata` once to display project information.  
2. Use `/health` for heartbeat monitoring.  
3. Invoke `/predict` with the user inputs (form fields, sliders, etc.).  
4. Optionally show `/sample` responses for demo or QA purposes.

## Model Artifacts

- `modelos/predictive_maintenance_model.keras`: Pre-trained binary classifier (LSTM).  
- `treinamento/training_summary.json`: document with accuracy, loss, dataset splits, and hyperparameters.  
- `dados/X_processed.npy`, `dados/y_processed.npy`: Processed features/labels aligned with the model input shape.

## Development Roadmap

- Add streaming predictions for near-real-time sensors.  
- Provide calibration and explainability endpoints (feature attributions, SHAP summaries).  
- Ship an end-to-end demo Space with the planned custom frontend.  
- Automate re-training using the notebooks and CI/CD triggers.

## License

Released under the [MIT License](LICENSE).  
Created by [Sidnei Almeida](https://github.com/sidnei-almeida). Pull requests and community contributions are welcome!
