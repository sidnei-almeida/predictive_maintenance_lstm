#!/bin/bash

set -euo pipefail

echo "🔧 Starting Predictive Maintenance LSTM API..."

if [ -d ".venv" ]; then
    echo "🐍 Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "🐍 Activating virtual environment (venv)..."
    source venv/bin/activate
fi

REQUIRED_FILES=(
    "dados/X_processed.npy"
    "dados/y_processed.npy"
    "modelos/predictive_maintenance_model.keras"
    "treinamento/training_summary.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: $file not found. The API will attempt to download or synthesize it."
    fi
done

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7860}

echo "🚀 Launching uvicorn on http://${HOST}:${PORT}"
uvicorn app:app --host "${HOST}" --port "${PORT}" "$@"
