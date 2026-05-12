import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="ML Monitoring Service")

METRICS_FILE = Path(os.getenv("METRICS_FILE", "/metrics.json"))


@app.get("/metrics")
def get_metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    try:
        data = json.loads(METRICS_FILE.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {e}")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}
