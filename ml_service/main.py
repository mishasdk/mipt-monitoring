import json
import os
from pathlib import Path

from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

METRICS_FILE = "/metrics.json"

INFERENCE_TIME = Gauge("ml_inference_time_ms", "Model inference time in milliseconds")
PRECISION = Gauge("ml_precision", "Model precision score")


@app.get("/metrics", response_class=Response)
def get_metrics():
    data = json.loads(METRICS_FILE.read_text())
    INFERENCE_TIME.set(data["inference_time_ms"])
    PRECISION.set(data["precision"])
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/metrics")
def get_metrics_json():
    return json.loads(METRICS_FILE.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}
