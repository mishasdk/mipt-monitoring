import json
import os
from pathlib import Path

from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

from model import precision_clean, run_drift_report

app = FastAPI()

METRICS_FILE = os.getenv("METRICS_FILE", "/metrics.json")

INFERENCE_TIME = Gauge("ml_inference_time_ms", "Model inference time in milliseconds")
PRECISION = Gauge("ml_precision", "Model precision score")
DRIFT_SCORE = Gauge("ml_drift_score", "Share of drifted features (0–1)")
DRIFTED_FEATURES = Gauge("ml_drifted_features_count", "Number of features with detected drift")
PRECISION_DRIFTED_GAUGE = Gauge("ml_precision_drifted", "Model precision evaluated on current data")

_drift_state: dict = {
    "drift_factor": 1.0,
    "drift_score": 0.0,
    "drifted_features_count": 0,
    "drift_detected": False,
    "precision_clean": precision_clean,
    "precision_drifted": precision_clean,  # no drift yet → same as clean
    "last_run_at": None,
}


def _apply_drift_result(result: dict) -> None:
    _drift_state.update(result)
    DRIFT_SCORE.set(result["drift_score"])
    DRIFTED_FEATURES.set(result["drifted_features_count"])
    PRECISION_DRIFTED_GAUGE.set(result["precision_drifted"])


@app.get("/metrics", response_class=Response)
def get_metrics():
    data = json.loads(Path(METRICS_FILE).read_text())
    INFERENCE_TIME.set(data["inference_time_ms"])
    PRECISION.set(data["precision"])
    DRIFT_SCORE.set(_drift_state["drift_score"])
    DRIFTED_FEATURES.set(_drift_state["drifted_features_count"])
    PRECISION_DRIFTED_GAUGE.set(_drift_state["precision_drifted"])
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/metrics")
def get_metrics_json():
    data = json.loads(Path(METRICS_FILE).read_text())
    data["precision_clean"] = _drift_state["precision_clean"]
    data["precision_drifted"] = _drift_state["precision_drifted"]
    data["drift_score"] = _drift_state["drift_score"]
    data["drifted_features_count"] = _drift_state["drifted_features_count"]
    return data


@app.get("/api/drift")
def get_drift_state():
    """Return current drift state (no recomputation)."""
    return _drift_state


@app.post("/api/drift/trigger")
def trigger_drift():
    """Simulate data drift: current batch = reference * 10."""
    _apply_drift_result(run_drift_report(factor=10.0))
    return _drift_state


@app.post("/api/drift/reset")
def reset_drift():
    """Reset to no drift: current batch matches reference distribution."""
    _apply_drift_result(run_drift_report(factor=1.0))
    return _drift_state


@app.get("/health")
def health():
    return {"status": "ok"}
