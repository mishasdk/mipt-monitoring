import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from evidently.ui.workspace import Workspace

WORKSPACE_PATH = os.getenv("EVIDENTLY_WORKSPACE", "/evidently_workspace")

_iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    _iris.data, _iris.target, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Reference batch — clean test-set distribution
reference_data: pd.DataFrame = X_test.reset_index(drop=True)
_y_test = y_test.reset_index(drop=True)

precision_clean: float = round(
    precision_score(_y_test, model.predict(reference_data), average="macro"), 4
)


def run_drift_report(factor: float = 1.0) -> dict:
    _rng = np.random.default_rng(0)
    current = (
        reference_data * factor + _rng.normal(0, 0.05, reference_data.shape)
    ).reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current)
    r = report.as_dict()["metrics"][0]["result"]

    precision_on_current = round(
        precision_score(_y_test, model.predict(current), average="macro"), 4
    )

    return {
        "drift_factor": factor,
        "drift_score": round(r["share_of_drifted_columns"], 4),
        "drifted_features_count": r["number_of_drifted_columns"],
        "drift_detected": r["dataset_drift"],
        "precision_clean": precision_clean,
        "precision_drifted": precision_on_current,
        "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_quality_report(factor: float = 1.0) -> dict:
    _rng = np.random.default_rng(0)
    current = (
        reference_data * factor + _rng.normal(0, 0.05, reference_data.shape)
    ).reset_index(drop=True)

    report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current)

    ws = Workspace.create(WORKSPACE_PATH)
    projects = ws.list_projects()
    project = next((p for p in projects if p.name == "ML Quality"), None)
    if project is None:
        project = ws.create_project("ML Quality")
        project.description = "Data quality and drift monitoring for Iris ML service"
        project.save()
    ws.add_report(project.id, report)

    summary = report.as_dict()["metrics"][0]["result"].get("current", {})
    n_rows = summary.get("number_of_rows", len(current))
    n_cols = summary.get("number_of_columns", len(current.columns))
    n_missing = summary.get("number_of_missing_values", 0)

    return {
        "number_of_missing_values": n_missing,
        "share_of_missing_values": round(n_missing / max(n_rows * n_cols, 1), 4),
        "number_of_duplicated_rows": summary.get("number_of_duplicated_rows", 0),
        "number_of_rows": n_rows,
        "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
