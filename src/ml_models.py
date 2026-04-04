"""
Simple sklearn models for interview-friendly explanations.

We use LogisticRegression inside a Pipeline with StandardScaler:
  - Scaler puts features on similar scales (helps linear models).
  - Logistic regression outputs probabilities -> easy risk bands.

Risk bands (tunable demo thresholds):
  low:    probability < 0.35
  medium: 0.35 <= probability < 0.65
  high:   probability >= 0.65
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DIABETES_MODEL_PATH, HEART_MODEL_PATH, MODELS_DIR
from src.preprocess import preprocess_diabetes, preprocess_heart

# Feature order must match training when predicting from dict/API
DIABETES_FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

HEART_FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "blood_pressure",
    "cholesterol",
    "fbs_over_120",
    "ekg_results",
    "max_hr",
    "exercise_angina",
    "st_depression",
    "slope_st",
    "vessels_fluro",
    "thallium",
]


def probability_to_risk_level(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def _build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def train_diabetes_model(df: pd.DataFrame) -> Pipeline:
    X = df[DIABETES_FEATURES]
    y = df["outcome"]
    model = _build_classifier()
    model.fit(X, y)
    return model


def train_heart_model(df: pd.DataFrame) -> Pipeline:
    X = df[HEART_FEATURES]
    y = df["heart_disease"]
    model = _build_classifier()
    model.fit(X, y)
    return model


def save_model(model: Any, path: str | Any) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Any) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_save_all() -> tuple[Pipeline, Pipeline]:
    d_df = preprocess_diabetes()
    h_df = preprocess_heart()
    d_model = train_diabetes_model(d_df)
    h_model = train_heart_model(h_df)
    save_model(d_model, DIABETES_MODEL_PATH)
    save_model(h_model, HEART_MODEL_PATH)
    return d_model, h_model


def load_or_train() -> tuple[Pipeline, Pipeline]:
    if DIABETES_MODEL_PATH.exists() and HEART_MODEL_PATH.exists():
        return load_model(DIABETES_MODEL_PATH), load_model(HEART_MODEL_PATH)
    return train_and_save_all()


@dataclass
class PredictionResult:
    probability: float
    risk_level: str
    label_hint: str


def predict_diabetes_row(model: Pipeline, row: dict) -> PredictionResult:
    X = pd.DataFrame([row])[DIABETES_FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    level = probability_to_risk_level(prob)
    hint = "Higher model score suggests elevated diabetes risk (demo — not medical advice)."
    return PredictionResult(probability=prob, risk_level=level, label_hint=hint)


def predict_heart_row(model: Pipeline, row: dict) -> PredictionResult:
    X = pd.DataFrame([row])[HEART_FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    level = probability_to_risk_level(prob)
    hint = "Higher model score suggests elevated heart-disease risk (demo — not medical advice)."
    return PredictionResult(probability=prob, risk_level=level, label_hint=hint)


def add_diabetes_predictions(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    out = df.copy()
    probs = model.predict_proba(out[DIABETES_FEATURES])[:, 1]
    out["ml_probability"] = probs
    out["risk_level"] = [probability_to_risk_level(p) for p in probs]
    return out


def add_heart_predictions(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    out = df.copy()
    probs = model.predict_proba(out[HEART_FEATURES])[:, 1]
    out["ml_probability"] = probs
    out["risk_level"] = [probability_to_risk_level(p) for p in probs]
    return out
