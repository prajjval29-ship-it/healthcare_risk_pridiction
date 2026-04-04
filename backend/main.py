"""
FastAPI backend — REST layer on top of ML + SQLite.

Endpoints:
  POST /predict-diabetes      — single patient features -> probability + risk band
  POST /predict-heart-risk    — same for heart dataset schema
  GET  /risk-summary          — aggregates from SQL (dashboard-friendly)

Run from project root:
  python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import get_connection, risk_summary_payload
from src.ml_models import (
    HEART_FEATURES,
    DIABETES_FEATURES,
    load_or_train,
    predict_diabetes_row,
    predict_heart_row,
)

app = FastAPI(
    title="Healthcare Risk Analyzer API",
    description="Internship demo: logistic regression + SQLite analytics (not for clinical use).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_d_model = None
_h_model = None


def get_models():
    global _d_model, _h_model
    if _d_model is None or _h_model is None:
        _d_model, _h_model = load_or_train()
    return _d_model, _h_model


class DiabetesInput(BaseModel):
    pregnancies: float = Field(..., ge=0)
    glucose: float = Field(..., ge=0)
    blood_pressure: float = Field(..., ge=0)
    skin_thickness: float = Field(..., ge=0)
    insulin: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    diabetes_pedigree: float = Field(..., ge=0)
    age: int = Field(..., ge=0, le=120)


class HeartInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)  # 1 = male, 0 = female in this CSV
    chest_pain_type: int = Field(..., ge=1, le=4)
    blood_pressure: int = Field(..., ge=0)
    cholesterol: int = Field(..., ge=0)
    fbs_over_120: int = Field(..., ge=0, le=1)
    ekg_results: int = Field(..., ge=0)
    max_hr: int = Field(..., ge=0)
    exercise_angina: int = Field(..., ge=0, le=1)
    st_depression: float = Field(..., ge=0)
    slope_st: int = Field(..., ge=0)
    vessels_fluro: int = Field(..., ge=0)
    thallium: float = Field(..., ge=0)


@app.get("/")
def root():
    return {
        "service": "Healthcare Risk Analyzer",
        "docs": "/docs",
        "endpoints": [
            "/predict-diabetes",
            "/predict-heart-risk",
            "/risk-summary",
        ],
    }


@app.post("/predict-diabetes")
def predict_diabetes(body: DiabetesInput):
    """
    Returns calibrated-style probability (model output) + low/medium/high band.
    """
    d_model, _ = get_models()
    row = body.model_dump()
    # Ensure column order for sklearn
    ordered = {k: row[k] for k in DIABETES_FEATURES}
    result = predict_diabetes_row(d_model, ordered)
    return {
        "probability": round(result.probability, 4),
        "risk_level": result.risk_level,
        "insights": result.label_hint,
        "features_used": DIABETES_FEATURES,
    }


@app.post("/predict-heart-risk")
def predict_heart(body: HeartInput):
    _, h_model = get_models()
    row = body.model_dump()
    ordered = {k: row[k] for k in HEART_FEATURES}
    result = predict_heart_row(h_model, ordered)
    return {
        "probability": round(result.probability, 4),
        "risk_level": result.risk_level,
        "insights": result.label_hint,
        "features_used": HEART_FEATURES,
    }


@app.get("/risk-summary")
def risk_summary():
    """
    Pulls precomputed rows from SQLite + runs grouped analytics queries.
    If DB missing, returns 503 with setup hint.
    """
    db_file = ROOT / "healthcare_risk.db"
    if not db_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Database not found. Run: python scripts/setup_project.py",
        )
    conn = get_connection(db_file)
    try:
        return risk_summary_payload(conn)
    finally:
        conn.close()
