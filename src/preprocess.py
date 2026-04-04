"""
Clean and preprocess raw CSVs before SQL + ML.

Diabetes (Pima Indians):
  Zeros in Glucose, BP, SkinThickness, Insulin, BMI are often missing —
  we replace with NaN and impute medians (common practice for this dataset).

Heart disease:
  Normalize column names, encode target Presence/Absence -> 1/0.
"""
from __future__ import annotations

import pandas as pd

from src.config import DIABETES_CSV, HEART_CSV


def load_raw_diabetes() -> pd.DataFrame:
    return pd.read_csv(DIABETES_CSV)


def load_raw_heart() -> pd.DataFrame:
    return pd.read_csv(HEART_CSV)


def preprocess_diabetes(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returns a cleaned copy ready for modeling and SQLite.
    Column names are snake_case for the database layer.
    """
    if df is None:
        df = load_raw_diabetes()
    out = df.copy()

    # Clinical zeros are invalid for these fields in this dataset
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        if col in out.columns:
            out[col] = out[col].replace(0, pd.NA)

    # Impute missing (invalid zeros) with column medians — standard for this dataset
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)

    rename_map = {
        "Pregnancies": "pregnancies",
        "Glucose": "glucose",
        "BloodPressure": "blood_pressure",
        "SkinThickness": "skin_thickness",
        "Insulin": "insulin",
        "BMI": "bmi",
        "DiabetesPedigreeFunction": "diabetes_pedigree",
        "Age": "age",
        "Outcome": "outcome",
    }
    out = out.rename(columns=rename_map)

    # Ensure plain numeric dtypes (avoids pd.NA / nullable ints breaking sklearn)
    feature_cols = [c for c in out.columns if c != "outcome"]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce")
    medians2 = out.median(numeric_only=True)
    out = out.fillna(medians2)
    out = out.dropna()
    out["outcome"] = out["outcome"].astype(int)
    return out


def preprocess_heart(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw_heart()
    out = df.copy()

    # Friendly snake_case names aligned with CSV meaning
    rename_map = {
        "Age": "age",
        "Sex": "sex",
        "Chest pain type": "chest_pain_type",
        "BP": "blood_pressure",
        "Cholesterol": "cholesterol",
        "FBS over 120": "fbs_over_120",
        "EKG results": "ekg_results",
        "Max HR": "max_hr",
        "Exercise angina": "exercise_angina",
        "ST depression": "st_depression",
        "Slope of ST": "slope_st",
        "Number of vessels fluro": "vessels_fluro",
        "Thallium": "thallium",
        "Heart Disease": "heart_disease",
    }
    out = out.rename(columns=rename_map)

    out["heart_disease"] = out["heart_disease"].map(
        {"Presence": 1, "Absence": 0}
    )
    out = out.dropna(subset=["heart_disease"])

    # Coerce numeric columns (handles odd strings if any)
    num_cols = [c for c in out.columns if c != "heart_disease"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna()
    return out
