"""
SQLite storage + helper queries used by the API and dashboard.

Why SQLite for an internship project?
  - No separate server install; one file (`healthcare_risk.db`).
  - Still "real SQL": CREATE TABLE, JOIN-style thinking, aggregates.

Run `python scripts/setup_project.py` to build the DB from cleaned CSVs.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DB_PATH
from src.ml_models import (
    add_diabetes_predictions,
    add_heart_predictions,
    load_or_train,
)
from src.preprocess import preprocess_diabetes, preprocess_heart


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    return sqlite3.connect(path)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS diabetes_patients;
        DROP TABLE IF EXISTS heart_patients;

        CREATE TABLE diabetes_patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pregnancies REAL NOT NULL,
            glucose REAL NOT NULL,
            blood_pressure REAL NOT NULL,
            skin_thickness REAL NOT NULL,
            insulin REAL NOT NULL,
            bmi REAL NOT NULL,
            diabetes_pedigree REAL NOT NULL,
            age INTEGER NOT NULL,
            outcome INTEGER NOT NULL,
            ml_probability REAL NOT NULL,
            risk_level TEXT NOT NULL
        );

        CREATE TABLE heart_patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER NOT NULL,
            sex INTEGER NOT NULL,
            chest_pain_type INTEGER NOT NULL,
            blood_pressure INTEGER NOT NULL,
            cholesterol INTEGER NOT NULL,
            fbs_over_120 INTEGER NOT NULL,
            ekg_results INTEGER NOT NULL,
            max_hr INTEGER NOT NULL,
            exercise_angina INTEGER NOT NULL,
            st_depression REAL NOT NULL,
            slope_st INTEGER NOT NULL,
            vessels_fluro INTEGER NOT NULL,
            thallium REAL NOT NULL,
            heart_disease INTEGER NOT NULL,
            ml_probability REAL NOT NULL,
            risk_level TEXT NOT NULL
        );
        """
    )
    conn.commit()


def init_database(db_path: Path | None = None) -> None:
    """Rebuild DB from scratch: preprocess -> train/load models -> insert rows."""
    path = db_path or DB_PATH
    d_model, h_model = load_or_train()

    d_df = preprocess_diabetes()
    h_df = preprocess_heart()
    d_df = add_diabetes_predictions(d_df, d_model)
    h_df = add_heart_predictions(h_df, h_model)

    conn = sqlite3.connect(path)
    try:
        create_schema(conn)
        d_df.to_sql("diabetes_patients", conn, if_exists="append", index=False)
        h_df.to_sql("heart_patients", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def query_high_risk_patients(conn: sqlite3.Connection) -> dict[str, Any]:
    """Patients flagged high by model OR positive clinical label."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM diabetes_patients
        WHERE risk_level = 'high' OR outcome = 1
        """
    )
    diabetes_count = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*) FROM heart_patients
        WHERE risk_level = 'high' OR heart_disease = 1
        """
    )
    heart_count = cur.fetchone()[0]
    cur.execute(
        """
        SELECT id, age, glucose, bmi, ml_probability, risk_level, outcome
        FROM diabetes_patients
        WHERE risk_level = 'high' OR outcome = 1
        ORDER BY ml_probability DESC
        LIMIT 15
        """
    )
    diabetes_samples = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    cur.execute(
        """
        SELECT id, age, sex, blood_pressure, cholesterol, ml_probability,
               risk_level, heart_disease
        FROM heart_patients
        WHERE risk_level = 'high' OR heart_disease = 1
        ORDER BY ml_probability DESC
        LIMIT 15
        """
    )
    heart_samples = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    return {
        "diabetes_high_risk_or_positive_count": diabetes_count,
        "heart_high_risk_or_positive_count": heart_count,
        "sample_diabetes_rows": diabetes_samples,
        "sample_heart_rows": heart_samples,
    }


def query_common_risk_factors(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compare mean features: positive vs negative labels (transparent SQL stats)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            AVG(CASE WHEN outcome = 1 THEN glucose END) AS avg_glucose_positive,
            AVG(CASE WHEN outcome = 0 THEN glucose END) AS avg_glucose_negative,
            AVG(CASE WHEN outcome = 1 THEN bmi END) AS avg_bmi_positive,
            AVG(CASE WHEN outcome = 0 THEN bmi END) AS avg_bmi_negative,
            AVG(CASE WHEN outcome = 1 THEN age END) AS avg_age_positive,
            AVG(CASE WHEN outcome = 0 THEN age END) AS avg_age_negative
        FROM diabetes_patients
        """
    )
    d = dict(zip([c[0] for c in cur.description], cur.fetchone()))
    cur.execute(
        """
        SELECT
            AVG(CASE WHEN heart_disease = 1 THEN blood_pressure END) AS avg_bp_positive,
            AVG(CASE WHEN heart_disease = 0 THEN blood_pressure END) AS avg_bp_negative,
            AVG(CASE WHEN heart_disease = 1 THEN cholesterol END) AS avg_chol_positive,
            AVG(CASE WHEN heart_disease = 0 THEN cholesterol END) AS avg_chol_negative,
            AVG(CASE WHEN heart_disease = 1 THEN age END) AS avg_age_positive,
            AVG(CASE WHEN heart_disease = 0 THEN age END) AS avg_age_negative
        FROM heart_patients
        """
    )
    h = dict(zip([c[0] for c in cur.description], cur.fetchone()))
    return {"diabetes": d, "heart": h}


def query_age_gender_breakdown(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Age buckets + gender (heart only; diabetes CSV has no sex column).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            CASE
                WHEN age < 30 THEN 'under_30'
                WHEN age < 45 THEN '30_44'
                WHEN age < 60 THEN '45_59'
                ELSE '60_plus'
            END AS age_bucket,
            COUNT(*) AS n,
            SUM(outcome) AS diabetes_positive
        FROM diabetes_patients
        GROUP BY age_bucket
        ORDER BY age_bucket
        """
    )
    diabetes_age = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT
            sex,
            CASE WHEN sex = 1 THEN 'male' ELSE 'female' END AS sex_label,
            COUNT(*) AS n,
            SUM(heart_disease) AS heart_positive
        FROM heart_patients
        GROUP BY sex
        """
    )
    heart_gender = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT
            CASE
                WHEN age < 40 THEN 'under_40'
                WHEN age < 55 THEN '40_54'
                ELSE '55_plus'
            END AS age_bucket,
            COUNT(*) AS n,
            SUM(heart_disease) AS heart_positive
        FROM heart_patients
        GROUP BY age_bucket
        ORDER BY age_bucket
        """
    )
    heart_age = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]

    return {
        "diabetes_by_age_bucket": diabetes_age,
        "heart_by_gender": heart_gender,
        "heart_by_age_bucket": heart_age,
    }


def risk_summary_payload(conn: sqlite3.Connection) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        "SELECT risk_level, COUNT(*) FROM diabetes_patients GROUP BY risk_level"
    )
    d_dist = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute(
        "SELECT risk_level, COUNT(*) FROM heart_patients GROUP BY risk_level"
    )
    h_dist = {row[0]: row[1] for row in cur.fetchall()}
    return {
        "diabetes_risk_distribution": d_dist,
        "heart_risk_distribution": h_dist,
        "high_risk": query_high_risk_patients(conn),
        "risk_factors": query_common_risk_factors(conn),
        "demographics": query_age_gender_breakdown(conn),
    }


def load_patients_dataframe(
    conn: sqlite3.Connection, table: str, filters: dict | None = None
) -> pd.DataFrame:
    """Load full table into pandas for Streamlit charts (optional filters)."""
    q = f"SELECT * FROM {table}"
    df = pd.read_sql_query(q, conn)
    if not filters:
        return df
    if "min_age" in filters:
        df = df[df["age"] >= filters["min_age"]]
    if "max_age" in filters:
        df = df[df["age"] <= filters["max_age"]]
    if filters.get("sex") is not None and "sex" in df.columns:
        df = df[df["sex"] == int(filters["sex"])]
    return df
