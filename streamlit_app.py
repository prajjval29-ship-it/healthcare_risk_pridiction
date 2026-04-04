"""
Streamlit dashboard — visual layer over the FastAPI + SQLite stack.

Run (from project root, with API already on port 8010 — avoids Windows port 8000 blocks):
  streamlit run streamlit_app.py

The dashboard calls http://127.0.0.1:8010 by default. If the API is down,
it shows a clear message (still lets you explore cached SQL via local fallback
if you extend it later).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st

from src.config import DB_PATH
from src.database import get_connection, load_patients_dataframe

import os

_DEFAULT_API = os.environ.get("HEALTHCARE_API_URL", "http://127.0.0.1:8010")


def fetch_json(base: str, path: str):
    try:
        r = requests.get(f"{base.rstrip('/')}{path}", timeout=5)
        if r.ok:
            return r.json(), None
        return None, f"API error {r.status_code}: {r.text}"
    except requests.RequestException as e:
        return None, str(e)


def post_json(base: str, path: str, payload: dict):
    try:
        r = requests.post(f"{base.rstrip('/')}{path}", json=payload, timeout=10)
        if r.ok:
            return r.json(), None
        return None, f"API error {r.status_code}: {r.text}"
    except requests.RequestException as e:
        return None, str(e)


st.set_page_config(
    page_title="Healthcare Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Healthcare Risk Analyzer")
st.caption(
    "Internship demo: Python + SQL + simple ML + API. Not for real clinical decisions."
)

# --- Sidebar filters (apply to charts loaded from SQLite) ---
st.sidebar.header("Chart filters")
api_base = st.sidebar.text_input("API base URL", value=_DEFAULT_API)
min_age = st.sidebar.slider("Minimum age", 0, 100, 25)
max_age = st.sidebar.slider("Maximum age", 0, 100, 80)
sex_filter = st.sidebar.selectbox(
    "Heart data: gender (Sex in CSV)",
    options=[None, 0, 1],
    format_func=lambda x: "All" if x is None else ("Female (0)" if x == 0 else "Male (1)"),
)

summary, err = fetch_json(api_base, "/risk-summary")
if err:
    st.warning(
        f"Could not reach API ({api_base}). Start it with: "
        f"`python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010`. "
        f"Details: {err}"
    )

tab_overview, tab_diabetes, tab_heart, tab_predict = st.tabs(
    ["Overview", "Diabetes cohort", "Heart cohort", "Predict new patient"]
)

# --- Overview ---
with tab_overview:
    if summary:
        st.subheader("Risk distribution (stored patients)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Diabetes model bands**")
            st.bar_chart(
                pd.Series(summary["diabetes_risk_distribution"]).sort_index()
            )
        with c2:
            st.write("**Heart model bands**")
            st.bar_chart(
                pd.Series(summary["heart_risk_distribution"]).sort_index()
            )

        st.subheader("High-risk snapshot (SQL)")
        st.write(
            "Counts include rows where the **model** says `high` **or** the **label** is positive."
        )
        st.metric(
            "Diabetes (high or outcome=1)",
            summary["high_risk"]["diabetes_high_risk_or_positive_count"],
        )
        st.metric(
            "Heart (high or disease=1)",
            summary["high_risk"]["heart_high_risk_or_positive_count"],
        )

        st.subheader("Common risk factors (mean comparison)")
        rf = summary["risk_factors"]
        st.write("Diabetes — glucose & BMI tend higher when Outcome=1:")
        st.json(rf["diabetes"])
        st.write("Heart — BP & cholesterol tend higher when disease=1:")
        st.json(rf["heart"])

        st.subheader("Age / gender breakdown")
        st.write("Diabetes: by age bucket (no gender in source CSV).")
        st.dataframe(pd.DataFrame(summary["demographics"]["diabetes_by_age_bucket"]))
        st.write("Heart: by gender and age bucket.")
        st.dataframe(pd.DataFrame(summary["demographics"]["heart_by_gender"]))
        st.dataframe(pd.DataFrame(summary["demographics"]["heart_by_age_bucket"]))

        st.success(
            "**Insight (plain English):** wider gaps between positive/negative means "
            "point to features the model can use; the API aggregates this in `/risk-summary`."
        )
    else:
        st.info("Start the API and refresh to see overview analytics.")

# --- Diabetes charts from SQLite ---
with tab_diabetes:
    if not DB_PATH.exists():
        st.error("Database missing. Run `python scripts/setup_project.py`.")
    else:
        conn = get_connection()
        try:
            f = {"min_age": min_age, "max_age": max_age}
            df = load_patients_dataframe(conn, "diabetes_patients", f)
            st.write(f"Rows after age filter: **{len(df)}**")
            if len(df):
                viz = df.assign(
                    _size=(df["ml_probability"] * 400 + 8).clip(8, 400)
                )
                st.scatter_chart(viz, x="glucose", y="bmi", size="_size")
                hist = df.groupby("risk_level").size()
                st.bar_chart(hist)
        finally:
            conn.close()

# --- Heart charts ---
with tab_heart:
    if not DB_PATH.exists():
        st.error("Database missing. Run `python scripts/setup_project.py`.")
    else:
        conn = get_connection()
        try:
            f = {"min_age": min_age, "max_age": max_age, "sex": sex_filter}
            df = load_patients_dataframe(conn, "heart_patients", f)
            st.write(f"Rows after filters: **{len(df)}**")
            if len(df):
                viz = df.assign(
                    _size=(df["ml_probability"] * 400 + 8).clip(8, 400)
                )
                st.scatter_chart(viz, x="age", y="cholesterol", size="_size")
                hist = df.groupby("risk_level").size()
                st.bar_chart(hist)
        finally:
            conn.close()

# --- Prediction forms ---
with tab_predict:
    st.subheader("Diabetes risk (API: POST /predict-diabetes)")
    with st.form("diabetes_form"):
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 300, 120)
            blood_pressure = st.number_input("Blood pressure", 0, 200, 70)
            skin_thickness = st.number_input("Skin thickness", 0, 100, 20)
        with c2:
            insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
            bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
            dpf = st.number_input("Diabetes pedigree", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 0, 100, 45)
        submitted_d = st.form_submit_button("Predict diabetes risk")
    if submitted_d:
        payload = {
            "pregnancies": float(pregnancies),
            "glucose": float(glucose),
            "blood_pressure": float(blood_pressure),
            "skin_thickness": float(skin_thickness),
            "insulin": float(insulin),
            "bmi": float(bmi),
            "diabetes_pedigree": float(dpf),
            "age": int(age),
        }
        out, e = post_json(api_base, "/predict-diabetes", payload)
        if e:
            st.error(e)
        else:
            level = out["risk_level"].upper()
            st.metric("Risk level", level)
            st.metric("Model probability (positive class)", out["probability"])
            st.write(out["insights"])

    st.subheader("Heart risk (API: POST /predict-heart-risk)")
    with st.form("heart_form"):
        c1, c2 = st.columns(2)
        with c1:
            h_age = st.number_input("Age (heart)", 0, 100, 55)
            h_sex = st.selectbox("Sex (1=male, 0=female)", [1, 0])
            chest = st.number_input("Chest pain type (1-4)", 1, 4, 4)
            h_bp = st.number_input("BP", 80, 250, 130)
            chol = st.number_input("Cholesterol", 100, 600, 250)
            fbs = st.selectbox("FBS over 120", [0, 1])
        with c2:
            ekg = st.number_input("EKG results", 0, 2, 2)
            mxhr = st.number_input("Max HR", 60, 220, 150)
            ex = st.selectbox("Exercise angina", [0, 1])
            stdep = st.number_input("ST depression", 0.0, 10.0, 1.0)
            slope = st.number_input("Slope of ST", 0, 3, 2)
            vessels = st.number_input("Vessels fluro", 0, 4, 0)
            thal = st.number_input("Thallium", 0.0, 10.0, 3.0)
        submitted_h = st.form_submit_button("Predict heart risk")
    if submitted_h:
        payload = {
            "age": int(h_age),
            "sex": int(h_sex),
            "chest_pain_type": int(chest),
            "blood_pressure": int(h_bp),
            "cholesterol": int(chol),
            "fbs_over_120": int(fbs),
            "ekg_results": int(ekg),
            "max_hr": int(mxhr),
            "exercise_angina": int(ex),
            "st_depression": float(stdep),
            "slope_st": int(slope),
            "vessels_fluro": int(vessels),
            "thallium": float(thal),
        }
        out, e = post_json(api_base, "/predict-heart-risk", payload)
        if e:
            st.error(e)
        else:
            st.metric("Risk level", out["risk_level"].upper())
            st.metric("Model probability (positive class)", out["probability"])
            st.write(out["insights"])
