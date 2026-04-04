# Healthcare Risk Analyzer

Internship-style project: clean two public health CSVs, store them in **SQLite**, train **logistic regression** models, expose **FastAPI** endpoints, and visualize everything in a **Streamlit** dashboard.

**Disclaimer:** This is a learning demo. It is **not** validated for clinical or diagnostic use.



1. **Data pipeline:** It preprocess the Pima diabetes file by treating clinical zeros as missing values where appropriate, impute medians, and normalize the heart-disease CSV (rename columns, encode `Presence`/`Absence` as 1/0).

2. **Database:** It load cleaned rows into SQLite with a clear schema. Analytics (high-risk counts, mean comparisons by label, age/gender buckets) are plain SQL — easy to show in `sql/analytics.sql` or in the API.

3. **ML:** It use **logistic regression** inside a **Pipeline** with **StandardScaler** for stable training, then map predicted probability to **low / medium / high** risk bands for storytelling.

4. **Backend:** FastAPI validates JSON with **Pydantic**, returns probabilities plus human-readable insight text, and aggregates SQL for `/risk-summary`.

5. **Frontend:** Streamlit filters cohorts by age (and gender for heart data), plots distributions, and calls the API for live predictions on new “patients.”

## Tech stack

| Layer        | Choice                                      |
|-------------|----------------------------------------------|
| Language    | Python 3.10+                                 |
| ML          | scikit-learn (LogisticRegression + Pipeline) |
| Database    | SQLite (`healthcare_risk.db`)                |
| API         | FastAPI + Uvicorn                            |
| Dashboard   | Streamlit                                    |

## Project layout

```
healthcare_risk_pridiction-main/
├── data/                          # diabetes.csv, Heart_Disease_Prediction.csv
├── sql/analytics.sql              # Documented SQL you can run in any SQLite client
├── src/
│   ├── config.py                  # Paths (portable across machines)
│   ├── preprocess.py              # Cleaning rules
│   ├── ml_models.py               # Train / load / predict / risk bands
│   ├── database.py                # Schema, load, analytics queries
│   └── load_data.py               # Quick CSV loader for exploration
├── backend/main.py                # FastAPI app
├── scripts/setup_project.py       # Train models + build DB (one command)
├── streamlit_app.py               # Dashboard
├── models/                        # *.pkl files created by setup (gitignored)
└── requirements.txt
```

## Setup and run

**Folder tip:** Some downloads unzip as *two* nested folders named `healthcare_risk_pridiction-main`. If `pip install -r requirements.txt` says the file is missing, you are probably in the **outer** folder: either use the `requirements.txt` in that outer folder, or `cd` into the **inner** folder (the one that contains `backend/`, `src/`, and `data/`).

**Commands below** must be run from the folder that contains `backend/`, `src/`, and `scripts/` (the inner app folder). If your `requirements.txt` lives one level up, install from there first, then `cd` into the inner folder for setup and servers.

From the **inner** `healthcare_risk_pridiction-main` folder (the one that contains `backend/main.py`):

### 1. Virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build database and models

```bash
python scripts/setup_project.py
```

This creates:

- `models/diabetes_logistic.pkl` and `models/heart_logistic.pkl`
- `healthcare_risk.db` with tables `diabetes_patients` and `heart_patients`

### 3. Start the API

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010
```

**Windows (two windows at once):** from the inner app folder, run `.\run_app.ps1` — it starts the API on port **8010** and Streamlit in separate terminals (uses the parent folder’s `.venv` if present).

Open interactive docs: [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)

**Windows:** If you see `WinError 10013` on port 8000, Windows often reserves that range — this project defaults to **8010** instead.

### 4. Start the dashboard (second terminal)

```bash
streamlit run streamlit_app.py
```

If the API runs on another host/port, set the **API base URL** in the Streamlit sidebar or use:

```bash
set HEALTHCARE_API_URL=http://127.0.0.1:8010
streamlit run streamlit_app.py
```

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict-diabetes` | JSON body with diabetes features → probability + `low`/`medium`/`high` |
| `POST` | `/predict-heart-risk` | JSON body with heart features → same |
| `GET` | `/risk-summary` | SQL-backed aggregates: distributions, high-risk samples, factor means, age/gender breakdown |

## Notes on gender filters

The **Pima diabetes CSV** does not include sex/gender. Age-wise breakdowns still work. The **heart** dataset includes `sex` (0/1); the dashboard uses it for filtering and SQL group-by.

## License / data

Use the datasets only in line with their original licenses and citations. This repository is a student demo.
