"""
Project paths — resolve files no matter where you run commands from.

Interview tip: using a single PROJECT_ROOT avoids broken paths when
`python` is started from different folders.
"""
from pathlib import Path

# healthcare_risk_pridiction-main/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "healthcare_risk.db"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

DIABETES_CSV = DATA_DIR / "diabetes.csv"
HEART_CSV = DATA_DIR / "Heart_Disease_Prediction.csv"

DIABETES_MODEL_PATH = MODELS_DIR / "diabetes_logistic.pkl"
HEART_MODEL_PATH = MODELS_DIR / "heart_logistic.pkl"
