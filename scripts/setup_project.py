"""
One-shot setup: train models (if needed) + build SQLite database.

From project root (healthcare_risk_pridiction-main):

    python scripts/setup_project.py

Interview line: "I preprocess CSVs, fit logistic regression, score every row,
and load everything into SQLite so the API can run fast SQL analytics."
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import init_database  # noqa: E402
from src.ml_models import train_and_save_all  # noqa: E402


def main() -> None:
    print("Training / refreshing sklearn models...")
    train_and_save_all()
    print("Building SQLite database...")
    init_database()
    print("Done. Run API:")
    print("  python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010")
    print("Then dashboard:")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
