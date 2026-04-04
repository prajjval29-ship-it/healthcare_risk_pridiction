"""
Minimal data loader — useful for quick exploration in a notebook or REPL.

Example:
  from src.load_data import load_both
  diabetes_df, heart_df = load_both()
  print(diabetes_df.shape, heart_df.shape)
"""
from __future__ import annotations

import pandas as pd

from src.config import DIABETES_CSV, HEART_CSV


def load_both() -> tuple[pd.DataFrame, pd.DataFrame]:
    diabetes = pd.read_csv(DIABETES_CSV)
    heart = pd.read_csv(HEART_CSV)
    return diabetes, heart


if __name__ == "__main__":
    d, h = load_both()
    print("Diabetes:", d.shape)
    print("Heart:", h.shape)
    print(d.head())
    print(h.head())
