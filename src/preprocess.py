"""
Pre‑processing pipeline for the Employee Salary Prediction project.

Usage:
    python -m src.preprocess
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from .config import RAW_DATA, CLEAN_DATA, MODEL_DIR

# ────────────────────────────────────────────────
FEATURES_TO_KEEP = [
    "age", "workclass", "educational-num",
    "occupation", "hours-per-week",
    "capital-gain", "capital-loss", "income"  # target
]

AGE_MIN, AGE_MAX = 18, 70


def load_data(path: Path) -> pd.DataFrame:
    """Read CSV and replace '?' with NaN."""
    df = pd.read_csv(path)
    return df.replace("?", pd.NA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a trimmed DataFrame."""
    # Drop rows with any missing values
    df = df.dropna()

    # Age filter
    df = df[(df["age"].astype(int) >= AGE_MIN) & (df["age"].astype(int) <= AGE_MAX)]

    # Keep only selected columns
    df = df[FEATURES_TO_KEEP]

    return df.reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved cleaned data to {path}.")


def main() -> None:
    df_raw = load_data(RAW_DATA)
    df_clean = clean(df_raw)
    save(df_clean, CLEAN_DATA)

    # (Optional) Fit & store scaler for numeric columns
    numeric_cols = ["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss"]
    scaler = StandardScaler().fit(df_clean[numeric_cols])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("✅ Scaler fitted & saved.")


if __name__ == "__main__":
    main()
