"""
General‑purpose helpers for the Employee Salary Prediction project.
"""
from pathlib import Path
import joblib, pandas as pd
from typing import Dict, Union

from .config import MODEL_DIR

# Default to the best model we've trained so far
DEFAULT_MODEL = MODEL_DIR / "model_boost.pkl"


# ─────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────
def load_model(path: Union[str, Path, None] = None):
    """Load a joblib‑serialized sklearn Pipeline."""
    if path is None:  # ← NEW: fall back to default
        path = DEFAULT_MODEL
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)



# ─────────────────────────────────────────────────────────────
# Single‑record formatting
# ─────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "age",
    "workclass",
    "educational-num",
    "occupation",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]


def format_single(sample: Dict) -> pd.DataFrame:
    """
    Convert one Python dict into a 1‑row DataFrame with the
    exact columns in FEATURE_ORDER.

    Missing keys raise KeyError for safety.
    """
    missing = [k for k in FEATURE_ORDER if k not in sample]
    if missing:
        raise KeyError(f"Missing keys in input sample: {missing}")

    return pd.DataFrame([{k: sample[k] for k in FEATURE_ORDER}])


# ─────────────────────────────────────────────────────────────
# End‑to‑end convenience
# ─────────────────────────────────────────────────────────────
def predict_sample(sample: Dict, model_path: Union[str, Path] = DEFAULT_MODEL):
    """
    Predict income label and probability for a single record.

    Returns dict → {"label": "...", "probability": 0.87}
    """
    model = load_model(model_path)
    X = format_single(sample)
    proba = model.predict_proba(X)[0, 1]  # probability of >50K
    label = ">50K" if proba >= 0.5 else "<=50K"
    return {"label": label, "probability": float(proba)}
