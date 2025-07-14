from pathlib import Path

# Root of the repository (relative to this file)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data locations
RAW_DATA   = ROOT_DIR / "data" / "raw" / "adult 3.csv"
CLEAN_DATA = ROOT_DIR / "data" / "processed" / "employee_salary_final.csv"

# Model artefacts
MODEL_DIR  = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
BOOST_MODEL_PATH = MODEL_DIR / "model_boost.pkl"


RANDOM_SEED = 42
TEST_SIZE   = 0.2
