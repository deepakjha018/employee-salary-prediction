import pytest
from pathlib import Path
from src.utils import load_model, predict_sample

MODEL_PATH = Path("models/model_boost.pkl")

def test_model_file_exists():
    assert MODEL_PATH.exists(), "Model file not found. Train the model first."

def test_predict_sample_keys():
    sample = {
        "age": 35,
        "workclass": "Private",
        "educational-num": 13,
        "occupation": "Exec-managerial",
        "hours-per-week": 40,
        "capital-gain": 0,
        "capital-loss": 0,
    }
    result = predict_sample(sample, model_path=MODEL_PATH)
    assert "label" in result
    assert "probability" in result
    assert result["label"] in [">50K", "<=50K"]
    assert 0.0 <= result["probability"] <= 1.0
