<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Salary%20Prediction-forestgreen?style=for-the-badge&logo=python" alt="ML Badge"/>
</p>

# Employee Salary Prediction 🧑‍💼💰

Predict whether an individual’s annual income exceeds **$50 K** using demographic and job‑related attributes.  
Built with **Python · scikit‑learn · Streamlit**.

<div align="center">
  <img src="docs/roc_curve.png" width="450" alt="ROC curve">
</div>

---

## 📂 Project Structure

employee-salary-prediction/
├── data/
│   ├── raw/                         # Original CSV dataset(s)
│   └── processed/                   # Cleaned & filtered CSVs
│
├── notebooks/
│   └── eda_preprocessing.ipynb      # Exploratory data analysis + preprocessing
│   └── model_training.ipynb         # Model training, evaluation, tuning
│
├── src/                             # All project source code
│   ├── __init__.py
│   ├── config.py                    # Configuration variables (e.g., paths, params)
│   ├── preprocess.py                # Functions for cleaning, encoding, scaling
│   ├── train.py                     # Script to train and save model
│   ├── evaluate.py                  # Model evaluation & metrics
│   ├── predict.py                   # Single-record prediction logic
│   └── utils.py                     # Utility/helper functions
│
├── models/
│   ├── model.pkl                    # Saved trained model
│   └── scaler.pkl                   # Saved scaler/encoder (optional)
│
├── app/                             # Optional Streamlit or Flask app
│   ├── app.py                       # Frontend interface for predictions
│   └── templates/                   # HTML (for Flask) or config (for Streamlit)
│
├── tests/
│   └── test_model.py                # Unit tests for model logic
│
├── requirements.txt                 # List of Python dependencies
├── README.md                        # Project overview & instructions
└── .gitignore                       # Files to ignore in version control


---

## 🚀 Quick Start

```bash
# 1. Clone + create venv
git clone https://github.com/<your‑github>/employee-salary-prediction.git
cd employee-salary-prediction
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Reproduce data + model (optional)
python -m src.preprocess      # cleans raw CSV → data/processed/employee_salary_final.csv
python -m src.train_boost     # trains HistGB → models/model_boost.pkl

# 4. Launch the Streamlit app
streamlit run app/app.py


🧪 Model Performance

Model	ROC‑AUC	F1
Logistic Regression	0.84	0.60
HistGradientBoosting	0.89	0.66

Scores obtained on a 20 % hold‑out test split.

📈 Dataset
UCI Adult Income dataset (48 842 rows → 44 k after cleaning).
Features used:

Numeric	Categorical
age, educational‑num, hours‑per‑week, capital‑gain, capital‑loss	workclass, occupation
