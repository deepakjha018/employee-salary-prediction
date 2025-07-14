import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Make project modules importable when running `streamlit run app/app.py`
# We add the repository root to sys.path and then import via the package name.
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.utils import load_model, FEATURE_ORDER, predict_sample  # noqa: E402

# Streamlit page config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")

st.title("💼 Employee Salary Prediction (>50K)")

# ---------------------------------------------------------------------
# Locate model file safely
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "model_boost_notebook.pkl"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_NAME

if not MODEL_PATH.exists():
    candidates = list(MODEL_DIR.glob("*.pkl"))
    if candidates:
        MODEL_PATH = candidates[0]
    else:
        st.error("❌ No model .pkl file found in the 'models' directory.\n"
                 "Please run `python -m src.train_boost` or `src.train` to generate a model.")
        st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This Streamlit app uses a Gradient‑Boosting model trained on the Adult Income dataset "
    "to predict whether an employee is likely to earn more than **$50K** per year."
)

st.sidebar.subheader("How it works")
st.sidebar.markdown(
    "- Fill in the form for a **single prediction**, or\n"
    "- Upload a **CSV** with the same feature columns for **batch predictions**.\n"
    "\nThe model outputs the probability of earning >50K and a final label."
)

# ---------------------------------------------------------------------
# Tabs: Single vs Batch
# ---------------------------------------------------------------------
tabs = st.tabs(["🧍 Single Prediction", "📂 Batch Prediction"])

# Pre‑defined options for select boxes
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]
OCCUPATION_OPTIONS = [
    "Prof-specialty", "Exec-managerial", "Adm-clerical", "Sales",
    "Machine-op-inspct", "Craft-repair", "Transport-moving",
    "Handlers-cleaners", "Other-service", "Farming-fishing",
    "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
]

# ---------------------------------------------------------------------
# 1️⃣ Single Prediction tab
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("Input employee attributes")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS, index=0)
        education_num = st.number_input("Years of Education (1‑16)", min_value=1, max_value=16, value=10, step=1)
    with col2:
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=0)
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, step=1)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0, step=100)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0, step=100)

    if st.button("Predict Salary Class", type="primary"):
        record = {
            "age": age,
            "workclass": workclass,
            "educational-num": education_num,
            "occupation": occupation,
            "hours-per-week": hours_per_week,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
        }
        pred = predict_sample(record, model_path=MODEL_PATH)
        st.success(f"**Prediction:** {pred['label']}  |  **Probability:** {pred['probability']:.2%}")

# ---------------------------------------------------------------------
# 2️⃣ Batch Prediction tab
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Upload CSV for batch prediction")
    st.markdown(
        "The CSV must contain these columns, in any order: "
        f"`{', '.join(FEATURE_ORDER)}`."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURE_ORDER if c not in df_input.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
        else:
            proba = model.predict_proba(df_input)[:, 1]
            labels = (proba >= 0.5).astype(int).map({0: "<=50K", 1: ">50K"})
            df_out = df_input.copy()
            df_out["pred_probability"] = proba
            df_out["pred_label"] = labels

            st.success("Predictions generated!")
            st.dataframe(df_out.head())

            # Download result
            csv_out = df_out.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv_out,
                file_name="salary_predictions.csv",
                mime="text/csv",
            )
