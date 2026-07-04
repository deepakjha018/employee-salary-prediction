import streamlit as st
import pandas as pd
import sys
from pathlib import Path


# -------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from src.utils import load_model, FEATURE_ORDER, predict_sample



# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)



# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

MODEL_PATH = ROOT_DIR / "models" / "model_boost_notebook.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.stop()


model = load_model(MODEL_PATH)



# -------------------------------------------------------
# Custom Styling
# -------------------------------------------------------

st.markdown(
    """
<style>

.main-title {
    font-size:45px;
    font-weight:800;
}

.subtitle {
    font-size:18px;
    color:#808080;
}

.card {
    padding:20px;
    border-radius:15px;
    background-color:#262730;
    margin-bottom:20px;
}


.footer {
    text-align:center;
    color:gray;
}

</style>

""",
    unsafe_allow_html=True
)



# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("💼 About Project")

    st.write(
    """
    This Machine Learning application predicts whether an employee's
    annual income is above or below **$50K**.
    
    Developed during:

    **IBM SkillsBuild & Edunet Foundation Internship**
    """
    )


    st.divider()


    st.subheader("ML Pipeline")

    st.write(
    """
    ✔ Data Cleaning  
    ✔ Feature Engineering  
    ✔ Model Training  
    ✔ Evaluation  
    ✔ Deployment  
    """
    )


    st.divider()


    st.subheader("Model")

    st.success(
        """
        Algorithm:
        
        Histogram Gradient Boosting
        
        ROC-AUC: 0.89
        """
    )





# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.markdown(
"""
<div class='main-title'>
💼 Employee Salary Prediction
</div>

<div class='subtitle'>
Predict employee income category using Machine Learning
</div>

""",
unsafe_allow_html=True
)


st.write("")


# -------------------------------------------------------
# Metric Cards
# -------------------------------------------------------

c1,c2,c3 = st.columns(3)

with c1:
    st.metric(
        "Model",
        "Gradient Boosting"
    )

with c2:
    st.metric(
        "ROC-AUC",
        "0.89"
    )

with c3:
    st.metric(
        "Prediction Type",
        "Binary"
    )



st.divider()



# -------------------------------------------------------
# Tabs
# -------------------------------------------------------

tab1, tab2 = st.tabs(
    [
        "🧑 Single Prediction",
        "📂 Batch Prediction"
    ]
)




WORKCLASS_OPTIONS = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay"
]


OCCUPATION_OPTIONS=[
    "Prof-specialty",
    "Exec-managerial",
    "Adm-clerical",
    "Sales",
    "Craft-repair",
    "Machine-op-inspct",
    "Tech-support",
    "Other-service",
    "Transport-moving"
]



# -------------------------------------------------------
# Single Prediction
# -------------------------------------------------------


with tab1:

    st.subheader("Enter Employee Details")


    col1,col2 = st.columns(2)


    with col1:

        age=st.number_input(
            "Age",
            18,
            70,
            30
        )

        workclass=st.selectbox(
            "Workclass",
            WORKCLASS_OPTIONS
        )

        education=st.number_input(
            "Education Years",
            1,
            16,
            10
        )


    with col2:

        occupation=st.selectbox(
            "Occupation",
            OCCUPATION_OPTIONS
        )

        hours=st.number_input(
            "Hours Per Week",
            1,
            100,
            40
        )

        gain=st.number_input(
            "Capital Gain",
            value=0
        )

        loss=st.number_input(
            "Capital Loss",
            value=0
        )



    if st.button(
        "🔮 Predict Salary",
        type="primary"
    ):


        record={
            "age":age,
            "workclass":workclass,
            "educational-num":education,
            "occupation":occupation,
            "hours-per-week":hours,
            "capital-gain":gain,
            "capital-loss":loss
        }



        result=predict_sample(
            record,
            model_path=MODEL_PATH
        )


        prob_high = result["probability"]


        if result["label"] == ">50K":

            confidence = prob_high

            st.success(
                f"""
                🎉 Prediction: {result['label']}

                Model Confidence: {confidence:.2%}

                Probability of earning >50K: {prob_high:.2%}
                """
            )


        else:

            confidence = 1 - prob_high

            st.info(
                f"""
                Prediction: {result['label']}

                Model Confidence: {confidence:.2%}

                Probability of earning >50K: {prob_high:.2%}
                """
            )





# -------------------------------------------------------
# Batch Prediction
# -------------------------------------------------------


with tab2:


    st.subheader(
        "Upload CSV File"
    )


    file=st.file_uploader(
        "Choose CSV",
        type="csv"
    )


    if file:

        data=pd.read_csv(file)


        missing=[
            x for x in FEATURE_ORDER
            if x not in data.columns
        ]


        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

        else:

            prediction=model.predict(data)

            data["Prediction"]=[
                ">50K" if x==1 else "<=50K"
                for x in prediction
            ]


            st.dataframe(data)


            csv=data.to_csv(index=False)


            st.download_button(
                "Download Results",
                csv,
                "salary_prediction.csv"
            )




# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.divider()

st.markdown(
"""
<div class='footer'>

Developed by Deepak Kumar Jha  
IBM SkillsBuild × Edunet Foundation Internship

</div>

""",
unsafe_allow_html=True
)
