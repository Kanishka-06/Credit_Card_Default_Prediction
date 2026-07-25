import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model(
    "credit_default_ann_v2.keras"
)

scaler = joblib.load(
    "scaler_v2.pkl"
)

encoder = joblib.load(
    "encoder_v2.pkl"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

.title{
    text-align:center;
    color:#003366;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

div.stButton > button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

.metric-card{
    padding:20px;
    border-radius:12px;
    background:white;
}

</style>
""",unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    "<div class='title'>💳 Credit Card Default Prediction using ANN</div>",
    unsafe_allow_html=True
)

# st.markdown(
#     "<div class='subtitle'>Artificial Neural Network based Credit Risk Assessment System</div>",
#     unsafe_allow_html=True
# )

st.write("")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Model Information")

st.sidebar.info(
# """
# Model : Artificial Neural Network

# Input Features : 9

# Dataset :
# Taiwan Credit Card Default Dataset

# Framework :
# TensorFlow + Streamlit
# """
)

# ==========================================================
# CUSTOMER DETAILS
# ==========================================================

st.header("Customer Information")

left,right = st.columns(2)

with left:

    customer_name = st.text_input(
        "Customer Name"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(

        "Gender",

        [

            "Male",

            "Female"

        ]

    )

    education = st.selectbox(

        "Education",

        [

            "Graduate School",

            "University",

            "High School",

            "Others"

        ]

    )

    marriage = st.selectbox(

        "Marital Status",

        [

            "Married",

            "Single",

            "Others"

        ]

    )

with right:

    credit_limit = st.number_input(

        "Credit Limit ($)",

        min_value=10000,

        value=100000,

        step=5000

    )

    avg_bill = st.number_input(

        "Average Monthly Bill ($)",

        min_value=0,

        value=30000

    )

    avg_payment = st.number_input(

        "Average Monthly Payment ($)",

        min_value=0,

        value=25000

    )

    pay_status = st.selectbox(

        "Current Repayment Status (PAY_0)",

        [

            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8

        ]

    )

    max_delay = st.number_input(

        "Maximum Delay",

        min_value=-2,

        max_value=8,

        value=0

    )

predict = st.button(
    "Predict Credit Risk"
)
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # ------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------

    if customer_name.strip() == "":

        st.error("Please enter the customer name.")

        st.stop()

    if avg_payment > credit_limit:

        st.warning(
            "Average payment is greater than the credit limit. Please verify the values."
        )

    # ------------------------------------------------------
    # CONVERT CATEGORICAL VALUES
    # ------------------------------------------------------

    sex = 1 if gender == "Male" else 2

    education_mapping = {

        "Graduate School": 1,
        "University": 2,
        "High School": 3,
        "Others": 4

    }

    education = education_mapping[education]

    marriage_mapping = {

        "Married": 1,
        "Single": 2,
        "Others": 3

    }

    marriage = marriage_mapping[marriage]

    # ------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # ------------------------------------------------------

    input_df = pd.DataFrame({

        "LIMIT_BAL": [credit_limit],

        "SEX": [sex],

        "EDUCATION": [education],

        "MARRIAGE": [marriage],

        "AGE": [age],

        "AVG_BILL": [avg_bill],

        "AVG_PAYMENT": [avg_payment],

        "PAY_0": [pay_status],

        "MAX_DELAY": [max_delay]

    })

    # ------------------------------------------------------
    # FEATURE LISTS
    # ------------------------------------------------------

    numerical_features = [

        "LIMIT_BAL",

        "AGE",

        "AVG_BILL",

        "AVG_PAYMENT"

    ]

    ordinal_features = [

        "EDUCATION",

        "PAY_0",

        "MAX_DELAY"

    ]

    categorical_features = [

        "SEX",

        "MARRIAGE"

    ]

    # ------------------------------------------------------
    # SCALE NUMERICAL FEATURES
    # ------------------------------------------------------

    scaled_numeric = scaler.transform(

        input_df[numerical_features]

    )

    scaled_numeric_df = pd.DataFrame(

        scaled_numeric,

        columns=numerical_features

    )

    # ------------------------------------------------------
    # ORDINAL FEATURES
    # ------------------------------------------------------

    ordinal_df = input_df[
        ordinal_features
    ].reset_index(drop=True)

    # ------------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # ------------------------------------------------------

    encoded = encoder.transform(

        input_df[categorical_features]

    )

    encoded_df = pd.DataFrame(

        encoded,

        columns=encoder.get_feature_names_out(
            categorical_features
        )

    )

    # ------------------------------------------------------
    # FINAL MODEL INPUT
    # ------------------------------------------------------

    model_input = pd.concat(

        [

            scaled_numeric_df,

            ordinal_df,

            encoded_df

        ],

        axis=1

    )

    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------

    probability = float(

        model.predict(

            model_input,

            verbose=0

        )[0][0]

    )

    prediction = int(probability >= 0.50)

    # ------------------------------------------------------
    # RISK CATEGORY
    # ------------------------------------------------------

    if probability < 0.30:

        risk = "LOW"

        recommendation = "APPROVE APPLICATION"

    elif probability < 0.60:

        risk = "MEDIUM"

        recommendation = "MANUAL REVIEW"

    else:

        risk = "HIGH"

        recommendation = "REJECT APPLICATION"

    # ------------------------------------------------------
    # AI EXPLANATION
    # ------------------------------------------------------

    reasons = []

    if avg_bill > credit_limit * 0.80:

        reasons.append(
            "High credit utilization."
        )

    if avg_payment < avg_bill * 0.50:

        reasons.append(
            "Low payment compared to average monthly bill."
        )

    if pay_status >= 2:

        reasons.append(
            "Customer has repayment delays."
        )

    if max_delay >= 3:

        reasons.append(
            "Maximum repayment delay is high."
        )

    if len(reasons) == 0:

        reasons.append(
            "Healthy repayment behaviour."
        )
    # ==========================================================
    # RESULTS
    # ==========================================================

    st.write("")
    st.markdown("---")
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Default Probability",

            f"{probability*100:.2f}%"

        )

    with col2:

        if risk == "LOW":

            st.success("🟢 LOW RISK")

        elif risk == "MEDIUM":

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")

    with col3:

        st.info(recommendation)

    st.write("")

    # ==========================================================
    # CUSTOMER SUMMARY
    # ==========================================================

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Profile")

        st.write(f"**Customer Name:** {customer_name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Education:** {education}")
        st.write(f"**Marital Status:** {marriage}")

    with right:

        st.subheader("Financial Information")

        st.write(f"**Credit Limit:** ${credit_limit:,.0f}")
        st.write(f"**Average Monthly Bill:** ${avg_bill:,.0f}")
        st.write(f"**Average Monthly Payment:** ${avg_payment:,.0f}")
        st.write(f"**Current Repayment Status:** {pay_status}")
        st.write(f"**Maximum Delay:** {max_delay}")

    st.markdown("---")

    # ==========================================================
    # AI EXPLANATION
    # ==========================================================

    st.subheader("AI Assessment")

    for item in reasons:

        st.write("•", item)

    st.markdown("---")

    # ==========================================================
    # FINANCIAL RATIOS
    # ==========================================================

    utilization = 0

    if credit_limit > 0:

        utilization = avg_bill / credit_limit

    payment_ratio = 0

    if avg_bill > 0:

        payment_ratio = avg_payment / avg_bill

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Credit Utilization",

            f"{utilization*100:.1f}%"

        )

    with col2:

        st.metric(

            "Payment / Bill Ratio",

            f"{payment_ratio:.2f}"

        )

    st.markdown("---")

    # ==========================================================
    # RECOMMENDATION
    # ==========================================================

    st.subheader("Loan Decision Recommendation")

    if risk == "LOW":

        st.success(
            """
            ✔ Customer shows healthy repayment behaviour.

            ✔ Credit risk is low.

            ✔ Application can be approved.
            """
        )

    elif risk == "MEDIUM":

        st.warning(
            """
            Customer presents moderate credit risk.

            Manual verification is recommended before approval.

            Review income documents and repayment capacity.
            """
        )

    else:

        st.error(
            """
            Customer has a high probability of default.

            Loan approval is not recommended without additional guarantees.

            Perform detailed financial assessment.
            """
        )
    # ==========================================================
    # RISK METER
    # ==========================================================

    st.markdown("---")

    st.subheader("Risk Probability")

    st.progress(float(probability))

    st.write(f"**Predicted Probability of Default:** {probability*100:.2f}%")

    # ==========================================================
    # MODEL INFORMATION
    # ==========================================================

#     with st.expander("Model Details"):

#         st.markdown("""
# ### Model Information

# **Model**
# - Artificial Neural Network (ANN)

# **Dataset**
# - Taiwan Credit Card Default Dataset

# **Input Features**
# 1. Credit Limit
# 2. Age
# 3. Gender
# 4. Education
# 5. Marital Status
# 6. Average Monthly Bill
# 7. Average Monthly Payment
# 8. Current Repayment Status (PAY_0)
# 9. Maximum Delay

# **Evaluation Metrics**

# - Accuracy : **81.53%**
# - Precision : **65.19%**
# - Recall : **67.25%**
# - F1 Score : **66.21%**
# - ROC-AUC : **87.67%**

# The model predicts the probability that a customer will default on the next month's credit card payment.
#         """)

#     # ==========================================================
#     # DOWNLOAD REPORT
#     # ==========================================================

#     report = pd.DataFrame({

#         "Customer Name":[customer_name],

#         "Age":[age],

#         "Gender":[gender],

#         "Education":[education],

#         "Marital Status":[marriage],

#         "Credit Limit":[credit_limit],

#         "Average Bill":[avg_bill],

#         "Average Payment":[avg_payment],

#         "PAY_0":[pay_status],

#         "Maximum Delay":[max_delay],

#         "Probability of Default":[round(probability*100,2)],

#         "Risk Level":[risk],

#         "Recommendation":[recommendation]

#     })

#     csv = report.to_csv(index=False)

#     st.download_button(

#         label="Download Prediction Report",

#         data=csv,

#         file_name="credit_default_prediction.csv",

#         mime="text/csv"

#     )

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.markdown("---")

    st.caption(
        "Credit Card Default Prediction System | Built using TensorFlow, Scikit-Learn and Streamlit"
    )
