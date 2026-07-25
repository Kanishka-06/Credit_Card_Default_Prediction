# Credit Card Default Prediction using Artificial Neural Network

End-to-End Machine Learning Application for Credit Risk Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Community%20Cloud-brightgreen)

---

## Project Overview

Credit card default prediction is an important financial risk problem where identifying customers who are likely to default can support better-informed credit decisions.

This project develops an **Artificial Neural Network (ANN)** to predict the probability of a customer's next-month credit card payment default using demographic, credit, billing, and repayment behavior.

The trained model is integrated into an interactive **Streamlit application**, allowing users to enter customer information and receive a predicted default probability along with a corresponding risk classification.

---

## Objectives

- Predict the probability of credit card payment default.
- Engineer meaningful features from billing and repayment behavior.
- Apply appropriate encoding and scaling to model inputs.
- Train and evaluate an Artificial Neural Network.
- Provide probability-based **Low, Medium, and High Risk** classification.
- Build and deploy an interactive machine learning application.

---

## Machine Learning Workflow

**Data → Feature Engineering → Encoding → Scaling → ANN → Default Probability → Risk Classification**

---

## Key Features

The model uses customer demographic, credit, billing, and repayment-related information, including:

- Credit Limit
- Age
- Education
- Marital Status
- Average Billing Amount
- Average Payment Amount
- Recent Repayment Status
- Maximum Repayment Delay

These features are processed and transformed before being passed to the trained Artificial Neural Network.

---

## Model & Technology Stack

| Component | Technology |
|---|---|
| Programming | Python 3.11 |
| Model | Artificial Neural Network |
| Deep Learning | TensorFlow / Keras |
| Preprocessing | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib, Keras |
| Web Application | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Application

The Streamlit application provides a simple and interactive interface where users can enter customer information and generate a credit default prediction.

The application processes the entered information using the same preprocessing pipeline used during model training and passes the transformed data to the trained ANN model.

---

## Output

### Default Probability

The application provides an estimated probability of the customer defaulting on their next credit card payment.

### Risk Classification

The predicted probability is converted into an easy-to-understand risk category:

🟢 **Low Risk**

🟡 **Medium Risk**

🔴 **High Risk**

This classification helps users interpret the model's prediction in a more practical manner.

---

## Deployment

The application is deployed using **Streamlit Community Cloud**, making the trained machine learning model accessible through a web browser without requiring users to set up a local machine learning environment.

### 🔗 Live Demo

https://creditcarddefaults.streamlit.app/

---

## Future Improvements

- Improve risk-threshold calibration.
- Perform additional hyperparameter tuning and model optimization.
- Add SHAP-based model explainability.
- Improve handling of class imbalance.
- Add prediction history and analytics.
- Introduce enhanced model monitoring.
- Explore advanced neural network architectures.
- Improve production deployment and scalability.

---

## Risk Classification

🟢 **Low Risk**

🟡 **Medium Risk**

🔴 **High Risk**
