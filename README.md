Credit Card Default Prediction using Artificial Neural Network
<p align="center"> <b>End-to-End Machine Learning Application for Credit Risk Prediction</b> </p>








Project Overview

Credit card default prediction is an important financial risk problem where identifying customers likely to default can support better-informed credit decisions.

This project develops an Artificial Neural Network (ANN) to predict the probability of a customer's next-month credit card payment default using demographic, credit, billing, and repayment behavior.

The trained model is integrated into an interactive Streamlit application, allowing users to enter customer information and receive a default probability and risk classification.

Objectives
Predict the probability of credit card payment default.
Engineer meaningful features from billing and repayment behavior.
Apply appropriate encoding and scaling to model inputs.
Train and evaluate an Artificial Neural Network.
Provide probability-based Low, Medium, and High Risk classification.
Build and deploy an interactive machine learning application.
Machine Learning Workflow

Data → Feature Engineering → Encoding → Scaling → ANN → Default Probability → Risk Classification

Key Features
Credit Limit
Age
Education
Marital Status
Average Billing Amount
Average Payment Amount
Recent Repayment Status
Maximum Repayment Delay
Model & Technology Stack
Component	Technology
Programming	Python 3.11
Model	Artificial Neural Network
Deep Learning	TensorFlow / Keras
Preprocessing	Scikit-learn
Data Processing	Pandas, NumPy
Model Persistence	Joblib, Keras
Web Application	Streamlit
Deployment	Streamlit Community Cloud
Application

The Streamlit application provides a simple interface for entering customer information and generating predictions.

Output

Default Probability
Estimated probability of the customer defaulting on their next payment.

Risk Classification
🟢 Low Risk
🟡 Medium Risk
🔴 High Risk

Deployment

The application is deployed using Streamlit Community Cloud, making the trained model accessible through a web browser without requiring a local machine learning environment.

🔗 Live Demo
https://creditcarddefaults.streamlit.app/
Future Improvements
Improve risk-threshold calibration.
Hyperparameter tuning and model optimization.
SHAP-based model explainability.
Improved handling of class imbalance.
Prediction history and analytics.
Enhanced model monitoring and production deployment.



🟢 Low Risk
🟡 Medium Risk
🔴 High Risk
