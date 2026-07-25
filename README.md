Credit Card Default Prediction using ANN

An end-to-end Machine Learning project that predicts the probability of credit card payment default using customer demographic, credit, billing, and repayment behavior.

The project uses an Artificial Neural Network (ANN) for prediction and provides an interactive Streamlit web application where users can enter customer details and receive a default probability and risk classification.

🎯 Key Features
Predicts the probability of credit card payment default
Uses customer credit, demographic, billing, and repayment information
Feature engineering using average billing, average payment, and repayment delay
Categorical encoding and numerical feature scaling
Artificial Neural Network built using TensorFlow/Keras
Probability-based prediction with Low, Medium, and High risk classification
Interactive Streamlit interface
Deployed using Streamlit Community Cloud
🧠 Machine Learning Workflow

Data → Preprocessing → Feature Engineering → ANN → Probability Prediction → Risk Classification

The model uses features such as:

Credit Limit
Age
Education
Marital Status
Average Billing Amount
Average Payment Amount
Recent Repayment Status
Maximum Repayment Delay

The preprocessing pipeline is preserved using saved encoder and scaler files to ensure that application inputs are processed consistently with the training data.

🛠️ Tech Stack
Category	Technologies
Language	Python
Machine Learning	TensorFlow, Keras, Scikit-learn
Data Processing	Pandas, NumPy
Model Persistence	Joblib, Keras
Web Application	Streamlit
Deployment	Streamlit Community Cloud
Development	Jupyter Notebook, GitHub
📂 Project Structure

app1.py — Streamlit application and prediction interface
credit_default_ann_v2.keras — Trained ANN model
encoder_v2.pkl — Saved categorical encoder
scaler_v2.pkl — Saved feature scaler
requirements.txt — Project dependencies
Credit_Card_Default_Prediction.ipynb — Model development and training notebook
🌐 Live Demo

Streamlit App:
[Add your deployed Streamlit URL here]

📊 Output

The application provides:

Default Probability — Estimated likelihood of the customer defaulting on their next payment.

Risk Level — Categorizes the prediction into:

🟢 Low Risk
🟡 Medium Risk
🔴 High Risk
