# 🛡️ Credit Card Fraud Detection System

An ML-powered web app to detect fraud with Explainable AI.

## 🚀 Features
- Real-time fraud prediction
- SHAP Explainability Visualizations
- Interactive Streamlit Dashboard

## 🛠️ Tech Stack
Python, Streamlit, Scikit-Learn, SHAP

## 📦 Installation
1. Clone repo
2. `pip install -r requirements.txt`
3. `python train_model.py`
4. `streamlit run app.py`


_**1. Overview**_

This project implements an intelligent, web-based system designed to detect and monitor fraudulent credit card transactions in real-time. Leveraging Machine Learning (ML) and Explainable AI (XAI), the system analyzes transaction patterns to classify them as "Legitimate" or "Fraudulent." Unlike traditional black-box models, this system provides transparent insights into why a transaction was flagged, enabling financial analysts to make informed decisions quickly.

_**2. Problem Statement**_

Rising Fraud: Credit card fraud causes billions in losses annually, growing with digital payment adoption.
Manual Limitations: Human analysts cannot manually verify millions of transactions daily.
Trust Gap: Standard ML models lack interpretability, making analysts hesitant to trust automated decisions.
Data Imbalance: Fraud cases are rare (0.17% of data), making accurate detection challenging.

_**3. Solution & Methodology**_

Machine Learning Engine: Uses a Random Forest Classifier trained on the Kaggle Credit Card Fraud dataset (284,807 transactions).
Imbalance Handling: Implements class_weight='balanced' to ensure the model learns from rare fraud cases effectively.
Explainable AI (SHAP): Integrates SHAP (SHapley Additive exPlanations) to visualize feature importance, showing exactly which factors (e.g., amount, time, anonymized variables) contributed to a fraud prediction.
Web Interface: Built with Streamlit, providing an interactive dashboard for single transaction prediction, batch analysis, and visual reporting.

_**4. Key Features**_


✅ Real-Time Prediction: Instant classification of transactions with probability scores.
✅ Explainability Dashboard: Visualizes global feature importance and individual transaction contributions (Red/Blue impact bars).
✅ Batch Analysis: Upload or sample datasets for bulk fraud monitoring.
✅ User-Friendly UI: Clean, responsive web interface accessible via browser (localhost or cloud).
✅ Model Serialization: Saves trained models (.pkl) for fast loading without retraining.

_**5. Technology Stack**_


Component
Technology
Language
Python 3.8+
Web Framework
Streamlit
Machine Learning
Scikit-Learn (Random Forest)
Data Processing
Pandas, NumPy
Explainability
SHAP (SHapley Additive exPlanations)
Visualization
Matplotlib, Seaborn
Deployment
Streamlit Cloud / Localhost

_**6. Project Outcome**_

Accuracy: Achieved 99.9% accuracy with high precision (95.2%) and recall (89.1%) on test data.
Transparency: Successfully demystifies AI decisions using SHAP force plots and bar charts.
Deployability: Fully containerized and ready for cloud deployment with a requirements.txt file.
Scalability: Architecture supports integration with real-time data streams (e.g., Kafka) for future production use.
