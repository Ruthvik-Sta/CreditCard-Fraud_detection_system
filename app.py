import os
import gdown

# Auto-download dataset if missing
DATA_URL = "https://drive.google.com/uc?id=1POdXbR2vL7Gz8VqQqJqKqJqKqJqKqJqK"  # Upload your CSV to Google Drive first

if not os.path.exists('data/creditcard.csv'):
    os.makedirs('data', exist_ok=True)
    try:
        import gdown
        gdown.download(DATA_URL, 'data/creditcard.csv', quiet=False)
    except:
        st.warning("⚠️ Please manually place creditcard.csv in data/ folder")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection AI", layout="wide")
st.title("🛡️ Credit Card Fraud Detection & Monitoring System")
st.markdown("### Powered by Machine Learning & Explainable AI")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fraud_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

if model is None:
    st.error("❌ Model not found. Please run 'train_model.py' first.")
    st.stop()
else:
    st.success("✅ System Ready: Model Loaded")

option = st.sidebar.selectbox("Choose Action", ["Single Transaction Demo", "Batch Analysis & XAI"])

if option == "Single Transaction Demo":
    st.header("🔍 Analyze Transaction")
    st.info("⚠️ Sampling from dataset for accurate prediction.")
    
    if st.button("Pick Random Transaction & Predict"):
        df = pd.read_csv('data/creditcard.csv')
        sample = df.sample(1)
        X_sample = sample.drop('Class', axis=1)
        y_actual = sample['Class'].values[0]
        X_sample_scaled = scaler.transform(X_sample)
        
        prediction = model.predict(X_sample_scaled)[0]
        prob = model.predict_proba(X_sample_scaled)[0][1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actual Class", "FRAUD" if y_actual == 1 else "LEGIT")
        with col2:
            st.metric("AI Prediction", "FRAUD" if prediction == 1 else "LEGIT")
        st.metric("Fraud Probability", f"{prob:.2%}")
        
        if prediction == 1:
            st.error("🚨 Alert: High Risk Transaction Detected!")
        else:
            st.success("✅ Transaction Safe.")

elif option == "Batch Analysis & XAI":
    st.header("📊 Dataset Overview & Explainability")
    df = pd.read_csv('data/creditcard.csv')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{df.shape[0]:,}")
    col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
    col3.metric("Fraud Rate", f"{(df['Class'].sum()/df.shape[0]):.2%}")
    
    st.subheader("🔮 Explainable AI (SHAP) Analysis")
    
    if st.button("Analyze Top Fraud Cases"):
        fraud_cases = df[df['Class'] == 1].sample(5, random_state=42)
        X_explain = fraud_cases.drop('Class', axis=1).reset_index(drop=True)
        X_explain_scaled = scaler.transform(X_explain)
        
        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain_scaled)
        
        # Handle SHAP output format
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        
        # Ensure shap_vals is 2D (n_samples, n_features)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # Take positive class for binary classification
        
        st.success("✅ SHAP Analysis Complete")
        
        # === PLOT 1: Feature Importance ===
        st.markdown("#### 1. Global Feature Importance")
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        mean_abs_shap = np.array(mean_abs_shap).flatten()  # Ensure 1D
        top_features = pd.DataFrame({
            'Feature': X_explain.columns,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False).head(10)
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.barh(range(len(top_features)), top_features['Importance'].values)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'].values)
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('Top 10 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
        
        # === PLOT 2: SHAP Values for First Transaction ===
        st.markdown("#### 2. Why was this transaction flagged?")
        st.caption("Red = pushes toward FRAUD, Blue = pushes toward LEGIT")
        
        # Get first transaction data
        shap_row = shap_vals[0]
        feature_row = X_explain.iloc[0].values
        
        # Create DataFrame
        shap_df = pd.DataFrame({
            'Feature': X_explain.columns,
            'SHAP_Value': shap_row,
            'Feature_Value': feature_row
        })
        shap_df['Abs_SHAP'] = np.abs(shap_df['SHAP_Value'])
        shap_df = shap_df.sort_values('Abs_SHAP', ascending=False).head(10)
        
        # Create bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors = ['red' if v > 0 else 'blue' for v in shap_df['SHAP_Value']]
        ax2.barh(shap_df['Feature'], shap_df['SHAP_Value'], color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('SHAP Value (Impact on Prediction)')
        ax2.set_title('Feature Contributions to Fraud Prediction')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        
        # Show table
        st.markdown("**Detailed Feature Analysis:**")
        display_df = shap_df[['Feature', 'Feature_Value', 'SHAP_Value']].copy()
        st.dataframe(display_df.style.format({
            'Feature_Value': '{:.4f}',
            'SHAP_Value': '{:.4f}'
        }))

st.markdown("---")
st.markdown("© 2024 Internship Project | Built with Streamlit & Scikit-Learn")
