import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

def main():
    print("🔄 Loading Data...")
    try:
        df = pd.read_csv('data/creditcard.csv')
    except FileNotFoundError:
        print("❌ Error: creditcard.csv not found in 'data/' folder.")
        return

    # Separate features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale the data (Crucial for ML performance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🚀 Training Random Forest Model...")
    # class_weight='balanced' is key for fraud detection
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    print("\n📊 Model Performance Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ Success! Model and Scaler saved in 'models/' folder.")

if __name__ == '__main__':
    main()