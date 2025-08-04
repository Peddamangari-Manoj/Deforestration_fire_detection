import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
def b():
    # Load the combined CSV (example path)
    df1 = pd.read_csv('modis_2021_India.csv')
    df2 = pd.read_csv('modis_2022_India.csv')
    df3 = pd.read_csv('modis_2023_India.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert date and extract hour
    df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df['hour'] = df['acq_time'].astype(str).str[:2].astype(int, errors='ignore')
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['daynight', 'satellite', 'instrument'], drop_first=True)
    
    # Remove outliers using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    
    numerical_cols = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
    for col in numerical_cols:
        df = remove_outliers_iqr(df, col)
    
    # Features and target
    features = numerical_cols
    target = 'type'
    
    X = df[features]
    y = df[target]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fire_type_classifier_rf.pkl")
    joblib.dump(scaler, "models/fire_type_scaler.pkl")
    
    print("✅ Model and scaler saved to 'models/'")


def a():
    import joblib
    import numpy as np
    
    # Load the scaler and model
    scaler = joblib.load("models/fire_type_scaler.pkl")
    model = joblib.load("models/fire_type_classifier_rf.pkl")
    
    # Example input (replace with your real values)
    input_data = {
        "brightness": 340.5,
        "scan": 1.2,
        "track": 1.1,
        "confidence": 85,
        "bright_t31": 295.3,
        "frp": 20.1
    }
    
    # Prepare data for prediction
    X_new = np.array([[
        input_data["brightness"],
        input_data["scan"],
        input_data["track"],
        input_data["confidence"],
        input_data["bright_t31"],
        input_data["frp"]
    ]])
    
    # Scale and predict
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    
    print("🔥 Predicted Fire Type:", prediction[0])
a()