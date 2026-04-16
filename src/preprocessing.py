# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    
    # Drop EmployeeID (not a feature)
    df.drop('EmployeeID', axis=1, inplace=True)
    
    # Handle missing values: fill TrainingHours with median by department
    df['TrainingHours'] = df.groupby('Department')['TrainingHours'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    print(f"Cleaned shape: {df.shape}")
    return df

def encode_categoricals(df, save_encoders=True):
    """Label encode categorical columns."""
    cat_cols = ['Gender', 'MaritalStatus', 'Department', 'Education']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    if save_encoders:
        joblib.dump(encoders, '../models/label_encoders.pkl')
    return df, encoders

def feature_engineering(df):
    """Create new features."""
    # Experience to Age ratio (productivity indicator)
    df['ExpPerAge'] = df['YearsExperience'] / df['Age']
    
    # Training effectiveness: TrainingHours per project
    df['TrainingPerProject'] = df['TrainingHours'] / (df['ProjectsCompleted'] + 1)
    
    # Salary per year of experience (career progression)
    df['SalaryPerExp'] = df['Salary'] / (df['YearsExperience'] + 1)
    
    # Absenteeism rate (per project)
    df['AbsenteeismPerProject'] = df['Absenteeism'] / (df['ProjectsCompleted'] + 1)
    
    return df

def scale_features(X, scaler=None, fit=True):
    """Scale numerical features."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[num_cols])
        joblib.dump(scaler, '../models/scaler.pkl')
    else:
        X_scaled = scaler.transform(X[num_cols])
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)
    # Keep categorical columns as is (already encoded)
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    X_final = pd.concat([X_scaled_df, X[cat_cols]], axis=1)
    return X_final, scaler

def prepare_data(raw_path, processed_path):
    """Complete preprocessing pipeline."""
    df = load_and_clean_data(raw_path)
    df, encoders = encode_categoricals(df)
    df = feature_engineering(df)
    
    # Separate features and target
    X = df.drop('PerformanceScore', axis=1)
    y = df['PerformanceScore']
    
    X_scaled, scaler = scale_features(X)
    
    # Save processed data
    processed_df = pd.concat([X_scaled, y], axis=1)
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
    return X_scaled, y

if __name__ == "__main__":
    raw_path = '../data/raw/employee_data.csv'
    processed_path = '../data/processed/employee_processed.csv'
    X, y = prepare_data(raw_path, processed_path)