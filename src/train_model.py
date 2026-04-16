# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('PerformanceScore', axis=1)
    y = df['PerformanceScore']
    
    # ✅ FIX: Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Save encoder for later use
    joblib.dump(le, '../models/label_encoder.pkl')
    
    return X, y


def train_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    }
    
    params = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1]
        }
    }
    
    best_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        gs = GridSearchCV(
            model,
            params[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'  # helps debugging
        )
        
        gs.fit(X_train, y_train)
        
        best_models[name] = gs.best_estimator_
        
        print(f"Best params for {name}: {gs.best_params_}")
        print(f"Best CV accuracy: {gs.best_score_:.4f}")
    
    return best_models


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.savefig(f'../outputs/figures/confusion_matrix_{model_name}.png')
    plt.close()
    
    return acc


def save_best_model(models, X, y):
    """Select best model based on full data"""
    
    best_model = models['XGBoost']
    
    # Retrain on full dataset
    best_model.fit(X, y)
    
    joblib.dump(best_model, '../models/xgb_model.pkl')
    
    print("✅ Best model (XGBoost) saved to models/xgb_model.pkl")
    
    return best_model


if __name__ == "__main__":
    
    X, y = load_processed_data('../data/processed/employee_processed.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # ✅ keeps class balance
    )
    
    models = train_models(X_train, y_train)
    
    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)
    
    best_model = save_best_model(models, X, y)