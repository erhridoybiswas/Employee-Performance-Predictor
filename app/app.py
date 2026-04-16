# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load('../models/xgb_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    encoders = joblib.load('../models/label_encoders.pkl')
    # Load sample data for visualizations
    df = pd.read_csv('../data/processed/employee_processed.csv')
    return model, scaler, encoders, df

model, scaler, encoders, df = load_artifacts()

st.title("🚀 Employee Performance Predictor")
st.markdown("Predict employee performance category (1-5) and explore HR insights.")

# Sidebar navigation
page = st.sidebar.selectbox("Choose Mode", ["📊 Dashboard", "🔮 Predict Performance", "📈 Model Insights"])

if page == "📊 Dashboard":
    st.header("HR Analytics Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        avg_perf = df['PerformanceScore'].mean()
        st.metric("Avg Performance Score", f"{avg_perf:.2f}")
    with col3:
        high_perf = (df['PerformanceScore'] >= 4).sum()
        st.metric("High Performers (Score 4-5)", high_perf)
    
    st.subheader("Performance Distribution")
    fig = px.histogram(df, x='PerformanceScore', nbins=5, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Correlations with Performance")
    # Load raw data for better feature names (but we can use processed names)
    corr = df.corr()['PerformanceScore'].sort_values(ascending=False)
    fig2 = px.bar(x=corr.index, y=corr.values, labels={'x':'Feature', 'y':'Correlation'})
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔮 Predict Performance":
    st.header("Predict Employee Performance")
    st.write("Enter employee details to get performance prediction.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            gender = st.selectbox("Gender", options=['Male', 'Female'])
            marital = st.selectbox("Marital Status", options=['Single', 'Married', 'Divorced'])
            dept = st.selectbox("Department", options=['Sales', 'R&D', 'HR', 'Finance', 'Operations', 'IT'])
            education = st.selectbox("Education", options=['High School', 'Bachelor', 'Master', 'PhD'])
        with col2:
            exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
            salary = st.number_input("Salary (USD)", min_value=20000, max_value=200000, value=60000, step=1000)
            training = st.number_input("Training Hours (last year)", min_value=0, max_value=200, value=40)
            projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=8)
            absenteeism = st.number_input("Absenteeism (days)", min_value=0, max_value=30, value=2)
        
        submitted = st.form_submit_button("Predict Performance")
    
    if submitted:
        # Create input DataFrame
        input_dict = {
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital,
            'Department': dept,
            'Education': education,
            'YearsExperience': exp,
            'Salary': salary,
            'TrainingHours': training,
            'ProjectsCompleted': projects,
            'Absenteeism': absenteeism
        }
        input_df = pd.DataFrame([input_dict])
        
        # Encode categoricals
        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])
        
        # Feature engineering (same as training)
        input_df['ExpPerAge'] = input_df['YearsExperience'] / input_df['Age']
        input_df['TrainingPerProject'] = input_df['TrainingHours'] / (input_df['ProjectsCompleted'] + 1)
        input_df['SalaryPerExp'] = input_df['Salary'] / (input_df['YearsExperience'] + 1)
        input_df['AbsenteeismPerProject'] = input_df['Absenteeism'] / (input_df['ProjectsCompleted'] + 1)
        
        # Ensure column order matches training
        # Load a sample to get columns
        sample = pd.read_csv('../data/processed/employee_processed.csv').drop('PerformanceScore', axis=1).iloc[0:0]
        for col in sample.columns:
            if col not in input_df.columns:
                input_df[col] = 0  # placeholder (shouldn't happen)
        input_df = input_df[sample.columns]
        
        # Scale numericals
        num_cols = scaler.feature_names_in_  # for newer sklearn
        input_scaled = scaler.transform(input_df[num_cols])
        input_df_scaled = input_df.copy()
        input_df_scaled[num_cols] = input_scaled
        
        # Predict
        prediction = model.predict(input_df_scaled)[0]
        proba = model.predict_proba(input_df_scaled)[0]
        
        st.success(f"### Predicted Performance Score: **{prediction}**")
        st.write("Prediction Probabilities:")
        proba_df = pd.DataFrame({'Score': [1,2,3,4,5], 'Probability': proba})
        fig = px.bar(proba_df, x='Score', y='Probability', color='Probability')
        st.plotly_chart(fig, use_container_width=True)
        
        # SHAP explanation (if needed)
        if st.checkbox("Show Explanation (SHAP)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df_scaled)
            # For multi-class, take class for prediction
            shap_values_class = shap_values[:, :, prediction-1] if len(shap_values.shape)==3 else shap_values
            st.subheader("Feature Impact on Prediction")
            fig_shap, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=shap_values_class[0], 
                                                  base_values=explainer.expected_value[prediction-1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                                  data=input_df_scaled.iloc[0],
                                                  feature_names=input_df_scaled.columns), show=False)
            st.pyplot(fig_shap)

elif page == "📈 Model Insights":
    st.header("Model Performance & Feature Importance")
    
    # Feature importance
    importance = model.feature_importances_
    feature_names = df.drop('PerformanceScore', axis=1).columns
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False)
    fig = px.bar(fi_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importances")
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP summary plot (global)
    st.subheader("SHAP Summary (Global Impact)")
    # Sample 200 rows for SHAP due to performance
    X_sample = df.drop('PerformanceScore', axis=1).sample(200, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # For multi-class, average over classes or pick one
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
    st.caption("Average impact on model output magnitude")