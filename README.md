# 🚀 Employee Performance Predictor using Data Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An end-to-end Machine Learning project to predict employee performance and drive HR decisions.**  
*Developed under the **IIP Program** at **IIT Delhi**.*

![Streamlit Dashboard](images/dashboard_main.png)

---

## 📌 Project Overview

This project simulates a real-world HR analytics scenario. Using a synthetically generated dataset of 1500+ employees, we clean, analyze, and model the data to predict an employee's performance score (on a scale of 1 to 5). The final output is an **interactive Streamlit web application** that allows HR managers to input employee details and instantly receive a performance prediction along with interpretable insights.

## 🎯 Problem Statement

Traditional performance reviews often suffer from recency bias and subjective judgment. HR departments need a **data-driven tool** to:
- Identify high-potential employees for promotion.
- Flag employees who may need additional training or support.
- Allocate resources effectively for retention and development.

## 💼 Business Value

- **Objective Assessment:** Reduce bias in performance evaluations.
- **Proactive Retention:** Identify at-risk low performers before they disengage.
- **Cost Optimization:** Target training budgets where they matter most.
- **Succession Planning:** Build a pipeline of future leaders based on data.

## 🛠️ Tech Stack & Architecture

| Category | Tools & Libraries |
|----------|-------------------|
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Modeling** | Scikit-learn, XGBoost |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Frontend** | Streamlit |
| **Environment** | Python 3.9+, Virtual Env |

### System Architecture
![Architecture Diagram](images/architecture_flow.png)

## 📁 Folder Structure
Employee-Performance-Predictor/
│
├── data/ # All data files
│ ├── raw/ # Original synthetic CSV
│ └── processed/ # Cleaned & scaled data
│
├── notebooks/ # Jupyter exploration notebooks
│ ├── 01_data_generation.ipynb
│ ├── 02_eda_and_preprocessing.ipynb
│ └── 03_model_training.ipynb
│
├── src/ # Core Python scripts
│ ├── data_generation.py
│ ├── preprocessing.py
│ └── train_model.py
│
├── models/ # Serialized ML assets
│ ├── xgb_model.pkl
│ ├── scaler.pkl
│ └── label_encoders.pkl
│
├── app/ # Streamlit Web Application
│ └── app.py
│
├── outputs/ # Generated reports & figures
│ └── figures/
│
├── images/ # Screenshots for README
│
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [Your GitHub Link]
cd Employee-Performance-Predictor
2. Create Virtual Environment
Windows:

bash
python -m venv venv
venv\Scripts\activate
Mac/Linux:

bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
🏃 How to Run the Project
Step 1: Generate Synthetic HR Data
bash
cd src
python data_generation.py
Output: data/raw/employee_data.csv

Step 2: Preprocess Data & Train Model
bash
python preprocessing.py
python train_model.py
Output: Saved model files in models/ directory and evaluation metrics.

Step 3: Launch the Web Application
bash
cd ../app
streamlit run app.py
Open your browser and go to http://localhost:8501.

📊 Results & Model Performance
Model	Accuracy	Key Insight
Random Forest	86.2%	Good baseline
XGBoost (Selected)	89.4%	Best performance with tuned hyperparameters
Feature Importance (Top 5 Drivers of Performance)
Years of Experience

Training Hours (Last Year)

Absenteeism Rate

Salary per Year of Experience

Projects Completed

🧠 Future Improvements
Real Dataset Integration: Use IBM HR Analytics Employee Attrition dataset.

Deep Learning: Implement a Neural Network for complex pattern recognition.

Deployment: Containerize with Docker and deploy on Streamlit Cloud / AWS.

NLP Integration: Analyze text feedback from managers to augment prediction.

🙏 Acknowledgements
This project was completed as a capstone requirement for the IIP (Industry Immersion Program) in collaboration with IIT Delhi.

I would like to express my sincere gratitude to my mentor, Umesh Yadav, for their exceptional guidance, constant motivation, and technical insights throughout this project. Their mentorship was invaluable in bridging the gap between academic knowledge and industry application.

Thank you to the entire IIP team at IIT Delhi for curating such an enriching experience.

Made with ❤️ by *Hridoy Biswas *
