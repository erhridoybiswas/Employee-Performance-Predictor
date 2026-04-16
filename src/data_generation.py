# src/data_generation.py
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def generate_employee_data(n=1200):
    """Generate synthetic HR data with realistic relationships."""
    
    # -------------------- Demographics --------------------
    age = np.random.randint(22, 65, n)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45])
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced'], n, p=[0.4, 0.5, 0.1]
    )
    
    # -------------------- Job details --------------------
    departments = ['Sales', 'R&D', 'HR', 'Finance', 'Operations', 'IT']
    dept = np.random.choice(
        departments, n, p=[0.25, 0.15, 0.1, 0.15, 0.2, 0.15]
    )
    
    # -------------------- Education --------------------
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n,
        p=[0.1, 0.5, 0.3, 0.1]
    )
    
    edu_level = {
        'High School': 1,
        'Bachelor': 2,
        'Master': 3,
        'PhD': 4
    }
    
    # ✅ FIX: convert to NumPy array
    edu_num = np.array([edu_level[e] for e in education])
    
    # -------------------- Experience --------------------
    exp_years = []
    for a, e in zip(age, edu_num):
        start_work = 18 + (e - 1) * 2  # approximate start age
        exp = max(0, a - start_work - random.randint(0, 3))
        exp_years.append(exp)
    
    exp_years = np.array(exp_years)
    
    # -------------------- Salary --------------------
    base_salary = {
        'Sales': 45000,
        'R&D': 55000,
        'HR': 40000,
        'Finance': 50000,
        'Operations': 42000,
        'IT': 60000
    }
    
    salary = [
        base_salary[d] +
        (exp_years[i] * 800) +
        (edu_num[i] * 2000) +
        np.random.normal(0, 5000)
        for i, d in enumerate(dept)
    ]
    
    salary = np.round(salary, -2)
    
    # -------------------- Training --------------------
    training_hours = np.random.poisson(lam=40, size=n) + (exp_years < 3) * 15
    training_hours = np.clip(training_hours, 0, 150)
    
    # -------------------- Projects --------------------
    projects = np.random.poisson(lam=5, size=n) + (exp_years / 3).astype(int)
    projects = np.clip(projects, 1, 20)
    
    # -------------------- Absenteeism --------------------
    absenteeism = np.random.poisson(lam=3, size=n)
    absenteeism = np.where(dept == 'Operations', absenteeism + 2, absenteeism)
    
    # -------------------- Performance Score --------------------
    perf_score_raw = (
        0.3 * (exp_years / 10) +
        0.2 * (edu_num / 2) +              # ✅ now works
        0.2 * (training_hours / 50) +
        0.15 * (projects / 10) -
        0.15 * (absenteeism / 5) +
        np.random.normal(0, 0.1, n)
    )
    
    # Scale to 1–5
    perf_score_raw = 2.5 + perf_score_raw * 2
    perf_score = np.clip(np.round(perf_score_raw), 1, 5).astype(int)
    
    # -------------------- DataFrame --------------------
    df = pd.DataFrame({
        'EmployeeID': [f'EMP{str(i).zfill(4)}' for i in range(1, n + 1)],
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'Department': dept,
        'Education': education,
        'YearsExperience': exp_years,
        'Salary': salary,
        'TrainingHours': training_hours,
        'ProjectsCompleted': projects,
        'Absenteeism': absenteeism,
        'PerformanceScore': perf_score
    })
    
    # -------------------- Missing Values --------------------
    missing_idx = np.random.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[missing_idx, 'TrainingHours'] = np.nan
    
    return df


# -------------------- Run Script --------------------
if __name__ == "__main__":
    df = generate_employee_data(1500)
    df.to_csv('../data/raw/employee_data.csv', index=False)
    print("✅ Synthetic data generated and saved to data/raw/employee_data.csv")