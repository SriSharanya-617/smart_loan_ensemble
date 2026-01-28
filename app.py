import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide"
)

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------
df = pd.read_csv("loan_data.csv")

# ------------------------------------------------
# PREPROCESSING
# ------------------------------------------------
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)   # Y=1, N=0

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------
# PIPELINE
# ------------------------------------------------
preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ------------------------------------------------
# BASE MODELS
# ------------------------------------------------
base_models = [
    ('lr', Pipeline([
        ('prep', preprocess),
        ('model', LogisticRegression(max_iter=1000))
    ])),
    ('dt', Pipeline([
        ('prep', preprocess),
        ('model', DecisionTreeClassifier(max_depth=5, random_state=42))
    ])),
    ('rf', Pipeline([
        ('prep', preprocess),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]))
]

# ------------------------------------------------
# STACKING MODEL
# ------------------------------------------------
meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

# ------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------
st.sidebar.header("👤 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", 0, 1000000, 5000)
co_income = st.sidebar.number_input("Co-Applicant Income", 0, 1000000, 0)
loan_amt = st.sidebar.number_input("Loan Amount", 0, 1000000, 150)
loan_term = st.sidebar.number_input("Loan Amount Term", 0, 500, 360)
credit_hist = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# Encode input
credit_val = 1 if credit_hist == "Yes" else 0
emp_val = 1 if employment == "Self-Employed" else 0
prop_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}

input_data = pd.DataFrame([[  
    app_income,
    co_income,
    loan_amt,
    loan_term,
    credit_val,
    emp_val,
    prop_map[property_area]
]], columns=X.columns)

# ------------------------------------------------
# MAIN UI
# ------------------------------------------------
st.markdown("## 🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be approved by combining multiple ML models."
)

st.markdown("### 🧠 Model Architecture")
st.write("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression
""")

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):
    # Base model predictions
    lr_pred = stack_model.named_estimators_['lr'].predict(input_data)[0]
    dt_pred = stack_model.named_estimators_['dt'].predict(input_data)[0]
    rf_pred = stack_model.named_estimators_['rf'].predict(input_data)[0]

    final_pred = stack_model.predict(input_data)[0]

    lr_res = "Approved" if lr_pred == 1 else "Rejected"
    dt_res = "Approved" if dt_pred == 1 else "Rejected"
    rf_res = "Approved" if rf_pred == 1 else "Rejected"
    final_res = "Approved" if final_pred == 1 else "Rejected"

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------
    st.markdown("### 📌 Prediction Result")
    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"""
- Logistic Regression → **{lr_res}**
- Decision Tree → **{dt_res}**
- Random Forest → **{rf_res}**
    """)

    st.markdown(f"### 🧠 Final Stacking Decision → **{final_res}**")

    st.markdown("### 💼 Business Explanation")
    st.write(
        f"Based on applicant income, credit history, loan amount, and combined "
        f"predictions from multiple models, the system predicts that the applicant "
        f"is **{'likely' if final_pred==1 else 'unlikely'}** to repay the loan. "
        f"Therefore, the loan is **{final_res.lower()}**."
    )
