"""Streamlit demo: Telecom Customer Churn Prediction.

Run locally:  streamlit run app.py
Or on Streamlit Cloud from this repo (Python 3.11+, see requirements.txt).

The app trains the LightGBM model in memory on launch from the bundled
IBM Telco dataset (7,043 rows) — no model binary is stored in the repo —
then predicts churn probability for the customer profile you enter.
"""
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@st.cache_resource
def load_model():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(n_estimators=300, num_leaves=31,
                               learning_rate=0.05, class_weight="balanced",
                               random_state=42, n_jobs=2, verbose=-1)),
    ])
    model.fit(X, y)
    return model, X.columns.tolist()


st.set_page_config(page_title="Churn Prediction — Telecom", layout="centered")
st.title("Telecom Customer Churn Prediction")
st.write(
    "LightGBM model trained on the IBM Telco Customer Churn dataset "
    "(7,043 customers). Enter a customer profile to estimate churn probability."
)

model, _ = load_model()

with st.form("customer"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("MonthlyCharges ($)", 18.0, 120.0, 70.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"])
    with col2:
        senior = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
        tech = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    submitted = st.form_submit_button("Predict churn probability")

if submitted:
    total = tenure * monthly
    row = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": tech,
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    proba = float(model.predict_proba(row)[0, 1])
    st.metric("Churn probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.warning("High churn risk — consider a retention offer (contract upgrade, "
                   "tech-support bundle, or payment-method review).")
    else:
        st.success("Low churn risk for this profile.")
    st.caption("Demo defaults: gender, phone and streaming fields are fixed to "
               "typical values; the full model uses all 19 features. See README.md "
               "for measured test-set metrics.")
