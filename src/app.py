import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate the chance this customer will leave.")


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")
    return joblib.load(path)


model = load_model()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)

threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

if st.button("Predict"):
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,  # approximation
    }])

    prob = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(min(float(prob), 1.0))

    if prob >= threshold:
        st.error("⚠️ High risk: consider a retention offer.")
    else:
        st.success("✅ Low risk of leaving.")