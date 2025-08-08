import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model and scaler
model = joblib.load('C:/Users/admin/Downloads/xgb.pkl')
scaler = joblib.load('C:/Users/admin/Downloads/scaler_.pkl')

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")
st.title("🏦 Bank Marketing Campaign - Term Deposit Prediction")
st.markdown("Predict whether a customer will **subscribe to a term deposit** based on their profile and contact history.")

# Input form
def user_input():
    col1, col_spacer1, col2, col_spacer2, col3 = st.columns([1, 0.2, 1, 0.2, 1])

    with col1:
        age = st.slider("Age", 18, 80, 30)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", ["Basic 4y", "Basic 6y", "Basic 9y", "high school", "professional course", "university degree"])
        housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
        loan = st.selectbox("Has Personal Loan?", ["yes", "no"])

    with col2:
        contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month", ["mar", "apr", "may", "jun", "jul", "aug", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Last Contact Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        duration = st.slider("Last Contact Duration (seconds)", 0, 3000, 200)
        campaign = st.slider("Number of Contacts in Campaign", 1, 50, 2)

    with col3:
        previous = st.slider("Number of Previous Contacts", 0, 10, 0)
        emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, step=0.1)
        cons_price_idx = st.number_input("Consumer Price Index", value=93.2, step=0.1)
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0, step=0.5)
        euribor3m = st.number_input("Euribor 3-Month Rate", value=4.8, step=0.1)

    with st.sidebar:
        st.markdown("### ℹ️ About this App")
        st.write("""
        This tool helps bank marketing teams predict whether a client will subscribe 
        to a term deposit. It is trained on historical campaign and client data.
        
        Model: XGBoost   
        """)

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'previous': previous,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m
    }
    return pd.DataFrame([data])

# Preprocess input
def preprocess_input(df):
    job_map = {
        "admin.": 0,
        "blue-collar": 1,
        "entrepreneur": 2,
        "housemaid": 3,
        "management": 4,
        "retired": 5,
        "self-employed": 6,
        "services": 7,
        "student": 8,
        "technician": 9,
        "unemployed": 10,
    }
    df['job'] = df['job'].map(job_map)

    marital_map = {
        "divorced": 0,
        "married": 1,
        "single": 2,
    }
    df['marital'] = df['marital'].map(marital_map)

    education_map = {
        "Basic 4y": 0,
        "Basic 6y": 1,
        "Basic 9y": 2,
        "high school": 3,
        "professional course": 4,
        "university degree": 5,
    }
    df['education'] = df['education'].map(education_map)

    housing_map = {
        "no": 0,
        "yes": 2
    }
    df['housing'] = df['housing'].map(housing_map)
    
    loan_map = {
        "no": 0,
        "yes": 2
    }
    df['loan'] = df['loan'].map(loan_map)

    contact_map = {
        "cellular": 0,
        "telephone": 1
    }
    df['contact'] = df['contact'].map(contact_map)

    month_map = {
        "apr": 0,
        "aug": 1,
        "dec": 2,
        "jul": 3,
        "jun": 4,
        "mar": 5,
        "may": 6,
        "nov": 7,
        "oct": 8,
        "sep": 9
    }
    df['month'] = df['month'].map(month_map)

    day_of_week_map = {
        "fri": 0,
        "mon": 1,
        "thu": 2,
        "tue": 3,
        "wed": 4
    }
    df['day_of_week'] = df['day_of_week'].map(day_of_week_map)

    return df

# Run Streamlit app
input_df = user_input()
st.write("### Input Summary", input_df)

if st.button("Predict"):
    processed_input = preprocess_input(input_df.copy())
    
    # Ensure column order is same as training
    columns_order = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month',
                     'day_of_week', 'duration', 'campaign', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
    
    processed_input = processed_input[columns_order]
    
    # Scale
    input_scaled = scaler.transform(processed_input)

    # Predict
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    label = "✔️ Subscribed" if prediction == 1 else "❌ Not Subscribed"

    # Display results
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Probability of Subscription**: `{probability:.2f}`")
