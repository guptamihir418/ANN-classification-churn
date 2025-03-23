import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('OHE_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App Title
st.title("🔮 Customer Churn Prediction")
st.markdown("This app predicts whether a customer is likely to churn based on their bank profile.")

st.markdown("---")
st.sidebar.header("📋 Customer Information")

# Sidebar Inputs
geography = st.sidebar.selectbox('🌍 Geography', label_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance')
credit_score = st.sidebar.number_input('📊 Credit Score')
estimated_salary = st.sidebar.number_input('📈 Estimated Salary')
tenure = st.sidebar.slider('📆 Tenure', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('✅ Is Active Member', [0, 1])

# Build DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-Hot Encode Geography
geo_converted = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_converted, columns=label_encoder_geo.get_feature_names_out())
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale and Predict
scaled = scaler.transform(input_data)
prediction = model.predict(scaled)
prediction_prob = prediction[0][0]

st.markdown("---")
st.subheader("📢 Prediction Result")

if prediction_prob > 0.5:
    st.error("⚠️ Customer is **likely to churn**")
else:
    st.success("✅ Customer is **not likely to churn**")

st.metric(label="🔎 Churn Probability", value=f"{prediction_prob:.2%}")
