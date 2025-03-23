import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('OHE_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

import pandas as pd

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    # 'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_converted = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_converted, columns=label_encoder_geo.get_feature_names_out())

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

scaled = scaler.transform(input_data)

prediction = model.predict(scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write("Customer is likely to churn")
    st.write('Probability is ', prediction_prob)
else:
    st.write("Person is not likely to churn")
    st.write('Probability is ', prediction_prob)

