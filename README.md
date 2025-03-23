# 🔮 Customer Churn Prediction using Artificial Neural Networks

This project is a full-fledged **machine learning web app** built with **TensorFlow**, **scikit-learn**, and **Streamlit** to predict whether a customer is likely to leave (churn) a bank based on their profile data.

---

## 📂 Project Structure

📁 ANN-classification-project/ │ ├── model.h5 # Trained ANN model ├── scaler.pkl # StandardScaler used during training ├── label_encoder_gender.pkl # LabelEncoder for gender ├── OHE_geo.pkl # OneHotEncoder for Geography ├── app.py # Streamlit web app ├── requirements.txt # Python dependencies └── README.md 


---

## 🚀 How It Works

This ANN model predicts **churn probability** using the following features:

- Credit Score
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card ownership
- Active Membership status
- Estimated Salary

---

## 🧠 Model Overview

- Built using `TensorFlow` and `Keras`
- 3-layer ANN:
  - Input → Dense(64, ReLU)
  - Hidden → Dense(32, ReLU)
  - Output → Dense(1, Sigmoid)
- Trained on a cleaned and preprocessed customer dataset
- Evaluation metrics: accuracy, loss, and churn probability

---

## 🎯 Features of the Web App

- 🧾 Simple & intuitive UI using Streamlit
- 🎛 Sidebar inputs for all customer features
- 🔁 Real-time predictions with probability
- ✅ Visual output using `st.success` / `st.error` messages
- 📊 Metric display for prediction confidence

---

## 🛠 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/ANN-classification-project.git
   cd ANN-classification-project

