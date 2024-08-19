import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('heart.csv')
    return data


data = load_data()
st.write("Heart Disease Dataset", data.head())

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# User input
st.title("Heart Disease Prediction")

age = st.number_input('Age', min_value=20, max_value=100, value=50)
sex = st.selectbox('Sex', options=[0, 1])
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Prepare input for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
st.write("Prediction: ", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
