import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sujid\Downloads\Fitness Data.csv")  # Ensure your dataset file is named correctly
    return df

df = load_data()

# Data Preprocessing
df.drop(columns=['Date', 'Time'], inplace=True)  # Drop non-numeric columns

# Define features and target
X = df.drop(columns=['Energy Burned'])  # Features
y = df['Energy Burned']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "Linear Regression": LinearRegression()
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Streamlit UI
st.title("🔥 Personal Fitness Tracker 🔥")

st.sidebar.header("📊 Enter Your Data")
step_count = st.sidebar.number_input("Step Count", min_value=0, max_value=50000, value=5000)
distance = st.sidebar.number_input("Distance (m)", min_value=0.0, max_value=50000.0, value=2000.0)
flights_climbed = st.sidebar.number_input("Flights Climbed", min_value=0, max_value=100, value=5)
walking_speed = st.sidebar.slider("Walking Speed (m/s)", 0.1, 2.0, 1.2)
double_support = st.sidebar.slider("Walking Double Support (%)", 0.0, 100.0, 20.0)

# Model selection
model_choice = st.sidebar.selectbox("🔍 Select Model", list(models.keys()))

# Prediction
if st.button("🚀 Predict Calories Burned"):
    input_data = np.array([[step_count, distance, flights_climbed, double_support, walking_speed]])
    pred = models[model_choice].predict(input_data)
    st.success(f"🔥 Estimated Energy Burned: {pred[0]:.2f} kcal")

# Model Performance
st.subheader("📈 Model Performance (on test data)")
for name, model in models.items():
    y_pred = model.predict(X_test)
    st.write(f"**{name}**")
    st.write(f"📊 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"📊 R² Score: {r2_score(y_test, y_pred):.2f}")

