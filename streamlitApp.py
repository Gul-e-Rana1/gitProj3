import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
# Load the updated model to be made live
model = joblib.load("liveModelV1.pkl")

# Load and train test splitting
data = pd.read_csv("mobile_price_range_data.csv")
X = data.iloc[: , :-1]
y = data.iloc[: , -1]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_train, y_test)

# Page title
st.title("Model Accuracy and Real-Time Prediction")

# Display accuracy
st.write(f"Model {accuracy}")

# Real-Time Prediction on users input
st.header("Real-Time Prediction")
input_data = []
for col in X_test.columns :
    input_value = st.number_input(f'Input for feature {col}', value=0) 
    input_data.append(input_value)
# Convert input data to Dataframe
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make Prediction 
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')