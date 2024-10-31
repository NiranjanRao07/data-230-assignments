import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris_dataset = load_iris()
# Use only the 0'th and 1'st columns (sepal length and width)
X = iris_dataset['data'][:, [0, 1]]
y = iris_dataset['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict and evaluate the model on the test set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit interface
st.title("Iris Flower Predictor")
st.write("Enter the sepal length and width to predict the type of Iris flower.")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Slider settings based on data range
sl_min = float(X_train[:, 0].min().round(2))
sl_max = float(X_train[:, 0].max().round(2))
sw_min = float(X_train[:, 1].min().round(2))
sw_max = float(X_train[:, 1].max().round(2))

# Input sliders for Sepal Length and Sepal Width
sepal_length = st.slider("Sepal Length (cm)", min_value=sl_min, max_value=sl_max, value=float(X_train[:, 0].mean().round(2)))
sepal_width = st.slider("Sepal Width (cm)", min_value=sw_min, max_value=sw_max, value=float(X_train[:, 1].mean().round(2)))

# Prediction function
def predict_flower(sepal_length, sepal_width):
    input_data = np.array([sepal_length, sepal_width]).reshape(1, -1)
    prediction = rf.predict(input_data)
    if prediction[0] == 0:
        return "Setosa"
    elif prediction[0] == 1:
        return "Versicolor"
    elif prediction[0] == 2:
        return "Virginica"

# Display the prediction on button click
if st.button("Predict"):
    prediction = predict_flower(sepal_length, sepal_width)
    st.write(f"Predicted Iris Species: **{prediction}**")
