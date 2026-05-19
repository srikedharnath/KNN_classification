import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("KNN Classification App")


iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


st.subheader("Enter Flower Measurements")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")


if st.button("Predict"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)[0]

    species = iris.target_names[prediction]

    st.success(f"Predicted Flower: {species}")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")