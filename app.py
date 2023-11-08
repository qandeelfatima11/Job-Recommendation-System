# Import necessary libraries
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Set the title of the web app
st.title('Iris Flower Species Prediction App')

# Create input fields for user input features
sepal_length = st.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Get user input and make prediction
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)

# Map the prediction to the species name
predicted_species = iris.target_names[prediction][0]

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted species is {predicted_species}')
