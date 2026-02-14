import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(
    page_title="Restaurant Rating Predictor",   
    page_icon="🍽️",             
    layout="wide"               
)

st.set_page_config(layout= "wide")

scaler = joblib.load("scaler.pkl")

st.title("Restaurant Rating Prediction App")

st.caption("This app helps you to predict a restaurants review class")

st.divider()

averagecost = st.number_input("Please enter the estimated average cost for two people", min_value=50, max_value=999999, step=200, value=1000)

tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])

pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)?", [1, 2, 3, 4])

predictbutton = st.button("Predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]

my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()
    
    result = model.predict(X)
    
    st.write("The predicted review value is:", result[0])

    if result[0] <2.5:
        st.write("This restaurant is likely to receive a bad review.")
    elif result[0] < 3.5:
        st.write("This restaurant is likely to receive an average review.")
    elif result[0] < 4:
        st.write("This restaurant is likely to receive a good review.")
    elif result[0] < 4.5:
        st.write("This restaurant is likely to receive a very good review.")
    else:
        st.write("This restaurant is likely to receive an excellent review.")
