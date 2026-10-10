#  this provides the frontend


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Sample dataset (prices in Indian rupees)
data = {
    "Area": [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "Price": [2000000, 2500000, 3000000, 3500000,
              4000000, 4500000, 5000000, 5500000,
              6000000, 7000000]
}

df = pd.DataFrame(data)

# Train the machine learning model
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# Frontend
st.title("🏠 House Price Prediction")
st.write(
    "Predict the estimated price of a house "
    "using Machine Learning."
)

st.divider()

st.subheader("Enter House Details")

area = st.number_input(
    "Area (square feet)",
    min_value=300,
    max_value=10000,
    value=None,
    step=100
)

bedrooms = st.selectbox(
    "Number of Bedrooms",
    [1, 2, 3, 4, 5, 6]
)

bathrooms = st.selectbox(
    "Number of Bathrooms",
    [1, 2, 3, 4, 5, 6]
)

# Predict button
if st.button("Predict House Price", type="primary"):
    new_house = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms]
    })

    predicted_price = model.predict(new_house)[0]

    st.success("Prediction Completed!")

    st.metric(
        label="Estimated House Price",
        value=f"₹ {predicted_price:,.0f}"
    )

    st.caption(
        "This is an educational estimate based on "
        "sample data, not a real market valuation."
    )

st.divider()

st.subheader("About This Project")
st.write("Algorithm: Linear Regression")
st.write("Frontend: Streamlit")
st.write("Language: Python")
st.write("Libraries: Pandas and Scikit-learn")