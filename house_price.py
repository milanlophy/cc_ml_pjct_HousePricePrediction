import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Sample house dataset
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

# Input and output
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Get house details from the user
print("HOUSE PRICE PREDICTION")
area = float(input("Enter area in square feet: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price
new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms]
})

price = model.predict(new_house)[0]

print("\nEstimated House Price: ₹", round(price, 2))

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error: ₹", round(mae, 2))