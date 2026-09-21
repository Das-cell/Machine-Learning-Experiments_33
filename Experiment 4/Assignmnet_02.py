# Practice Problem 2 - Multiple Linear Regression

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([
    [500, 1],
    [700, 2],
    [900, 2],
    [1100, 3],
    [1300, 3],
    [1500, 4],
    [1700, 4],
    [1900, 5]
])

y = np.array([20, 28, 35, 43, 50, 58, 66, 74])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)


new_house = np.array([[1200, 3]])
predicted_price = model.predict(new_house)

print("Predicted House Price:",
      round(predicted_price[0], 2),
      "lakh")

print("\nCoefficient of Area:", model.coef_[0])
print("Coefficient of Bedrooms:", model.coef_[1])
print("Intercept:", model.intercept_)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))
