# Practice Problem 1 - House Price Prediction

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([500, 700, 900, 1100, 1300, 1500]).reshape(-1, 1)

y = np.array([20, 28, 35, 43, 50, 58])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

new_area = np.array([[1000]])
predicted_price = model.predict(new_area)

print("Predicted price for 1000 sq. ft.:",
      predicted_price[0])

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

plt.scatter(X, y, color="blue", label="Actual Prices")
plt.plot(X, y_pred, color="red", label="Regression Line")

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("House Price (Lakh)")
plt.title("House Area vs House Price")
plt.legend()
plt.grid(True)
plt.show()
