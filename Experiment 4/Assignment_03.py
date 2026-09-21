# Practice Problem 3 - Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

linear_model = LinearRegression()
linear_model.fit(X, y)

linear_pred = linear_model.predict(X)
linear_r2 = r2_score(y, linear_pred)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_pred = poly_model.predict(X_poly)
poly_r2 = r2_score(y, poly_pred)

print("Linear Regression R2 Score:",
      round(linear_r2, 4))

print("Polynomial Regression R2 Score:",
      round(poly_r2, 4))

plt.scatter(X, y, color="blue", label="Actual Data")

plt.plot(
    X,
    linear_pred,
    color="red",
    label="Linear Regression"
)

plt.plot(
    X,
    poly_pred,
    color="green",
    label="Polynomial Regression"
)

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()
