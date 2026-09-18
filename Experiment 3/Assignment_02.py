# Practice Problem 2 - MinMaxScaler

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# Apply MinMaxScaler
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\n--- Data after MinMax Scaling ---")
print(scaled_df)
