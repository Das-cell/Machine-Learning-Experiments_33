# Practice Problem 3 - Compare Scaling Methods

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

# StandardScaler
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(df)

standard_df = pd.DataFrame(
    standard_data,
    columns=df.columns
)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_data,
    columns=df.columns
)

print("--- Original Data ---")
print(df)

print("\n--- StandardScaler Output ---")
print(standard_df)

print("\n--- MinMaxScaler Output ---")
print(minmax_df)

print("\n--- StandardScaler Range ---")
print(
    "Minimum:",
    standard_df.min().min()
)

print(
    "Maximum:",
    standard_df.max().max()
)

print("\n--- MinMaxScaler Range ---")
print(
    "Minimum:",
    minmax_df.min().min()
)

print(
    "Maximum:",
    minmax_df.max().max()
)