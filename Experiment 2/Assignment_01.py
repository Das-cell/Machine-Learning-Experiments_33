

# Experiment 2 - Problem 1
# Complete EDA on Wine Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine Dataset
wine = load_wine()

# 2. Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target column
df["target"] = wine.target

# 3. Display first five rows
print("--- First Five Rows ---")
print(df.head())

# 4. Display last five rows
print("\n--- Last Five Rows ---")
print(df.tail())

# 5. Dataset shape
print("\n--- Dataset Shape ---")
print(df.shape)

# 6. Dataset information
print("\n--- Dataset Information ---")
df.info()

# 7. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 8. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 9. Duplicate values
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 10. Target class distribution
print("\n--- Target Class Distribution ---")
print(df["target"].value_counts())

# 11. Correlation matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# 12. Histogram of all numerical features
df.hist(figsize=(15, 12), bins=15)
plt.suptitle("Wine Dataset - Feature Distributions")
plt.tight_layout()
plt.show()

# 13. Scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target",
    palette="viridis"
)

plt.title("Alcohol vs Color Intensity")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.show()

# 14. Target class count plot
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="target")

plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.show()