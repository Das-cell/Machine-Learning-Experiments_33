
# Experiment 2 - Problem 3
# Correlation Heatmap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine Dataset
wine = load_wine()

# 2. Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 3. Calculate correlation matrix
corr_matrix = df.corr()

# 4. Print correlation matrix
print("--- Correlation Matrix ---")
print(corr_matrix)

# 5. Find strongest positive correlation
# Remove self-correlation from the diagonal
corr_without_diag = corr_matrix.copy()

for i in range(len(corr_without_diag)):
    corr_without_diag.iloc[i, i] = float("-inf")

# Find the feature pair
max_corr = corr_without_diag.stack().idxmax()
max_value = corr_without_diag.stack().max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", max_corr[0])
print("Feature 2:", max_corr[1])
print("Correlation:", max_value)

# 6. Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.show()