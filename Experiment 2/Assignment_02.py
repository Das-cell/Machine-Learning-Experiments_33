
# Experiment 2 - Problem 2
# Boxplots for all numerical attributes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine Dataset
wine = load_wine()

# 2. Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 3. Create boxplots for all numerical features
plt.figure(figsize=(16, 8))

sns.boxplot(data=df)

plt.title("Boxplots of All Wine Dataset Features")
plt.xlabel("Features")
plt.ylabel("Values")

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()