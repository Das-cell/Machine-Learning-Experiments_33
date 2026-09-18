# Practice Problem 1 - Complete Data Preprocessing

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Create dataset with missing values
data = {
    "Age": [22, 25, None, 28, 30, 24],
    "Salary": [25000, 32000, 28000, None, 45000, 30000],
    "Department": [
        "IT", "HR", "IT", "Sales", None, "HR"
    ],
    "Years_Experience": [1, 3, 2, 5, None, 2]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# 2. Separate features and target
# Here, Salary is selected as the target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# 3. Define columns
numeric_features = [
    "Age",
    "Years_Experience"
]

categorical_features = [
    "Department"
]

# 4. Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# 6. Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 7. Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)

# 9. Display results
print("\n--- Processed Data ---")
print(X_processed)

print("\nProcessed Shape:", X_processed.shape)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])