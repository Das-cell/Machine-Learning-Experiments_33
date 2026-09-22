# Experiment 5: Logistic Regression
# Assignment 1: Student Pass/Fail Prediction

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


X = np.array([
    [1, 55],
    [2, 60],
    [2, 65],
    [3, 60],
    [3, 70],
    [4, 65],
    [4, 75],
    [5, 70],
    [5, 80],
    [6, 75],
    [6, 85],
    [7, 80],
    [7, 90],
    [8, 85],
    [8, 95],
    [9, 90],
    [1, 50],
    [2, 55],
    [4, 60],
    [6, 70]
])

# Target values
# 0 = Fail
# 1 = Pass

y = np.array([
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("----- Logistic Regression Results -----")

print("Actual Values    :", y_test)
print("Predicted Values :", y_pred)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n--- Confusion Matrix ---")
print(cm)


new_students = np.array([
    [7, 85],
    [2, 60],
    [5, 75]
])

new_students_scaled = scaler.transform(new_students)

new_predictions = model.predict(new_students_scaled)

print("\n--- New Student Predictions ---")

for i in range(len(new_students)):
    if new_predictions[i] == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(
        f"Study Hours: {new_students[i][0]}, "
        f"Attendance: {new_students[i][1]}%, "
        f"Prediction: {result}"
    )