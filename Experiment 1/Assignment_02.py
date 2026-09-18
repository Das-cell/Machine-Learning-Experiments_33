import pandas as pd

data = {
    "Student Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [72, 85, 68, 91, 77],
    "Attendance": [88, 92, 76, 95, 81]
}

df = pd.DataFrame(data)

print("--- Complete DataFrame ---")
print(df)

print("\n--- Students Scoring Above 80 ---")
print(df[df["Marks"] > 80])