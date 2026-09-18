import pandas as pd

data = {
    "Student Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Marks": [72, 85, 68, 91, 55]
}

df = pd.DataFrame(data)

def calculate_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Marks"].apply(calculate_grade)

print(df)