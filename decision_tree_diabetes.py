import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)

try:
    from google.colab import files
    print("Please select your diabetes.csv file...")
    uploaded = files.upload()
    csv_filename = next(iter(uploaded))
except ImportError:
    csv_filename = "diabetes.csv"

df = pd.read_csv(csv_filename)
print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)
    df[column] = df[column].fillna(df[column].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)
y_pred_full = tree_full.predict(X_test)

accuracy_full  = accuracy_score(y_test, y_pred_full)
precision_full = precision_score(y_test, y_pred_full)
recall_full    = recall_score(y_test, y_pred_full)
f1_full        = f1_score(y_test, y_pred_full)

print("\n================ FULL DECISION TREE PERFORMANCE ================")
print(f"Accuracy  : {accuracy_full:.4f}")
print(f"Precision : {precision_full:.4f}")
print(f"Recall    : {recall_full:.4f}")
print(f"F1-Score  : {f1_full:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_full))

tree_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_depth3.fit(X_train, y_train)

y_pred_depth3 = tree_depth3.predict(X_test)

accuracy_depth3  = accuracy_score(y_test, y_pred_depth3)
precision_depth3 = precision_score(y_test, y_pred_depth3)
recall_depth3    = recall_score(y_test, y_pred_depth3)
f1_depth3        = f1_score(y_test, y_pred_depth3)

print("\n================ DECISION TREE max_depth=3 PERFORMANCE ================")
print(f"Accuracy  : {accuracy_depth3:.4f}")
print(f"Precision : {precision_depth3:.4f}")
print(f"Recall    : {recall_depth3:.4f}")
print(f"F1-Score  : {f1_depth3:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_depth3))

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Full Decision Tree": [accuracy_full, precision_full, recall_full, f1_full],
    "Decision Tree max_depth=3": [accuracy_depth3, precision_depth3, recall_depth3, f1_depth3]
})

print("\n================ MODEL COMPARISON ================")
print(comparison.to_string(index=False))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_full.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n================ FEATURE IMPORTANCE (Full Tree) ================")
print(feature_importance.to_string(index=False))

print("\n================ DECISION TREE PREDICTION ================")
def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a numeric value.")

pregnancies       = get_float("Pregnancies: ")
glucose           = get_float("Glucose: ")
blood_pressure    = get_float("Blood Pressure: ")
skin_thickness    = get_float("Skin Thickness: ")
insulin           = get_float("Insulin: ")
bmi               = get_float("BMI: ")
diabetes_pedigree = get_float("Diabetes Pedigree Function: ")
age               = get_float("Age: ")

user_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, diabetes_pedigree, age
]], columns=X.columns)
for column in zero_columns:
    if user_data.loc[0, column] == 0:
        user_data.loc[0, column] = X[column].median()

prediction_full = tree_full.predict(user_data)[0]
probability_full = tree_full.predict_proba(user_data)[0][1]

prediction_depth3 = tree_depth3.predict(user_data)[0]
probability_depth3 = tree_depth3.predict_proba(user_data)[0][1]

print("\n================ PREDICTION RESULT ================")
print("\nFULL DECISION TREE:")
print("Prediction:", "DIABETES" if prediction_full == 1 else "NO DIABETES")
print(f"Probability: {probability_full * 100:.2f}%")
print("\nDECISION TREE max_depth=3:")
print("Prediction:", "DIABETES" if prediction_depth3 == 1 else "NO DIABETES")
print(f"Probability: {probability_depth3 * 100:.2f}%")