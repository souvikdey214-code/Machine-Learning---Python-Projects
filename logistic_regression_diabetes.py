import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)

try:
    from google.colab import files
    print("Please select your diabetes.csv file...")
    uploaded = files.upload()
    csv_filename = next(iter(uploaded))
except ImportError:
    # Fallback for non-Colab environments
    csv_filename = "diabetes.csv"

df = pd.read_csv(csv_filename)

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nZero Values per column:")
print((df == 0).sum())

zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for column in zero_columns:
    if column in df.columns:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\n================ LOGISTIC REGRESSION PERFORMANCE ================")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_probability):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})
coefficients["Absolute_Importance"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values("Absolute_Importance", ascending=False)

print("\n================ MODEL COEFFICIENT INTERPRETATION ================")
print(coefficients[["Feature", "Coefficient"]].to_string(index=False))

print("\n================ DIABETES PREDICTION SYSTEM ================")
print("Enter Patient Information:")

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
    if column in user_data.columns and user_data.loc[0, column] == 0:
        user_data.loc[0, column] = X[column].median()
user_scaled = scaler.transform(user_data)
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]
print("\n================ PREDICTION RESULT ================")
print("Prediction:", "DIABETES" if prediction == 1 else "NO DIABETES")
print(f"Diabetes Probability: {probability * 100:.2f}%")