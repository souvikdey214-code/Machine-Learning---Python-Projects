import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

try:
    from google.colab import files
    print("Please select your titanic.csv file...")
    uploaded = files.upload()
    csv_filename = next(iter(uploaded))
except ImportError:
    csv_filename = "titanic.csv"
df = pd.read_csv(csv_filename)

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df = df[features + ["Survived"]].copy()

print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\n================ XGBOOST PERFORMANCE ================")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_probability):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n================ FEATURE IMPORTANCE ================")
print(importance.to_string(index=False))

print("\n================ TITANIC SURVIVAL PREDICTION ================")
def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a numeric value.")

def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  Please enter a whole number.")

pclass = get_int("Passenger Class (1/2/3): ")

sex_input = input("Sex (male/female): ").strip().lower()
sex = 1 if sex_input == "female" else 0

age   = get_float("Age: ")
sibsp = get_int("Number of Siblings/Spouses: ")
parch = get_int("Number of Parents/Children: ")
fare  = get_float("Fare: ")

user_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], columns=X.columns)
user_scaled = scaler.transform(user_data)

prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

print("\n================ PREDICTION RESULT ================")
print("Prediction:", "SURVIVED" if prediction == 1 else "DID NOT SURVIVE")
print(f"Survival Probability: {probability * 100:.2f}%")