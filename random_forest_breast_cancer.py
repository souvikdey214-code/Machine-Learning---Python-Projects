import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print("Dataset Loaded Successfully!")
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n================ RANDOM FOREST PERFORMANCE ================")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n================ TOP 10 FEATURE IMPORTANCES ================")
print(importance.head(10).to_string(index=False))

print("\n================ BREAST CANCER PREDICTION ================")
print(f"This dataset needs all {len(data.feature_names)} feature values.")
print("Tip: if you're testing quickly, you can copy a real row's values")
print("from X.iloc[0].values instead of typing 30 numbers.\n")

user_values = []
for feature in data.feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            break
        except ValueError:
            print("  Please enter a numeric value.")
    user_values.append(value)

user_input = pd.DataFrame([user_values], columns=data.feature_names)
prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0]
print("\n================ PREDICTION RESULT ================")
print("Prediction:", "MALIGNANT" if prediction == 0 else "BENIGN")
print(f"Malignant Probability: {probability[0] * 100:.2f}%")
print(f"Benign Probability   : {probability[1] * 100:.2f}%")