
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# -----------------------------------
# 1. LOAD DATASET
# -----------------------------------

print("Loading dataset...")

df = pd.read_csv("dataset/creditcard.csv")

# Remove duplicate records
df = df.drop_duplicates().copy()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# -----------------------------------
# 2. DATA PREPROCESSING
# -----------------------------------

print("\nStarting preprocessing...")

# Separate input features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle missing values if any
X = X.fillna(X.median())

print("Features separated successfully.")

# -----------------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# -----------------------------------
# 4. FEATURE SCALING
# -----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# -----------------------------------
# 5. HANDLE IMBALANCED DATA USING SMOTE
# -----------------------------------

print("\nApplying SMOTE...")

smote = SMOTE(
    sampling_strategy=0.1,
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("SMOTE completed!")
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_balanced).value_counts().to_dict())

# -----------------------------------
# 6. TRAIN MACHINE LEARNING MODELS
# -----------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

print("\nStarting model training...")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_balanced, y_train_balanced)

    predictions = model.predict(X_test_scaled)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[name] = model

    print("\nModel:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

# -----------------------------------
# 7. COMPARE MODELS
# -----------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# -----------------------------------
# 8. SELECT BEST MODEL
# -----------------------------------

best_index = results_df["F1 Score"].idxmax()
best_model_name = results_df.loc[best_index, "Model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# Save model and scaler
joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# -----------------------------------
# 9. CONFUSION MATRIX
# -----------------------------------

best_predictions = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")
plt.close()

# -----------------------------------
# 10. MODEL COMPARISON GRAPH
# -----------------------------------

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("graphs/model_comparison.png")
plt.close()

# -----------------------------------
# 11. FINAL OUTPUT
# -----------------------------------

print("\n====================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("====================================")

print("Best Model:", best_model_name)
print("Saved Model: models/fraud_model.pkl")
print("Saved Scaler: models/scaler.pkl")
print("Saved Graphs:")
print("- graphs/confusion_matrix.png")
print("- graphs/model_comparison.png")
