# Task 3: Heart Disease Prediction

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)
# Step 1: Load Dataset
# Replace with your dataset file name
df = pd.read_csv("heart.csv")

print("First 5 Rows:")
print(df.head())

# Step 2: Dataset Information

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Step 3: Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values if present
df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 4: Exploratory Data Analysis
# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

# Chest Pain Type vs Target
plt.figure(figsize=(8,5))
sns.countplot(
    x='cp',
    hue='target',
    data=df
)
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

# Step 5: Split Features and Target

X = df.drop('target', axis=1)
y = df['target']

# Step 6: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Step 7: Feature Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train Logistic Regression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Predictions

y_pred = model.predict(X_test)

# Step 10: Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# Step 11: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\nConfusion Matrix:")
print(cm)

# Step 12: Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 13: ROC Curve and AUC

y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.3f}"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
print("\nROC-AUC Score:")
print(round(auc_score,3))

# Step 14: Feature Importance

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
print("\nMost Important Features:")
print(feature_importance)
plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)
plt.title("Feature Importance")
plt.show()