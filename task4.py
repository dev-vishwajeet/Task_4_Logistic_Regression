import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ===================================
# Load Dataset
# ===================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Loaded Successfully!\n")

# ===================================
# Dataset Information
# ===================================

print("Dataset Shape:", X.shape)
print("\nFirst 5 Rows:\n")
print(X.head())

# ===================================
# Train Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ===================================
# Feature Scaling
# ===================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# ===================================
# Logistic Regression Model
# ===================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ===================================
# Predictions
# ===================================
y_pred = model.predict(X_test)

# ===================================
# Actual vs Predicted
# ===================================

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:\n")
print(comparison.head(21))

# ===================================
# Evaluation
# ===================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===================================
# Confusion Matrix
# ===================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot()
plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ===================================
# ROC-AUC
# ===================================

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(10, 6))

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ===================================
# ROC-AUC
# ===================================

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(10, 6))

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ===================================
# ROC-AUC
# ===================================

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(10, 6))

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ===================================
# ROC-AUC
# ===================================

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", auc)
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


