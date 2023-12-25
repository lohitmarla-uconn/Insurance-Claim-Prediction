"""
Title: Analyzing Insurance Claims for Future Financial Safeguard
Authors: Lohit Marla
Date: 2023-11-15
"""

# Import necessary Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset (replace 'insurance_data.csv' with the actual file path)
insurance_data = pd.read_csv("data/insurance_data.csv")

# Data Exploration
print("Dataset Information:")
print(insurance_data.info())
print("\nDataset Description:")
print(insurance_data.describe())

# Shapiro-Wilk Test for Normality
print("\nShapiro-Wilk Test for Normality:")
for column in insurance_data.select_dtypes(include=[np.number]).columns:
    stat, p = shapiro(insurance_data[column])
    print(f"{column}: Test Statistic = {stat:.3f}, p-value = {p:.3f}")

# Removing duplicates
insurance_data = insurance_data.drop_duplicates()

# Encoding categorical variables
insurance_data = pd.get_dummies(insurance_data, drop_first=True)

# Splitting the data into training and testing sets
target_column = 'target_column'  # Replace with your actual target column
X = insurance_data.drop(target_column, axis=1)
y = insurance_data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Logistic Regression Model
glm_model = LogisticRegression()
glm_model.fit(X_train, y_train)
glm_predictions = glm_model.predict_proba(X_test)[:, 1]
glm_pred_class = (glm_predictions > 0.5).astype(int)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict_proba(X_test)[:, 1]
rf_pred_class = (rf_predictions > 0.5).astype(int)

# XGBoost Model
xgb_model = XGBClassifier(random_state=123, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred_class = (xgb_predictions > 0.5).astype(int)

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=123)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict_proba(X_test)[:, 1]
dt_pred_class = (dt_predictions > 0.5).astype(int)

# Metrics for all models
models = {
    "Logistic Regression": (glm_predictions, glm_pred_class),
    "Random Forest": (rf_predictions, rf_pred_class),
    "XGBoost": (xgb_predictions, xgb_pred_class),
    "Decision Tree": (dt_predictions, dt_pred_class),
}

print("\nModel Evaluation Metrics:")
for model_name, (proba, pred) in models.items():
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, pred))
    print(f"{model_name} Accuracy: {accuracy_score(y_test, pred):.2f}")
    print(f"{model_name} AUC: {roc_auc_score(y_test, proba):.2f}")

# Confusion Matrix Plot for Logistic Regression
conf_matrix = confusion_matrix(y_test, glm_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curves for All Models
plt.figure(figsize=(10, 8))
for model_name, (proba, _) in models.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Feature Importance for Random Forest
rf_importances = rf_model.feature_importances_
rf_features = pd.Series(rf_importances, index=X.columns).sort_values(ascending=False)
rf_features.plot(kind='bar', figsize=(12, 6), title="Feature Importance: Random Forest")
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=True, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
