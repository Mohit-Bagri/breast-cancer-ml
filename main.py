import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. IMPORT ML LIBRARIES
# =============================
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap

# =============================
# 2. LOAD DATA
# =============================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="label")

# =============================
# 3. TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test shapes:")
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# =============================
# 4. BASE PIPELINE (SCALER + LOGREG)
# =============================
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# =============================
# 5. METRICS (BASELINE)
# =============================
print("\n=== BASELINE LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =============================
# 6. CROSS VALIDATION
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# =============================
# 7. GRID SEARCH (HYPERPARAMS)
# =============================
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n=== BEST LOGISTIC REGRESSION (GRID SEARCH) ===")
print("Best Params:", grid_search.best_params_)
print("Best CV:", grid_search.best_score_)
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1:", f1_score(y_test, y_pred_best))

# =============================
# 8. SHAP INTERPRETATION
# =============================
shap.initjs()
final_model = best_model.named_steps["model"]
final_scaler = best_model.named_steps["scaler"]

X_test_scaled = final_scaler.transform(X_test)

explainer = shap.LinearExplainer(final_model, X_test_scaled)
shap_values = explainer.shap_values(X_test_scaled)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# =============================
# 9. RANDOM FOREST COMPARISON
# =============================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== RANDOM FOREST METRICS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))

# =============================
# 10. THRESHOLD TUNING
# =============================
y_proba = best_model.predict_proba(X_test)[:, 1]

prec, rec, thresh = precision_recall_curve(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Plot PR Curve
plt.figure(figsize=(6, 5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Find best threshold for F1
best_thresh = 0
best_f1 = 0

for t in np.linspace(0, 1, 200):
    preds_t = (y_proba >= t).astype(int)
    score = f1_score(y_test, preds_t)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print("\nBest Threshold:", best_thresh)
print("Best F1 Score:", best_f1)

y_pred_custom = (y_proba >= best_thresh).astype(int)

print("\n=== CUSTOM THRESHOLD METRICS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))
print("F1:", f1_score(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))


import joblib
import json

# =============================
# 11. SAVE MODEL + METADATA
# =============================

joblib.dump(best_model, "breast_cancer_best_pipeline.joblib")
print("\nModel saved successfully as breast_cancer_best_pipeline.joblib")

metadata = {
    "feature_names": list(X.columns),
    "model_type": "LogisticRegression",
    "hyperparameters": grid_search.best_params_,
    "threshold": float(best_thresh),
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "version": "1.0",
}

with open("breast_cancer_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Metadata saved as breast_cancer_metadata.json")

# =============================
# 12. LOAD AND TEST SAVED MODEL
# =============================

loaded_model = joblib.load("breast_cancer_best_pipeline.joblib")
print("\nLoaded model:", loaded_model)

loaded_pred = loaded_model.predict(X_test)

print("\n=== Loaded Model Metrics ===")
print("Accuracy:", accuracy_score(y_test, loaded_pred))
print("Precision:", precision_score(y_test, loaded_pred))
print("Recall:", recall_score(y_test, loaded_pred))
print("F1 Score:", f1_score(y_test, loaded_pred))
