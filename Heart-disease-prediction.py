import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# 1. Load data
data = fetch_openml(name='heart-statlog', version=1, as_frame=True)
df = data.frame

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df['class'].value_counts())

# 2. Prepare data
X = df.drop('class', axis=1)
y = df['class'].map({'absent': 0, 'present': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain: {X_train.shape}")
print(f"Test:  {X_test.shape}")

# 3. Pipeline - Random Forest
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

pipe_rf.fit(X_train, y_train)

y_pred_rf = pipe_rf.predict(X_test)
y_proba_rf = pipe_rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("Random Forest")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"AUC:      {roc_auc_score(y_test, y_proba_rf):.4f}")
print("\nReport:")
print(classification_report(y_test, y_pred_rf))

# 4. Pipeline - Gradient Boosting
pipe_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

pipe_gb.fit(X_train, y_train)

y_pred_gb = pipe_gb.predict(X_test)
y_proba_gb = pipe_gb.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("Gradient Boosting")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"AUC:      {roc_auc_score(y_test, y_proba_gb):.4f}")
print("\nReport:")
print(classification_report(y_test, y_pred_gb))

# 5. Cross-Validation
print("\n" + "=" * 50)
print("Cross-Validation (5-fold)")
print("=" * 50)

cv_rf = cross_val_score(pipe_rf, X, y, cv=5, scoring='roc_auc')
cv_gb = cross_val_score(pipe_gb, X, y, cv=5, scoring='roc_auc')

print(f"Random Forest:     {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")
print(f"Gradient Boosting: {cv_gb.mean():.4f} +/- {cv_gb.std():.4f}")

# 6. GridSearch
print("\n" + "=" * 50)
print("GridSearch - Random Forest")
print("=" * 50)

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 15],
    'model__min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train, y_train)

print(f"\nBest params: {grid_rf.best_params_}")
print(f"Best AUC: {grid_rf.best_score_:.4f}")

y_pred_best = grid_rf.predict(X_test)
y_proba_best = grid_rf.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_proba_best):.4f}")

# 7. Feature Importance
best_model = grid_rf.best_estimator_.named_steps['model']
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(
    [X.columns[i] for i in indices][::-1],
    importances[indices][::-1]
)
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# 8. ROC Curve
plt.figure(figsize=(8, 6))

for name, y_proba in [
    ('Random Forest', y_proba_rf),
    ('Gradient Boosting', y_proba_gb),
    ('RF (Tuned)', y_proba_best)
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Healthy', 'Sick'],
    yticklabels=['Healthy', 'Sick']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 10. Predict new sample
new_patient = X_test.iloc[[0]]
print("\nNew sample:")
print(new_patient)

pred = grid_rf.predict(new_patient)[0]
proba = grid_rf.predict_proba(new_patient)[0]

print(f"\nPrediction: {'Sick' if pred == 1 else 'Healthy'}")
print(f"Prob Healthy: {proba[0]:.2%}")
print(f"Prob Sick:    {proba[1]:.2%}")