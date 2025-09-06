#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# Make sure output folders exist
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# In[2]:


# Prefer preprocessed dataset if available
data_path = "water_potability_preprocessed.csv" if os.path.exists("water_potability_preprocessed.csv") else "water_potability.csv"
df = pd.read_csv(data_path)

# Safety: check target column
if "Potability" not in df.columns:
    raise ValueError("Dataset must contain a 'Potability' column.")

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.mean(numeric_only=True))

X = df.drop("Potability", axis=1)
y = df["Potability"].astype(int)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

df.head()


# In[3]:


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
}


# In[4]:


def get_proba(estimator, X_):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X_)
        s_min, s_max = np.min(s), np.max(s)
        return (s - s_min) / (s_max - s_min + 1e-9)
    return estimator.predict(X_)


# In[5]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_name, best_score, best_estimator = None, -1, None
reports_md = ["# Week 2 – Model Evaluation\n"]

for name, est in models.items():
    cv_acc = cross_val_score(est, X, y, cv=cv, scoring="accuracy").mean()

    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    y_proba = get_proba(est, X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = float("nan")

    results.append({
        "Model": name,
        "CV_Accuracy": round(cv_acc, 4),
        "Test_Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan
    })

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()

    if (auc if not np.isnan(auc) else acc) > best_score:
        best_score = auc if not np.isnan(auc) else acc
        best_name, best_estimator = name, est

print("Done training. Best model:", best_name)


# In[6]:


df_res = pd.DataFrame(results).sort_values(by=["ROC_AUC", "Test_Accuracy"], ascending=False)
df_res


# In[7]:


# Save best model
dump(best_estimator, "models/best_model.pkl")

# Save results table
df_res.to_csv("reports/results.csv", index=False)

# Bar plot of test accuracies
plt.figure(figsize=(7,4))
plt.bar(df_res["Model"], df_res["Test_Accuracy"])
plt.xticks(rotation=15, ha="right")
plt.ylim(0,1)
plt.ylabel("Test Accuracy")
plt.title("Model Comparison (Test Accuracy)")
plt.tight_layout()
plt.savefig("plots/model_comparison_accuracy.png")
plt.show()


# In[8]:


if best_name.startswith("Random Forest"):
    rf = best_estimator
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(7,4.5))
    plt.barh(importances.index, importances.values)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance_random_forest.png")
    plt.show()


# In[9]:


from joblib import load

# Load your saved best model
best_model = load("models/best_model.pkl")

# Test it on some data
sample = X_test.iloc[0:5]   # first 5 rows from your test set
print("Predictions:", best_model.predict(sample))


# In[ ]:




