# ---- BLOOD DONOR RETURN PREDICTION ----
# Dataset: UCI Blood Transfusion Service Center (748 donors)
# Goal: predict whether a donor will donate again
# Method: Logistic Regression

# ---- 1. IMPORTS ----
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ---- 2. LOAD DATA ----
df = pd.read_csv('../data/transfusion.csv')

# Rename columns to clean short names
df.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Class']

print("=== DATASET SHAPE ===")
print(df.shape)

# ---- 3. EDA ----
print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== CLASS BALANCE ===")
print(df['Class'].value_counts())
print(f"Positive rate: {df['Class'].mean()*100:.1f}%")

print("\n=== BASIC STATS ===")
print(df.describe())

# ---- 4. FEATURE ENGINEERING ----
# Drop Monetary: equals Frequency x 250 — perfectly multicollinear
# Keeping it gives the model redundant info — we remove it deliberately
df_clean = df.drop(columns=['Monetary'])

X = df_clean.drop(columns=['Class'])
y = df_clean['Class']

print(f"\nFeatures used: {list(X.columns)}")
print("Dropped: Monetary (= Frequency x 250, multicollinear)")

# ---- 5. TRAIN / TEST SPLIT ----
# stratify=y keeps same positive rate in both train and test
# random_state=42 makes results reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean()*100:.1f}%")
print(f"Test positive rate:  {y_test.mean()*100:.1f}%")

# ---- 6. SCALING ----
# IMPORTANT: fit scaler on TRAIN only, then apply to test
# Fitting on all data = data leakage = dishonest performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- 7. MODEL ----
# class_weight='balanced' handles imbalance
# tells model: missing a donor is worse than a false alarm
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---- 8. EVALUATION ----
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['No Donation', 'Donation']))

print("=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("""
[[TN  FP]
 [FN  TP]]
TN = correctly predicted non-donors
FP = false alarms
FN = missed donors  <- most costly
TP = correctly caught donors
""")

auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {auc:.3f}")

# ---- 9. COEFFICIENTS ----
coef_df = pd.DataFrame({
    'Feature':        X.columns,
    'Coefficient':    model.coef_[0]
})
print("\n=== COEFFICIENTS ===")
print(coef_df)
print("\nNegative Recency = recent donors more likely to return (makes sense)")

# ---- 10. ROC CURVE ----
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f'Logistic Regression (AUC = {auc:.3f})')
plt.plot([0,1],[0,1], color='gray', lw=1, linestyle='--', label='Random (0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve — Blood Donor Return Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('../blood_donor_roc.png', dpi=150)
print("\nROC curve saved as blood_donor_roc.png")

# ---- 11. SUMMARY ----
print("""
=== YOUR INTERVIEW SUMMARY ===
748 real blood donors, ~24% donated again (imbalanced dataset)
Dropped Monetary: it equals Frequency x 250 (multicollinear, no new info)
Stratified 75/25 split, scaler fit on train only (no data leakage)
Logistic Regression with balanced class weights (handles imbalance)
AUC ~0.75: meaningfully better than random (0.5)
Key finding: Recency strongest predictor — recent donors return most
Next steps: Random Forest comparison, SMOTE resampling, k-fold CV
""")