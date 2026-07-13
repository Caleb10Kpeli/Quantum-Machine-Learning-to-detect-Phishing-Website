import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_data.csv.csv')

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nClass distribution:\n{df['status'].value_counts().to_string()}\n")

# ── 2. Features & label ───────────────────────────────────────────────────────
# Drop the raw URL string; keep all numeric features.
# Features aligned with project methodology are marked below:
#   [URL-len]   length_url, length_hostname
#   [special-chars] nb_dots, nb_hyphens, nb_at, nb_qm, nb_and, nb_percent ...
#   [https]     https_token, http_in_path
#   [domain]    domain_age, domain_registration_length, whois_registered_domain
FEATURES = [c for c in df.columns if c not in ('url', 'status')]

X = df[FEATURES].copy()

# Encode label: legitimate → 0, phishing → 1
le = LabelEncoder()
y = le.fit_transform(df['status'])
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Handle any missing values
missing = X.isnull().sum().sum()
if missing:
    print(f"Filling {missing} missing values with column medians")
    X.fillna(X.median(), inplace=True)

# ── 3. Train / test split (80 / 20, stratified) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nSplit -> train: {len(X_train)}, test: {len(X_test)}")

# ── 4. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train SVM ──────────────────────────────────────────────────────────────
print("\nTraining classical SVM (RBF kernel)...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
print("Training complete.")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = svm.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("  Classical SVM Baseline Results")
print("="*50)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("="*50)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"  {'':12s}  legitimate  phishing")
print(f"  {'legitimate':12s}  {cm[0,0]:10d}  {cm[0,1]:8d}")
print(f"  {'phishing':12s}  {cm[1,0]:10d}  {cm[1,1]:8d}")

# ── 7. Save model, scaler, encoder for QML comparison later ──────────────────
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(svm,    os.path.join(models_dir, 'classical_svm.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(le,     os.path.join(models_dir, 'label_encoder.pkl'))

print(f"\nSaved to models/  ->  classical_svm.pkl | scaler.pkl | label_encoder.pkl")
print("These will be reused by the QML pipeline for a fair comparison.")
