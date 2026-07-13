"""
Trains and saves the Classical SVM (Fair) model —
same 6 features and 200 samples as the QSVM, for a head-to-head comparison.
Replicates the fair-SVM training block from qsvm.py exactly.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

N_QUBITS    = 6
N_TRAIN     = 100
N_TEST      = 50
C_PARAM     = 10
RANDOM_SEED = 42

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_data.csv.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

df       = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c not in ('url', 'status')]
le       = LabelEncoder()
y_all    = le.fit_transform(df['status'])
X_all    = df[features].fillna(df[features].median())

mi_scores    = mutual_info_classif(X_all, y_all, random_state=RANDOM_SEED)
top_idx      = np.argsort(mi_scores)[::-1][:N_QUBITS]
TOP_FEATURES = [features[i] for i in top_idx]
X_selected   = X_all[TOP_FEATURES].values

np.random.seed(RANDOM_SEED)
phish_idx = np.where(y_all == 1)[0]
legit_idx = np.where(y_all == 0)[0]
train_phish = np.random.choice(phish_idx, N_TRAIN, replace=False)
train_legit = np.random.choice(legit_idx, N_TRAIN, replace=False)
test_phish  = np.random.choice(np.setdiff1d(phish_idx, train_phish), N_TEST, replace=False)
test_legit  = np.random.choice(np.setdiff1d(legit_idx, train_legit), N_TEST, replace=False)
train_idx   = np.random.permutation(np.concatenate([train_phish, train_legit]))
test_idx    = np.random.permutation(np.concatenate([test_phish, test_legit]))

X_train_raw, y_train = X_selected[train_idx], y_all[train_idx]
X_test_raw,  y_test  = X_selected[test_idx],  y_all[test_idx]

scaler    = StandardScaler()
X_train   = scaler.fit_transform(X_train_raw)
X_test    = scaler.transform(X_test_raw)

model = SVC(kernel='rbf', C=C_PARAM, gamma='scale', random_state=RANDOM_SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1       : {f1_score(y_test, y_pred)*100:.2f}%")
print(f"Features : {TOP_FEATURES}")

joblib.dump(model,       os.path.join(MODELS_DIR, 'fair_svm.pkl'))
joblib.dump(scaler,      os.path.join(MODELS_DIR, 'fair_svm_scaler.pkl'))
joblib.dump(TOP_FEATURES, os.path.join(MODELS_DIR, 'fair_svm_features.pkl'))
print("Saved: fair_svm.pkl | fair_svm_scaler.pkl | fair_svm_features.pkl")
