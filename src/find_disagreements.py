"""
Finds test-set URLs where the Classical SVM (Fair) mislabels a phishing site
but the QSVM and/or QNN correctly catches it.
Prints a ready-to-use table for the supervisor demo.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

N_QUBITS    = 6
N_TRAIN     = 100
N_TEST      = 50
RANDOM_SEED = 42

BASE        = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH   = os.path.join(BASE, 'data', 'phishing_data.csv.csv')
MODELS_DIR  = os.path.join(BASE, 'models')

# ── 1. Reconstruct identical train/test split ─────────────────────────────────
print("Loading dataset and reconstructing split...")
df          = pd.read_csv(DATA_PATH)
all_feats   = [c for c in df.columns if c not in ('url', 'status')]
le          = LabelEncoder()
y_all       = le.fit_transform(df['status'])           # legitimate=0, phishing=1
X_all       = df[all_feats].fillna(df[all_feats].median())

mi_scores    = mutual_info_classif(X_all, y_all, random_state=RANDOM_SEED)
top_idx      = np.argsort(mi_scores)[::-1][:N_QUBITS]
TOP_FEATURES = [all_feats[i] for i in top_idx]
X_selected   = X_all[TOP_FEATURES].values

np.random.seed(RANDOM_SEED)
phish_idx   = np.where(y_all == 1)[0]
legit_idx   = np.where(y_all == 0)[0]
train_phish = np.random.choice(phish_idx, N_TRAIN, replace=False)
train_legit = np.random.choice(legit_idx, N_TRAIN, replace=False)
test_phish  = np.random.choice(np.setdiff1d(phish_idx, train_phish), N_TEST, replace=False)
test_legit  = np.random.choice(np.setdiff1d(legit_idx, train_legit), N_TEST, replace=False)
train_idx   = np.random.permutation(np.concatenate([train_phish, train_legit]))
test_idx    = np.random.permutation(np.concatenate([test_phish,  test_legit]))

X_train_raw = X_selected[train_idx]
X_test_raw  = X_selected[test_idx]
y_test      = y_all[test_idx]
test_urls   = df['url'].iloc[test_idx].values if 'url' in df.columns else np.array([f"sample_{i}" for i in test_idx])

print(f"  Test set: {len(y_test)} samples ({y_test.sum()} phishing, {(y_test==0).sum()} legitimate)")
print(f"  Features: {TOP_FEATURES}\n")

# ── 2. Classical SVM (Fair) predictions ──────────────────────────────────────
print("Running Classical SVM (Fair)...")
fair_svm    = joblib.load(os.path.join(MODELS_DIR, 'fair_svm.pkl'))
fair_scaler = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_scaler.pkl'))
X_test_cls  = fair_scaler.transform(X_test_raw)
cls_pred    = fair_svm.predict(X_test_cls)
cls_score   = fair_svm.decision_function(X_test_cls)
print(f"  Accuracy: {(cls_pred == y_test).mean()*100:.1f}%\n")

# ── 3. QSVM predictions ───────────────────────────────────────────────────────
print("Running QSVM (computing quantum kernel — takes ~1 min)...")
qsvm        = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
qsvm_scaler = joblib.load(os.path.join(MODELS_DIR, 'qsvm_scaler.pkl'))
X_train_q   = np.load(os.path.join(MODELS_DIR, 'qsvm_X_train.npy'))
X_test_q    = qsvm_scaler.transform(X_test_raw)

dev = qml.device("default.qubit", wires=N_QUBITS)

def feature_map(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='X')
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='Z')
    for i in range(N_QUBITS - 2, -1, -1):
        qml.CNOT(wires=[i + 1, i])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))

def qk(x1, x2):
    return float(kernel_circuit(np.array(x1, dtype=float), np.array(x2, dtype=float))[0])

CACHE = os.path.join(MODELS_DIR, 'qsvm_K_test_cache.npy')
if os.path.exists(CACHE):
    print("  Loading cached kernel matrix...")
    K_test = np.load(CACHE)
else:
    n_test, n_train = len(X_test_q), len(X_train_q)
    K_test = np.zeros((n_test, n_train))
    t0 = time.time()
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = qk(X_test_q[i], X_train_q[j])
        pct = (i + 1) / n_test * 100
        eta = (time.time() - t0) / (i + 1) * (n_test - i - 1)
        print(f"\r  {pct:5.1f}%  ETA {eta:4.0f}s", end="", flush=True)
    print()
    np.save(CACHE, K_test)
    print("  Kernel cached for future runs.")

qsvm_pred  = qsvm.predict(K_test)
qsvm_score = qsvm.decision_function(K_test)
print(f"  Accuracy: {(qsvm_pred == y_test).mean()*100:.1f}%\n")

# ── 4. QNN predictions ────────────────────────────────────────────────────────
print("Running QNN...")
qnn_weights  = np.load(os.path.join(MODELS_DIR, 'qnn_weights.npy'))
qnn_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qnn_scaler.pkl'))
X_test_qnn   = qnn_scaler.transform(X_test_raw)

dev_qnn = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev_qnn)
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='X')
    for layer in range(weights.shape[0]):
        for qubit in range(N_QUBITS):
            qml.Rot(weights[layer, qubit, 0],
                    weights[layer, qubit, 1],
                    weights[layer, qubit, 2], wires=qubit)
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])
    return qml.expval(qml.PauliZ(0))

raw_preds = np.array([float(qnn_circuit(X_test_qnn[i], qnn_weights)) for i in range(len(X_test_qnn))])
qnn_pred  = (raw_preds >= 0).astype(int)
print(f"  Accuracy: {(qnn_pred == y_test).mean()*100:.1f}%\n")

# ── 5. Find disagreements ─────────────────────────────────────────────────────
def label(p): return "PHISHING  " if p == 1 else "legitimate"

print("=" * 90)
print("  PHISHING sites the Classical SVM (Fair) MISSES but quantum models CATCH")
print("=" * 90)

found = 0
for i in range(len(y_test)):
    true     = y_test[i]
    cls_p    = cls_pred[i]
    qsvm_p   = qsvm_pred[i]
    qnn_p    = qnn_pred[i]

    # Only show: true phishing, classical WRONG, at least one quantum RIGHT
    if true == 1 and cls_p == 0 and (qsvm_p == 1 or qnn_p == 1):
        found += 1
        url = test_urls[i] if test_urls is not None else f"sample_{i}"
        print(f"\n  [{found}] {url}")
        print(f"       True label      : PHISHING")
        print(f"       Classical (Fair): {label(cls_p)}  << MISSED")
        print(f"       QSVM            : {label(qsvm_p)}  {'[caught]' if qsvm_p==1 else '[also missed]'}")
        print(f"       QNN             : {label(qnn_p)}  {'[caught]' if qnn_p==1 else '[also missed]'}")
        feat_vals = {f: round(X_test_raw[i, fi], 4) for fi, f in enumerate(TOP_FEATURES)}
        print(f"       Features        : {feat_vals}")

if found == 0:
    print("  No cases found where classical misses but quantum catches.")
    print("  (All models agree on this test set — try the reverse below.)\n")

print()
print("=" * 90)
print("  All model disagreements (any model differs from others)")
print("=" * 90)
any_found = 0
for i in range(len(y_test)):
    preds = [cls_pred[i], qsvm_pred[i], qnn_pred[i]]
    if len(set(preds)) > 1:
        any_found += 1
        url = test_urls[i] if test_urls is not None else f"sample_{i}"
        print(f"\n  [{any_found}] {url}")
        print(f"       True            : {label(y_test[i])}")
        print(f"       Classical (Fair): {label(cls_pred[i])}")
        print(f"       QSVM            : {label(qsvm_pred[i])}")
        print(f"       QNN             : {label(qnn_pred[i])}")

print(f"\n  Total disagreements: {any_found} / {len(y_test)} test samples")
