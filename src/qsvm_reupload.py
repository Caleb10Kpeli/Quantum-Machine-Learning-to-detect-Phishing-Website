"""
QSVM v2 -- Data Re-uploading feature map.
Same data split, same qubit count, same C as the baseline QSVM (src/qsvm.py).
The ONLY architectural change: the encode-and-entangle block is applied
N_REPS=2 times instead of once, re-uploading the same classical data deeper
into the circuit (Perez-Salinas et al., "Data re-uploading for a universal
quantum classifier", 2020). This is a controlled ablation, not a new model
family -- everything else is held fixed so the accuracy delta can be
attributed to re-uploading alone.

Architecture:
  - Feature map U(x): [AngleEmbedding (RX) -> CNOT chain -> AngleEmbedding (RZ) -> reverse CNOT] x N_REPS
  - Kernel: K(x,x') = |<0|U(x')' U(x)|0>|^2
  - Classifier: scikit-learn SVC with precomputed kernel matrix
"""

import os
import time
import numpy as np
import pandas as pd
import pennylane as qml
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

# ── Config (identical to baseline QSVM except N_REPS) ──────────────────────────
N_QUBITS    = 6    # features fed into quantum circuit (1 qubit per feature)
N_REPS      = 2    # NEW: data re-uploading repetitions (baseline = 1)
N_TRAIN     = 100  # samples per class for training  -> 200 total
N_TEST      = 50   # samples per class for testing   -> 100 total
C_PARAM     = 10   # SVM regularisation -- same as baseline
RANDOM_SEED = 42

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_data.csv.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  QSVM v2 Training -- Data Re-uploading Feature Map")
print("=" * 60)
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)
ALL_FEATURES = [c for c in df.columns if c not in ('url', 'status')]

le    = LabelEncoder()
y_all = le.fit_transform(df['status'])
X_all = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
print(f"      {len(df)} samples | classes: {dict(zip(le.classes_, [0, 1]))}")

# ── 2. Select top N_QUBITS features -- identical seed/method to baseline ──────
print(f"\n[2/7] Selecting top {N_QUBITS} features (mutual information)...")
mi_scores    = mutual_info_classif(X_all, y_all, random_state=RANDOM_SEED)
top_idx      = np.argsort(mi_scores)[::-1][:N_QUBITS]
TOP_FEATURES = [ALL_FEATURES[i] for i in top_idx]
for rank, (feat, idx) in enumerate(zip(TOP_FEATURES, top_idx)):
    print(f"        {rank+1}. {feat:35s}  MI score: {mi_scores[idx]:.4f}")

X_selected = X_all[TOP_FEATURES].values

# ── 3. Identical balanced split (same seed => same rows as baseline QSVM) ─────
print(f"\n[3/7] Reconstructing the same balanced subset as baseline "
      f"({N_TRAIN*2} train / {N_TEST*2} test)...")
np.random.seed(RANDOM_SEED)
phish_idx = np.where(y_all == 1)[0]
legit_idx = np.where(y_all == 0)[0]

train_phish = np.random.choice(phish_idx, N_TRAIN, replace=False)
train_legit = np.random.choice(legit_idx, N_TRAIN, replace=False)
test_phish  = np.random.choice(np.setdiff1d(phish_idx, train_phish), N_TEST, replace=False)
test_legit  = np.random.choice(np.setdiff1d(legit_idx, train_legit), N_TEST, replace=False)

train_idx = np.random.permutation(np.concatenate([train_phish, train_legit]))
test_idx  = np.random.permutation(np.concatenate([test_phish,  test_legit]))

X_train_raw, y_train = X_selected[train_idx], y_all[train_idx]
X_test_raw,  y_test  = X_selected[test_idx],  y_all[test_idx]

# ── 4. Scale features to [0, pi] for angle encoding ───────────────────────────
print("\n[4/7] Scaling features to [0, pi] for angle encoding...")
scaler_q = MinMaxScaler(feature_range=(0, np.pi))
X_train  = scaler_q.fit_transform(X_train_raw)
X_test   = scaler_q.transform(X_test_raw)
print("      Done.")

# ── 5. Build re-uploading quantum kernel circuit ──────────────────────────────
print(f"\n[5/7] Building {N_QUBITS}-qubit re-uploading kernel circuit "
      f"({N_REPS} repetitions)...")

dev = qml.device("default.qubit", wires=N_QUBITS)

def feature_map(x):
    """
    Data re-uploading feature map: the baseline's single encode-entangle
    block (RX embed -> CNOT chain -> RZ embed -> reverse CNOT) is repeated
    N_REPS times, re-uploading the SAME classical data at each repetition.
    Repeating the upload interleaved with entanglement lets the circuit
    reach higher-frequency components of the data than a single pass can,
    without adding qubits.
    """
    for _ in range(N_REPS):
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

def quantum_kernel(x1, x2):
    return float(kernel_circuit(x1, x2)[0])

print(f"      Circuit: [RX embed -> CNOT -> RZ embed -> reverse CNOT] x {N_REPS} -> adjoint")
print(f"      Device:  PennyLane default.qubit simulator")
print(f"      C param: {C_PARAM}")

# ── 6. Compute kernel matrices ─────────────────────────────────────────────────
def compute_kernel_matrix(A, B, label):
    n, m = len(A), len(B)
    K = np.zeros((n, m))
    t0 = time.time()
    for i in range(n):
        for j in range(m):
            K[i, j] = quantum_kernel(A[i], B[j])
        elapsed = time.time() - t0
        done    = (i + 1) * m
        total   = n * m
        rate    = done / elapsed if elapsed > 0 else 1
        eta     = (total - done) / rate
        pct     = done / total * 100
        print(f"\r      {label}: {pct:5.1f}%  ({i+1}/{n} rows)  ETA: {eta:4.0f}s",
              end="", flush=True)
    print(f"\r      {label}: 100.0%  ({n}/{n} rows)  done in {time.time()-t0:.1f}s")
    return K

print(f"\n[6/7] Computing kernel matrices...")
print(f"      Training: {N_TRAIN*2} x {N_TRAIN*2} = {(N_TRAIN*2)**2:,} circuit evals")
print(f"      Test:     {N_TEST*2} x {N_TRAIN*2} = {N_TEST*2*N_TRAIN*2:,} circuit evals\n")

t_start = time.time()
K_train = compute_kernel_matrix(X_train, X_train, "Train kernel")
K_test  = compute_kernel_matrix(X_test,  X_train, "Test kernel ")
print(f"\n      Kernel matrices done in {time.time()-t_start:.1f}s")

# ── 7. Train QSVM v2 & evaluate ────────────────────────────────────────────────
print("\n[7/7] Training QSVM v2 on precomputed quantum kernel...")
qsvm_v2 = SVC(kernel='precomputed', C=C_PARAM, random_state=RANDOM_SEED)
qsvm_v2.fit(K_train, y_train)

y_pred    = qsvm_v2.predict(K_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  QSVM v2 (Data Re-uploading) -- Final Results")
print("=" * 60)
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall   : {recall*100:.2f}%")
print(f"  F1       : {f1*100:.2f}%")
print("=" * 60)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save artifacts (new filenames -- baseline untouched) ──────────────────────
print("Saving model artifacts...")
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(qsvm_v2,      os.path.join(MODELS_DIR, 'qsvm_reupload.pkl'))
joblib.dump(scaler_q,     os.path.join(MODELS_DIR, 'qsvm_reupload_scaler.pkl'))
joblib.dump(TOP_FEATURES, os.path.join(MODELS_DIR, 'qsvm_reupload_features.pkl'))
np.save(os.path.join(MODELS_DIR, 'qsvm_reupload_X_train.npy'), X_train)

# ── Update results.json ────────────────────────────────────────────────────────
import json
RESULTS_PATH = os.path.join(MODELS_DIR, 'results.json')
with open(RESULTS_PATH) as f:
    results = json.load(f)

results['qsvm_reupload'] = {
    "label":       "QSVM v2 (Data Re-uploading)",
    "samples":     N_TRAIN * 2,
    "features":    N_QUBITS,
    "qubits":      N_QUBITS,
    "reps":        N_REPS,
    "accuracy":    round(accuracy,  4),
    "precision":   round(precision, 4),
    "recall":      round(recall,    4),
    "f1":          round(f1,        4),
    "description": (
        f"Same 6-qubit ZZ-style feature map as the baseline QSVM, but the "
        f"encode-entangle block is re-uploaded {N_REPS}x (Perez-Salinas et al. "
        f"2020 data re-uploading) instead of once. Same data split, same C={C_PARAM}. "
        f"Isolates the effect of feature-map depth on the same 200 training samples."
    ),
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print("Saved: qsvm_reupload.pkl | qsvm_reupload_scaler.pkl | qsvm_reupload_features.pkl | qsvm_reupload_X_train.npy")
print("Updated: results.json (key: qsvm_reupload)")
print(f"Total time: {time.time()-t_start:.1f}s")
