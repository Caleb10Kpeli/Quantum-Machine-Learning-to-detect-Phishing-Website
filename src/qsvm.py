"""
Quantum Support Vector Machine (QSVM) for phishing detection.
Uses PennyLane quantum kernel with angle embedding + entanglement.

Architecture:
  - Feature map U(x): AngleEmbedding (RX) -> CNOT chain -> AngleEmbedding (RZ)
  - Kernel: K(x,x') = |<0|U(x')† U(x)|0>|²
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

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS    = 6    # features fed into quantum circuit (1 qubit per feature)
N_TRAIN     = 100  # samples per class for training  -> 200 total
N_TEST      = 50   # samples per class for testing   -> 100 total
C_PARAM     = 10   # SVM regularisation — higher = more flexible boundary
RANDOM_SEED = 42

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_data.csv.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 55)
print("  QSVM Training -- Hybrid QML Phishing Detection")
print("=" * 55)
print("\n[1/8] Loading dataset...")
df = pd.read_csv(DATA_PATH)
ALL_FEATURES = [c for c in df.columns if c not in ('url', 'status')]

le    = LabelEncoder()
y_all = le.fit_transform(df['status'])          # legitimate=0, phishing=1
X_all = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

print(f"      {len(df)} samples | classes: {dict(zip(le.classes_, [0,1]))}")

# ── 2. Select top N_QUBITS features from ALL 87 via mutual information ────────
print(f"\n[2/8] Selecting top {N_QUBITS} features from all 87 (mutual information)...")
mi_scores    = mutual_info_classif(X_all, y_all, random_state=RANDOM_SEED)
top_idx      = np.argsort(mi_scores)[::-1][:N_QUBITS]
TOP_FEATURES = [ALL_FEATURES[i] for i in top_idx]

print("      Selected features:")
for rank, (feat, idx) in enumerate(zip(TOP_FEATURES, top_idx)):
    print(f"        {rank+1}. {feat:35s}  MI score: {mi_scores[idx]:.4f}")

X_selected = X_all[TOP_FEATURES].values

# ── 3. Balanced subset ────────────────────────────────────────────────────────
print(f"\n[3/8] Building balanced subset ({N_TRAIN*2} train / {N_TEST*2} test)...")
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

# ── 4. Fair classical SVM comparison (same data, same features) ───────────────
# This is the honest comparison: QSVM vs Classical SVM on identical conditions.
print("\n[4/8] Training fair classical SVM (same 6 features, same 200 samples)...")
from sklearn.preprocessing import StandardScaler

scaler_cls  = StandardScaler()
X_train_cls = scaler_cls.fit_transform(X_train_raw)
X_test_cls  = scaler_cls.transform(X_test_raw)

cls_svm = SVC(kernel='rbf', C=C_PARAM, gamma='scale', random_state=RANDOM_SEED)
cls_svm.fit(X_train_cls, y_train)
cls_pred = cls_svm.predict(X_test_cls)

cls_acc  = accuracy_score(y_test, cls_pred)
cls_prec = precision_score(y_test, cls_pred)
cls_rec  = recall_score(y_test, cls_pred)
cls_f1   = f1_score(y_test, cls_pred)
print(f"      Classical SVM (fair): Accuracy={cls_acc:.4f}  Precision={cls_prec:.4f}  "
      f"Recall={cls_rec:.4f}  F1={cls_f1:.4f}")

# ── 5. Scale features to [0, pi] for quantum angle encoding ──────────────────
print("\n[5/8] Scaling features to [0, pi] for angle encoding...")
scaler_q = MinMaxScaler(feature_range=(0, np.pi))
X_train  = scaler_q.fit_transform(X_train_raw)
X_test   = scaler_q.transform(X_test_raw)
print("      Done.")

# ── 6. Build quantum kernel circuit ───────────────────────────────────────────
print(f"\n[6/8] Building {N_QUBITS}-qubit quantum kernel circuit...")

dev = qml.device("default.qubit", wires=N_QUBITS)

def feature_map(x):
    """
    ZZ-style feature map (2-layer):
      RX rotations -> CNOT entanglement -> RZ rotations -> reverse CNOT
    Two layers give the kernel more expressive power to capture
    non-linear class boundaries in the quantum feature space.
    """
    # Layer 1
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='X')
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    # Layer 2
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation='Z')
    for i in range(N_QUBITS - 2, -1, -1):
        qml.CNOT(wires=[i + 1, i])

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Applies U(x1) then U†(x2).
    P(|00...0>) = |<phi(x2)|phi(x1)>|^2 = kernel value K(x1, x2).
    """
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))

def quantum_kernel(x1, x2):
    return float(kernel_circuit(x1, x2)[0])

print(f"      Circuit: RX embed -> CNOT -> RZ embed -> reverse CNOT -> adjoint (2 layers)")
print(f"      Device:  PennyLane default.qubit simulator")
print(f"      C param: {C_PARAM}")

# ── 7. Compute kernel matrices ────────────────────────────────────────────────
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

print(f"\n[7/8] Computing kernel matrices...")
print(f"      Training: {N_TRAIN*2} x {N_TRAIN*2} = {(N_TRAIN*2)**2:,} circuit evals")
print(f"      Test:     {N_TEST*2} x {N_TRAIN*2} = {N_TEST*2*N_TRAIN*2:,} circuit evals\n")

t_start = time.time()
K_train = compute_kernel_matrix(X_train, X_train, "Train kernel")
K_test  = compute_kernel_matrix(X_test,  X_train, "Test kernel ")
print(f"\n      Kernel matrices done in {time.time()-t_start:.1f}s")

# ── 8. Train QSVM & evaluate ──────────────────────────────────────────────────
print("\n[8/8] Training QSVM on precomputed quantum kernel...")
qsvm = SVC(kernel='precomputed', C=C_PARAM, random_state=RANDOM_SEED)
qsvm.fit(K_train, y_train)

y_pred    = qsvm.predict(K_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Final Comparison")
print("=" * 55)
print(f"  {'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print(f"  {'-'*54}")
print(f"  {'Classical SVM (full, 9144 samp)':<30} {'95.41%':>7} {'95.77%':>7} {'95.01%':>7} {'95.39%':>7}")
print(f"  {'Classical SVM (fair, 200 samp)':<30} {cls_acc*100:>6.2f}% {cls_prec*100:>6.2f}% {cls_rec*100:>6.2f}% {cls_f1*100:>6.2f}%")
print(f"  {'QSVM (quantum kernel)':<30} {accuracy*100:>6.2f}% {precision*100:>6.2f}% {recall*100:>6.2f}% {f1*100:>6.2f}%")
print("=" * 55)
print(f"  Features used: {N_QUBITS}  |  Training samples: {N_TRAIN*2}  |  C: {C_PARAM}")
print("=" * 55)

print("\nQSVM Detailed Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save artifacts ────────────────────────────────────────────────────────────
print("Saving model artifacts...")
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(qsvm,         os.path.join(MODELS_DIR, 'qsvm.pkl'))
joblib.dump(scaler_q,     os.path.join(MODELS_DIR, 'qsvm_scaler.pkl'))
joblib.dump(TOP_FEATURES, os.path.join(MODELS_DIR, 'qsvm_features.pkl'))
np.save(os.path.join(MODELS_DIR, 'qsvm_X_train.npy'), X_train)

print("Saved: qsvm.pkl | qsvm_scaler.pkl | qsvm_features.pkl | qsvm_X_train.npy")
print(f"Total time: {time.time()-t_start:.1f}s")
