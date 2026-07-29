"""
QNN v2 -- Data Re-uploading variational circuit.
Same qubits, layers, parameter count, data split and training budget as the
baseline QNN (src/qnn.py). The ONLY architectural change: AngleEmbedding is
re-applied at the START of every variational layer instead of once at the
very start of the circuit (Perez-Salinas et al., "Data re-uploading for a
universal quantum classifier", 2020). Trainable parameter count is
IDENTICAL (54 = 3 layers x 6 qubits x 3 Rot angles) -- the comparison
isolates the effect of re-uploading, not extra capacity.

Architecture:
  - Per layer   : AngleEmbedding (RX, re-uploaded) -> Rot(theta,phi,omega) x 6 -> ring CNOT
  - Output      : Expval(PauliZ(0)) -> sign -> binary label
  - Gradient    : backprop through default.qubit statevector
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

# ── Config (identical to baseline QNN) ─────────────────────────────────────────
N_QUBITS    = 6
N_LAYERS    = 3     # same depth as baseline
N_TRAIN     = 100
N_TEST      = 50
BATCH_SIZE  = 10
N_EPOCHS    = 100
LR          = 0.01
RANDOM_SEED = 42

DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_data.csv.csv')
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_PATH = os.path.join(MODELS_DIR, 'results.json')

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  QNN v2 Training -- Data Re-uploading Variational Circuit")
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
    print(f"        {rank+1}. {feat:35s}  MI: {mi_scores[idx]:.4f}")

X_selected = X_all[TOP_FEATURES].values

# ── 3. Identical balanced split (same seed => same rows as baseline QNN) ──────
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
scaler  = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

y_train_qnn = np.where(y_train == 1,  1.0, -1.0)
y_test_qnn  = np.where(y_test  == 1,  1.0, -1.0)
print("      Done.")

# ── 5. Build re-uploading variational circuit ─────────────────────────────────
print(f"\n[5/7] Building {N_QUBITS}-qubit re-uploading QNN ({N_LAYERS} layers)...")

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="backprop")
def circuit(inputs, weights):
    """
    Data re-uploading circuit: the baseline embeds the input ONCE before
    all layers. Here, AngleEmbedding is re-applied at the start of EVERY
    layer, interleaved with the same trainable Rot+ring-CNOT block used
    in the baseline. Same 54 trainable parameters -- only the re-encoding
    schedule changes.
    """
    for layer in range(N_LAYERS):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='X')
        for qubit in range(N_QUBITS):
            qml.Rot(
                weights[layer, qubit, 0],
                weights[layer, qubit, 1],
                weights[layer, qubit, 2],
                wires=qubit,
            )
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])
    return qml.expval(qml.PauliZ(0))

weights_shape = (N_LAYERS, N_QUBITS, 3)
n_params      = N_LAYERS * N_QUBITS * 3
print(f"      Trainable parameters : {n_params}  (same as baseline -- {N_LAYERS} layers x {N_QUBITS} wires x 3)")
print(f"      Re-uploads            : input re-embedded before each of the {N_LAYERS} layers")
print(f"      Diff method            : backprop (fast statevector autodiff)")

# ── 6. Train ──────────────────────────────────────────────────────────────────
print(f"\n[6/7] Training -- {N_EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}...")

np.random.seed(RANDOM_SEED)
weights = pnp.array(
    np.random.uniform(-np.pi, np.pi, weights_shape),
    requires_grad=True,
)

opt = qml.AdamOptimizer(stepsize=LR)

def batch_cost(weights, X_b, y_b):
    preds = pnp.stack([
        circuit(pnp.array(x, requires_grad=False), weights)
        for x in X_b
    ])
    return pnp.mean((preds - y_b) ** 2)

loss_history = []
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    perm = np.random.permutation(len(X_train))
    X_sh = X_train[perm]
    y_sh = y_train_qnn[perm]
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_sh), BATCH_SIZE):
        Xb = X_sh[i:i + BATCH_SIZE]
        yb = y_sh[i:i + BATCH_SIZE]
        weights, loss_val = opt.step_and_cost(lambda w: batch_cost(w, Xb, yb), weights)
        epoch_loss += float(loss_val)
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
        train_preds = np.sign([
            float(circuit(pnp.array(x, requires_grad=False), weights))
            for x in X_train
        ])
        train_acc = np.mean(train_preds == y_train_qnn)
        elapsed   = time.time() - t0
        eta       = (elapsed / epoch) * (N_EPOCHS - epoch)
        print(f"      Epoch {epoch:3d}/{N_EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"train_acc={train_acc*100:.1f}%  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

total_time = time.time() - t0
print(f"\n      Training complete in {total_time:.1f}s")

# ── 7. Evaluate on test set ────────────────────────────────────────────────────
print("\n[7/7] Evaluating on test set...")
raw_preds = np.array([
    float(circuit(pnp.array(x, requires_grad=False), weights))
    for x in X_test
])
y_pred = np.where(raw_preds >= 0, 1, 0)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  QNN v2 (Data Re-uploading) -- Final Results")
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
np.save(os.path.join(MODELS_DIR, 'qnn_reupload_weights.npy'),      np.array(weights))
np.save(os.path.join(MODELS_DIR, 'qnn_reupload_loss_history.npy'), np.array(loss_history))
joblib.dump(scaler,       os.path.join(MODELS_DIR, 'qnn_reupload_scaler.pkl'))
joblib.dump(TOP_FEATURES, os.path.join(MODELS_DIR, 'qnn_reupload_features.pkl'))

with open(RESULTS_PATH) as f:
    results = json.load(f)

results['qnn_reupload'] = {
    "label":       "QNN v2 (Data Re-uploading)",
    "samples":     N_TRAIN * 2,
    "features":    N_QUBITS,
    "qubits":      N_QUBITS,
    "layers":      N_LAYERS,
    "epochs":      N_EPOCHS,
    "accuracy":    round(accuracy,  4),
    "precision":   round(precision, 4),
    "recall":      round(recall,    4),
    "f1":          round(f1,        4),
    "description": (
        f"Same 3-layer Rot+ring-CNOT circuit and 54 parameters as the baseline QNN, "
        f"but the input is re-uploaded (Perez-Salinas et al. 2020) before every layer "
        f"instead of once at the start. Same data split, same {N_EPOCHS}-epoch Adam "
        f"training. Isolates the effect of re-uploading on the same 200 training samples."
    ),
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print("Saved: qnn_reupload_weights.npy | qnn_reupload_scaler.pkl | qnn_reupload_features.pkl | qnn_reupload_loss_history.npy")
print("Updated: results.json (key: qnn_reupload)")
print(f"Total time: {total_time:.1f}s")
