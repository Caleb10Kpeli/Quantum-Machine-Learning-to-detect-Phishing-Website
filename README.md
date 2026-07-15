# Hybrid QML Phishing Website Detector

Final year project comparing a classical SVM against quantum machine learning
models (QSVM, QNN) for phishing URL detection, with a Flask app for live
URL analysis and a Streamlit app for the URL analyser / model comparison views.

## Models

| Model | Samples | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Classical SVM (full) | 9,144 | 87 | 95.41% | 95.77% | 95.01% | 95.39% |
| Classical SVM (fair) | 200 | 6 | 86.00% | 83.33% | 90.00% | 86.54% |
| QSVM (quantum kernel) | 200 | 6 | 86.00% | 84.62% | 88.00% | 86.27% |
| QNN (variational) | 200 | 6 | 89.00% | 84.21% | 96.00% | 89.72% |

The "fair" classical SVM, QSVM, and QNN are all trained on the same 200
samples and same 6 mutual-info-selected features (`google_index`,
`web_traffic`, `domain_age`, `ratio_intHyperlinks`, `ratio_extHyperlinks`,
`nb_hyperlinks`) for a head-to-head comparison. The full classical SVM (87
features, 9,144 samples) is the model used in the Streamlit URL analyser.

Baseline (ablation) quantum architectures — a simpler single-layer feature
map QSVM and a `BasicEntanglerLayers` QNN — are in `models/results_baseline.json`,
for comparing against the improved feature maps above.

## Running

**Flask app** (live URL analyser, all three fair-comparison models):
```
cd flask_app
python app.py
```
Serves at http://127.0.0.1:5000.

**Streamlit app** (full classical SVM URL analyser + model comparison tab):
```
streamlit run app.py
```

Both need `pennylane`, `flask`/`streamlit`, `scikit-learn`, `joblib`, `numpy`,
`pandas`, `requests`, `beautifulsoup4`, and optionally `whois` installed.
No `requirements.txt` yet.

## Quantum execution

QSVM and QNN run real PennyLane circuits (angle embedding + entanglement,
quantum kernel evaluation / variational forward pass) on
`qml.device("default.qubit")` — an exact, noiseless **classical simulator**,
not real quantum hardware.

## Supervisor demo: live example

`src/find_disagreements.py` reconstructs the train/test split and finds
test-set cases where the Classical (Fair) SVM misclassifies a phishing site
that a quantum model catches. Verified live (2026-07-14 and 2026-07-15):

**`http://www.pracadarepublicaembeja.net/men`** — true label: phishing
- Classical (Fair) SVM: predicts **legitimate** (wrong)
- QSVM: predicts **legitimate** (wrong, agrees with Classical)
- QNN: predicts **phishing** (correct) — but only ~4% confidence, right at
  the decision boundary, not a landslide win

Two other candidates from the same disagreement scan did not hold up when
re-tested live: `albel.intnet.mu/...` is now dead (site down), and
`re-redirection-pp-account-id98763432.blogspot.com` is now correctly caught
by Classical too (its content likely changed since the training dataset
snapshot was taken). Phishing URLs sourced from `data/phishing_data.csv.csv`
are a snapshot in time — always re-verify liveness against the Flask app's
`/analyse` endpoint before using one in a live demo.
