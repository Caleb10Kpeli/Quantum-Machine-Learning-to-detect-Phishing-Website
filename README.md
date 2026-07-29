# Hybrid QML Phishing Website Detector

Final year project comparing a classical SVM against quantum machine learning
models (QSVM, QNN) for phishing URL detection, with a Flask app for live
URL analysis and a Streamlit app for the URL analyser / model comparison views.

**Live demo:** https://qml-phishing-detector.onrender.com
(free tier — sleeps after 15 min idle, first load after that takes ~30-50s)

## Models

| Model | Samples | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Classical SVM (full) | 9,144 | 87 | 95.41% | 95.77% | 95.01% | 95.39% |
| Classical SVM (fair) | 200 | 6 | 86.00% | 83.33% | 90.00% | 86.54% |
| QSVM (quantum kernel) | 200 | 6 | 86.00% | 84.62% | 88.00% | 86.27% |
| QNN (variational) | 200 | 6 | 89.00% | 84.21% | 96.00% | 89.72% |
| QSVM v2 (data re-uploading) | 200 | 6 | 87.00% | 84.91% | 90.00% | 87.38% |
| QNN v2 (data re-uploading) | 200 | 6 | 88.00% | 85.19% | 92.00% | 88.46% |

The "fair" classical SVM, QSVM, and QNN are all trained on the same 200
samples and same 6 mutual-info-selected features (`google_index`,
`web_traffic`, `domain_age`, `ratio_intHyperlinks`, `ratio_extHyperlinks`,
`nb_hyperlinks`) for a head-to-head comparison. The full classical SVM (87
features, 9,144 samples) is the model used in the Streamlit URL analyser.

Baseline (ablation) quantum architectures — a simpler single-layer feature
map QSVM and a `BasicEntanglerLayers` QNN — are in `models/results_baseline.json`,
for comparing against the improved feature maps above.

**v2 models (`src/qsvm_reupload.py`, `src/qnn_reupload.py`)** apply one
isolated, named technique — data re-uploading (Pérez-Salinas et al., 2020) —
on top of the QSVM/QNN above, now treated as the baseline: the QSVM's
encode-entangle block runs twice instead of once, and the QNN re-embeds its
input before every layer instead of only at the start. Everything else
(qubits, data split, training budget, and for the QNN, parameter count) is
held fixed. Re-uploading improved the QSVM on every metric but slightly
hurt the QNN — a controlled result showing the same technique doesn't
transfer uniformly across model families.

## Running

**Flask app** (live URL analyser — 5 models, Site Health check, and a
Model Consensus dashboard):
```
cd flask_app
python app.py
```
Serves at http://127.0.0.1:5000. Install dependencies first:
```
pip install -r requirements.txt
```

**Streamlit app** (full classical SVM URL analyser + model comparison tab):
```
streamlit run app.py
```

### Deploying (Render)

`render.yaml` at the repo root is a Render Blueprint — connect the repo at
render.com, choose "New Blueprint", and it deploys the Flask app with
gunicorn automatically. `.python-version` pins the Python version to match
what's tested locally.

## Flask app features

- **URL Analyser** — pick a model (Classical, QSVM, QNN, QSVM v2, QNN v2)
  and analyse a single URL, with a live feature breakdown and a
  borderline-confidence flag (<25%) recommending manual review.
- **Site Health check** — before/instead of classifying, checks DNS
  resolution, HTTP reachability, the redirect chain, and the SSL
  certificate's issuer/age — flags very-recently-issued certs as a
  (weak, caveated) signal. Useful for confirming a URL is actually live
  before trusting a prediction.
- **Model Consensus dashboard** — fetches a URL's live content once and
  runs all 5 models against it in one request, highlighting which models
  dissent from the majority vote.

## Quantum execution

QSVM and QNN run real PennyLane circuits (angle embedding + entanglement,
quantum kernel evaluation / variational forward pass) on
`qml.device("default.qubit")` — an exact, noiseless **classical simulator**,
not real quantum hardware.

## Supervisor demo: live examples

`src/find_disagreements.py` reconstructs the train/test split and finds
test-set cases where the Classical (Fair) SVM disagrees with a quantum
model. Two cases verified live, most recently 2026-07-27:

**`http://www.pracadarepublicaembeja.net/men`** — true label: phishing
- Classical (Fair) SVM: predicts **legitimate** (wrong)
- QSVM: predicts **legitimate** (wrong, agrees with Classical)
- QNN: predicts **phishing** (correct) — but only ~4% confidence, right at
  the decision boundary, not a landslide win

**`https://www.walmart.com/cp/ipods-mp3-players/96469`** — true label: legitimate
- Classical (Fair) SVM: predicts **phishing** (wrong, ~2% confidence)
- QSVM: predicts **phishing** (wrong)
- QNN: predicts **legitimate** (correct)

This second case was found live, independent of the dataset snapshot — the
frozen CSV's feature values for this URL actually show the opposite result,
which is itself a reminder that live content drifts from any fixed dataset.

Other candidates from the same disagreement scan did not hold up when
re-tested live: `albel.intnet.mu/...` is now dead, and several others
(`natuerlich-netzwerk.de`, `truma.no`, the `re-redirection-pp-account...`
blogspot URL) either now redirect elsewhere or are correctly caught by every
model, so the disagreement no longer holds. Phishing URLs sourced from
`data/phishing_data.csv.csv` are a snapshot in time — always re-verify
liveness and predictions against the Flask app's `/analyse` or
`/analyse-all` endpoint before using one in a live demo.
