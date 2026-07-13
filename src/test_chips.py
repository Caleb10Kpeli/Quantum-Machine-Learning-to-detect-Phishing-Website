"""
Tests all sample chip URLs through all 3 models.
Run: python src/test_chips.py
Flask server does NOT need to be running.
"""

import os, sys, json, joblib, numpy as np, requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pennylane as qml

try:
    import whois as python_whois
    from dateutil import parser as dateparser
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

BASE       = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE, 'models')

# ── Load models ───────────────────────────────────────────────────────────────
fair_svm          = joblib.load(os.path.join(MODELS_DIR, 'fair_svm.pkl'))
fair_svm_scaler   = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_scaler.pkl'))
fair_svm_features = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_features.pkl'))

qsvm_model    = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
qsvm_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qsvm_scaler.pkl'))
qsvm_features = joblib.load(os.path.join(MODELS_DIR, 'qsvm_features.pkl'))
qsvm_X_train  = np.load(os.path.join(MODELS_DIR, 'qsvm_X_train.npy'))

qnn_weights  = np.load(os.path.join(MODELS_DIR, 'qnn_weights.npy'))
qnn_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qnn_scaler.pkl'))
qnn_features = joblib.load(os.path.join(MODELS_DIR, 'qnn_features.pkl'))

NQ = len(qsvm_features)
dev_qsvm = qml.device("default.qubit", wires=NQ)
dev_qnn  = qml.device("default.qubit", wires=NQ)

def _fmap(x):
    qml.AngleEmbedding(x, wires=range(NQ), rotation='X')
    for i in range(NQ - 1): qml.CNOT(wires=[i, i + 1])
    qml.AngleEmbedding(x, wires=range(NQ), rotation='Z')
    for i in range(NQ - 2, -1, -1): qml.CNOT(wires=[i + 1, i])

@qml.qnode(dev_qsvm)
def _qk_circuit(x1, x2):
    _fmap(x1); qml.adjoint(_fmap)(x2)
    return qml.probs(wires=range(NQ))

def qk(x1, x2):
    return float(_qk_circuit(np.array(x1, dtype=float), np.array(x2, dtype=float))[0])

@qml.qnode(dev_qnn)
def qnn_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(NQ), rotation='X')
    for layer in range(weights.shape[0]):
        for q in range(NQ):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        for q in range(NQ):
            qml.CNOT(wires=[q, (q + 1) % NQ])
    return qml.expval(qml.PauliZ(0))

# ── Neutral defaults ──────────────────────────────────────────────────────────
_NEUTRAL = {'web_traffic': 799374, 'domain_age': 3893}

def fetch_features(url, feature_list):
    vals      = {f: float(_NEUTRAL.get(f, 0.0)) for f in feature_list}
    estimated = set()
    parsed    = urlparse(url if url.startswith('http') else 'http://' + url)
    domain    = parsed.netloc.lstrip('www.') or url.split('/')[0]

    try:
        resp = requests.get(url, timeout=8,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            allow_redirects=True)
        soup  = BeautifulSoup(resp.text, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]
        total = len(hrefs)
        int_l = sum(1 for h in hrefs if h.startswith('/') or domain in h)
        ext_l = total - int_l
        if 'nb_hyperlinks'       in vals: vals['nb_hyperlinks']       = total
        if 'ratio_intHyperlinks' in vals: vals['ratio_intHyperlinks'] = int_l / total if total else 0
        if 'ratio_extHyperlinks' in vals: vals['ratio_extHyperlinks'] = ext_l / total if total else 0
        if 'google_index'        in vals: vals['google_index']        = 1
        link_ok = True
    except Exception:
        for f in ['nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks', 'google_index']:
            if f in vals: estimated.add(f)
        link_ok = False

    if 'domain_age' in vals:
        if WHOIS_OK:
            try:
                w       = python_whois.whois(domain)
                created = w.creation_date
                if isinstance(created, list): created = created[0]
                if created and not isinstance(created, datetime):
                    try: created = dateparser.parse(str(created))
                    except Exception: created = None
                if created and isinstance(created, datetime):
                    now = datetime.now(tz=timezone.utc) if created.tzinfo else datetime.now()
                    vals['domain_age'] = max(0, (now - created).days)
                else:
                    estimated.add('domain_age')
            except Exception:
                estimated.add('domain_age')
        else:
            estimated.add('domain_age')

    if 'web_traffic' in vals:
        estimated.add('web_traffic')

    return vals, estimated, link_ok

def predict_classical(vals):
    x = np.array([[vals[f] for f in fair_svm_features]])
    x_s = fair_svm_scaler.transform(x)
    p = fair_svm.predict(x_s)[0]
    s = fair_svm.decision_function(x_s)[0]
    label = 'PHISHING' if p == 1 else 'legitimate'
    conf  = round(min(abs(s) / 3 * 100, 99), 1)
    return label, conf

def predict_qsvm(vals):
    x = np.array([[vals[f] for f in qsvm_features]])
    x_s = qsvm_scaler.transform(x)[0]
    kernel_row = np.array([qk(x_s, xt) for xt in qsvm_X_train])
    p = qsvm_model.predict(kernel_row.reshape(1, -1))[0]
    d = float(qsvm_model.decision_function(kernel_row.reshape(1, -1))[0])
    label = 'PHISHING' if p == 1 else 'legitimate'
    conf  = round(min(abs(d) / 2 * 100, 99), 1)
    return label, conf

def predict_qnn(vals):
    x = np.array([[vals[f] for f in qnn_features]])
    x_s = np.clip(qnn_scaler.transform(x)[0], 0, np.pi)
    raw = float(qnn_circuit(x_s, qnn_weights))
    label = 'PHISHING' if raw >= 0 else 'legitimate'
    conf  = round(min(abs(raw) * 100, 99), 1)
    return label, conf, raw

# ── Chips to test ─────────────────────────────────────────────────────────────
CHIPS = [
    ('BBC',           'https://www.bbc.com'),
    ('Stack Overflow','https://stackoverflow.com'),
    ('GitHub',        'https://www.github.com'),
    ('Wikipedia',     'https://www.wikipedia.org'),
    ('Amazon',        'https://www.amazon.com'),
]

print()
print('=' * 80)
print('  CHIP URL TEST — all 3 models')
print('=' * 80)

for name, url in CHIPS:
    print(f'\n  [{name}]  {url}')
    vals, estimated, link_ok = fetch_features(url, list(set(fair_svm_features + qsvm_features + qnn_features)))
    print(f'    Links fetched : {int(vals.get("nb_hyperlinks", 0))}   '
          f'int={round(vals.get("ratio_intHyperlinks",0),2)}  '
          f'ext={round(vals.get("ratio_extHyperlinks",0),2)}  '
          f'domain_age={int(vals.get("domain_age",0))}d')
    print(f'    Estimated     : {sorted(estimated) or "none"}')

    cls_label, cls_conf = predict_classical(vals)
    print(f'    Classical(Fair): {cls_label:10s}  {cls_conf:.1f}%')

    print(f'    QSVM           : computing... ', end='', flush=True)
    q_label, q_conf = predict_qsvm(vals)
    print(f'{q_label:10s}  {q_conf:.1f}%')

    qnn_label, qnn_conf, qnn_raw = predict_qnn(vals)
    flag = '  *** FALSE POSITIVE ***' if qnn_label == 'PHISHING' else ''
    print(f'    QNN            : {qnn_label:10s}  {qnn_conf:.1f}%  (raw={qnn_raw:.3f}){flag}')

print()
print('=' * 80)
print('  Done.')
print('=' * 80)
