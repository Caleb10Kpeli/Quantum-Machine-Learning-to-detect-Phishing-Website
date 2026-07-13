import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify

try:
    import whois as python_whois
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_extractor import extract_features, FEATURE_COLUMNS

app = Flask(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── Classical SVM (Full) — used nowhere in tabs, kept for reference ───────────
_svm_full    = joblib.load(os.path.join(MODELS_DIR, 'classical_svm.pkl'))
_scaler_full = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
le           = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))

# ── Classical SVM (Fair) — same 6 features / 200 samples as quantum models ───
fair_svm          = joblib.load(os.path.join(MODELS_DIR, 'fair_svm.pkl'))
fair_svm_scaler   = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_scaler.pkl'))
fair_svm_features = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_features.pkl'))

with open(os.path.join(MODELS_DIR, 'results.json')) as f:
    results = json.load(f)

# ── QSVM ─────────────────────────────────────────────────────────────────────
QSVM_OK = False
qsvm_model = qsvm_scaler = qsvm_features = qsvm_X_train = None

try:
    qsvm_model    = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
    qsvm_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qsvm_scaler.pkl'))
    qsvm_features = joblib.load(os.path.join(MODELS_DIR, 'qsvm_features.pkl'))
    qsvm_X_train  = np.load(os.path.join(MODELS_DIR, 'qsvm_X_train.npy'))

    _NQ = len(qsvm_features)
    _dev_qsvm = qml.device("default.qubit", wires=_NQ)

    def _qsvm_fmap(x):
        qml.AngleEmbedding(x, wires=range(_NQ), rotation='X')
        for i in range(_NQ - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.AngleEmbedding(x, wires=range(_NQ), rotation='Z')
        for i in range(_NQ - 2, -1, -1):
            qml.CNOT(wires=[i + 1, i])

    @qml.qnode(_dev_qsvm)
    def _qsvm_kernel_circuit(x1, x2):
        _qsvm_fmap(x1)
        qml.adjoint(_qsvm_fmap)(x2)
        return qml.probs(wires=range(_NQ))

    def _qk(x1, x2):
        return float(_qsvm_kernel_circuit(
            np.array(x1, dtype=float), np.array(x2, dtype=float)
        )[0])

    QSVM_OK = True
    print(f"[QSVM] Loaded — features: {qsvm_features}")
except Exception as e:
    print(f"[QSVM] Not available: {e}")

# ── QNN ───────────────────────────────────────────────────────────────────────
QNN_OK = False
qnn_weights = qnn_scaler = qnn_features = None

try:
    qnn_weights  = np.load(os.path.join(MODELS_DIR, 'qnn_weights.npy'))
    qnn_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qnn_scaler.pkl'))
    qnn_features = joblib.load(os.path.join(MODELS_DIR, 'qnn_features.pkl'))

    _NQ_QNN = len(qnn_features)
    _dev_qnn = qml.device("default.qubit", wires=_NQ_QNN)

    @qml.qnode(_dev_qnn)
    def _qnn_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(_NQ_QNN), rotation='X')
        for layer in range(weights.shape[0]):
            for qubit in range(_NQ_QNN):
                qml.Rot(weights[layer, qubit, 0],
                        weights[layer, qubit, 1],
                        weights[layer, qubit, 2], wires=qubit)
            for qubit in range(_NQ_QNN):
                qml.CNOT(wires=[qubit, (qubit + 1) % _NQ_QNN])
        return qml.expval(qml.PauliZ(0))

    QNN_OK = True
    print(f"[QNN] Loaded — features: {qnn_features}")
except Exception as e:
    print(f"[QNN] Not available: {e}")


# Neutral defaults — training data means, so unknown features don't bias the model
# (defaulting to 0 makes every site look like a brand-new zero-traffic domain)
_NEUTRAL = {
    'web_traffic': 799374,   # training mean — far better than 0
    'domain_age':  3893,     # training mean (~10.7 years) — neutral when WHOIS fails
}

# ── Content feature fetcher (QSVM / QNN) ─────────────────────────────────────
def fetch_content_features(url, feature_list):
    """
    Fetches real-time page content and WHOIS data to populate quantum model features.
    Unknown features default to training-data means (not 0) to avoid biasing the model.
    Returns (feature_dict, estimated_set).
    """
    estimated = set()
    # Start with neutral (mean) defaults, not zero
    vals = {f: float(_NEUTRAL.get(f, 0.0)) for f in feature_list}

    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    domain = parsed.netloc.lstrip('www.') or url.split('/')[0]

    # ── Fetch page → hyperlink counts ─────────────────────────────────────
    page_html = None
    try:
        resp = requests.get(
            url, timeout=8,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
            allow_redirects=True,
        )
        page_html = resp.text
    except Exception:
        for f in ['nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks', 'google_index']:
            if f in vals:
                estimated.add(f)

    if page_html:
        soup  = BeautifulSoup(page_html, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]
        total = len(hrefs)
        int_l = sum(1 for h in hrefs if h.startswith('/') or domain in h)
        ext_l = total - int_l

        if 'nb_hyperlinks'        in vals: vals['nb_hyperlinks']        = total
        if 'ratio_intHyperlinks'  in vals: vals['ratio_intHyperlinks']  = int_l / total if total else 0
        if 'ratio_extHyperlinks'  in vals: vals['ratio_extHyperlinks']  = ext_l / total if total else 0
        if 'google_index'         in vals: vals['google_index']         = 1

    # ── WHOIS → domain age ────────────────────────────────────────────────
    if 'domain_age' in vals:
        if WHOIS_OK:
            try:
                from dateutil import parser as dateparser
                w = python_whois.whois(domain)
                created = w.creation_date
                if isinstance(created, list):
                    created = created[0]
                # Handle both datetime objects and date strings
                if created and not isinstance(created, datetime):
                    try:
                        created = dateparser.parse(str(created))
                    except Exception:
                        created = None
                if created and isinstance(created, datetime):
                    now = datetime.now(tz=timezone.utc) if created.tzinfo else datetime.now()
                    vals['domain_age'] = max(0, (now - created).days)
                else:
                    estimated.add('domain_age')  # stays at neutral default
            except Exception:
                estimated.add('domain_age')      # stays at neutral default
        else:
            estimated.add('domain_age')

    # ── Web traffic (no free public API — stays at neutral default) ───────
    if 'web_traffic' in vals:
        estimated.add('web_traffic')

    return vals, estimated


# ── Shared URL feature display ─────────────────────────────────────────────────
def build_key_features(features):
    return {
        'URL Length':         {'value': features['length_url'],         'suspicious': features['length_url'] > 75},
        'Hostname Length':    {'value': features['length_hostname'],    'suspicious': features['length_hostname'] > 30},
        'Uses HTTPS':         {'value': 'Yes' if features['https_token'] == 0 else 'No', 'suspicious': features['https_token'] != 0},
        'IP as Hostname':     {'value': 'Yes' if features['ip'] == 1 else 'No',          'suspicious': features['ip'] == 1},
        'Dots in URL':        {'value': features['nb_dots'],            'suspicious': features['nb_dots'] > 4},
        'Hyphens in URL':     {'value': features['nb_hyphens'],         'suspicious': features['nb_hyphens'] > 1},
        '@ Symbol':           {'value': features['nb_at'],              'suspicious': features['nb_at'] > 0},
        '% Encoding':         {'value': features['nb_percent'],         'suspicious': features['nb_percent'] > 2},
        'Subdomains':         {'value': features['nb_subdomains'],      'suspicious': features['nb_subdomains'] > 2},
        'Suspicious TLD':     {'value': 'Yes' if features['suspecious_tld'] == 1 else 'No',    'suspicious': features['suspecious_tld'] == 1},
        'Phishing Keywords':  {'value': features['phish_hints'],        'suspicious': features['phish_hints'] > 0},
        'Brand in Subdomain': {'value': 'Yes' if features['brand_in_subdomain'] == 1 else 'No','suspicious': features['brand_in_subdomain'] == 1},
        'URL Shortener':      {'value': 'Yes' if features['shortening_service'] == 1 else 'No','suspicious': features['shortening_service'] == 1},
        'DNS Record':         {'value': 'Found' if features['dns_record'] == 1 else 'Missing', 'suspicious': features['dns_record'] == 0},
        'Digit Ratio':        {'value': round(features['ratio_digits_url'], 3),                'suspicious': features['ratio_digits_url'] > 0.2},
        'Random Domain':      {'value': 'Yes' if features['random_domain'] == 1 else 'No',    'suspicious': features['random_domain'] == 1},
        'Punycode':           {'value': 'Yes' if features['punycode'] == 1 else 'No',          'suspicious': features['punycode'] == 1},
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', results=results, qsvm_ok=QSVM_OK, qnn_ok=QNN_OK)


@app.route('/analyse', methods=['POST'])
def analyse():
    url   = request.form.get('url', '').strip()
    model = request.form.get('model', 'classical')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    features, base_estimated = extract_features(url)
    kf = build_key_features(features)

    # ── QSVM ──────────────────────────────────────────────────────────────
    if model == 'qsvm' and QSVM_OK:
        content_vals, content_estimated = fetch_content_features(url, qsvm_features)

        x_raw    = np.array([[content_vals[f] for f in qsvm_features]])
        x_scaled = qsvm_scaler.transform(x_raw)[0]

        kernel_row = np.array([_qk(x_scaled, xt) for xt in qsvm_X_train])
        prediction = qsvm_model.predict(kernel_row.reshape(1, -1))[0]
        decision   = float(qsvm_model.decision_function(kernel_row.reshape(1, -1))[0])

        label      = 'phishing' if prediction == 1 else 'legitimate'
        confidence = round(min(abs(decision) / 2 * 100, 99), 1)

        quantum_features = {
            'Hyperlinks (total)':  {'value': int(content_vals['nb_hyperlinks']),              'estimated': 'nb_hyperlinks' in content_estimated},
            'Internal link ratio': {'value': round(content_vals['ratio_intHyperlinks'], 3),   'estimated': 'ratio_intHyperlinks' in content_estimated},
            'External link ratio': {'value': round(content_vals['ratio_extHyperlinks'], 3),   'estimated': 'ratio_extHyperlinks' in content_estimated},
            'Domain age (days)':   {'value': int(content_vals['domain_age']),                 'estimated': 'domain_age' in content_estimated},
            'Google indexed':      {'value': 'Yes' if content_vals['google_index'] else 'No', 'estimated': 'google_index' in content_estimated},
            'Web traffic rank':    {'value': int(content_vals['web_traffic']),                'estimated': 'web_traffic' in content_estimated},
        }

        return jsonify({
            'label':             label,
            'confidence':        confidence,
            'model_used':        'QSVM',
            'model_detail':      f'Quantum Kernel · {_NQ} qubits · {len(qsvm_X_train)} training samples',
            'model_accuracy':    '86.00%',
            'key_features':      kf,
            'quantum_features':  quantum_features,
            'estimated_count':   len(content_estimated),
            'estimated_list':    sorted(content_estimated),
        })

    # ── QNN ───────────────────────────────────────────────────────────────
    elif model == 'qnn' and QNN_OK:
        content_vals, content_estimated = fetch_content_features(url, qnn_features)

        x_raw    = np.array([[content_vals[f] for f in qnn_features]])
        x_scaled = np.clip(qnn_scaler.transform(x_raw)[0], 0, np.pi)
        raw_pred = float(_qnn_circuit(x_scaled, qnn_weights))

        label      = 'phishing' if raw_pred >= 0 else 'legitimate'
        confidence = round(min(abs(raw_pred) * 100, 99), 1)

        qnn_acc = results.get('qnn', {}).get('accuracy', 0)

        return jsonify({
            'label':           label,
            'confidence':      confidence,
            'model_used':      'QNN',
            'model_detail':    f'Variational · {_NQ_QNN} qubits · 3 layers · Adam trained',
            'model_accuracy':  f'{qnn_acc * 100:.2f}%',
            'key_features':    kf,
            'estimated_count': len(content_estimated),
            'estimated_list':  sorted(content_estimated),
        })

    # ── Classical SVM (Fair) — default tab ───────────────────────────────
    else:
        content_vals, content_estimated = fetch_content_features(url, fair_svm_features)

        x_raw    = np.array([[content_vals[f] for f in fair_svm_features]])
        x_scaled = fair_svm_scaler.transform(x_raw)
        pred_enc = fair_svm.predict(x_scaled)[0]
        label    = 'phishing' if pred_enc == 1 else 'legitimate'
        score    = fair_svm.decision_function(x_scaled)[0]
        confidence = round(min(abs(score) / 3 * 100, 99), 1)

        quantum_features = {
            'Hyperlinks (total)':  {'value': int(content_vals['nb_hyperlinks']),              'estimated': 'nb_hyperlinks' in content_estimated},
            'Internal link ratio': {'value': round(content_vals['ratio_intHyperlinks'], 3),   'estimated': 'ratio_intHyperlinks' in content_estimated},
            'External link ratio': {'value': round(content_vals['ratio_extHyperlinks'], 3),   'estimated': 'ratio_extHyperlinks' in content_estimated},
            'Domain age (days)':   {'value': int(content_vals['domain_age']),                 'estimated': 'domain_age' in content_estimated},
            'Google indexed':      {'value': 'Yes' if content_vals['google_index'] else 'No', 'estimated': 'google_index' in content_estimated},
            'Web traffic rank':    {'value': int(content_vals['web_traffic']),                'estimated': 'web_traffic' in content_estimated},
        }

        return jsonify({
            'label':            label,
            'confidence':       confidence,
            'model_used':       'Classical SVM (Fair)',
            'model_detail':     '6 features · 200 training samples · RBF kernel',
            'model_accuracy':   '86.00%',
            'key_features':     kf,
            'quantum_features': quantum_features,
            'estimated_count':  len(content_estimated),
            'estimated_list':   sorted(content_estimated),
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
