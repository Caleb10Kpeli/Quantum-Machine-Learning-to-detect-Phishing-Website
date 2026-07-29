import os
import sys
import json
import time
import ssl
import socket
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

# ── QSVM v2 (Data Re-uploading) ─────────────────────────────────────────────────
QSVM_V2_OK = False
qsvm_v2_model = qsvm_v2_scaler = qsvm_v2_features = qsvm_v2_X_train = None

try:
    qsvm_v2_model    = joblib.load(os.path.join(MODELS_DIR, 'qsvm_reupload.pkl'))
    qsvm_v2_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qsvm_reupload_scaler.pkl'))
    qsvm_v2_features = joblib.load(os.path.join(MODELS_DIR, 'qsvm_reupload_features.pkl'))
    qsvm_v2_X_train  = np.load(os.path.join(MODELS_DIR, 'qsvm_reupload_X_train.npy'))

    _NQ_V2   = len(qsvm_v2_features)
    _N_REPS  = 2
    _dev_qsvm_v2 = qml.device("default.qubit", wires=_NQ_V2)

    def _qsvm_v2_fmap(x):
        for _ in range(_N_REPS):
            qml.AngleEmbedding(x, wires=range(_NQ_V2), rotation='X')
            for i in range(_NQ_V2 - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.AngleEmbedding(x, wires=range(_NQ_V2), rotation='Z')
            for i in range(_NQ_V2 - 2, -1, -1):
                qml.CNOT(wires=[i + 1, i])

    @qml.qnode(_dev_qsvm_v2)
    def _qsvm_v2_kernel_circuit(x1, x2):
        _qsvm_v2_fmap(x1)
        qml.adjoint(_qsvm_v2_fmap)(x2)
        return qml.probs(wires=range(_NQ_V2))

    def _qk_v2(x1, x2):
        return float(_qsvm_v2_kernel_circuit(
            np.array(x1, dtype=float), np.array(x2, dtype=float)
        )[0])

    QSVM_V2_OK = True
    print(f"[QSVM v2] Loaded — features: {qsvm_v2_features}")
except Exception as e:
    print(f"[QSVM v2] Not available: {e}")

# ── QNN v2 (Data Re-uploading) ───────────────────────────────────────────────────
QNN_V2_OK = False
qnn_v2_weights = qnn_v2_scaler = qnn_v2_features = None

try:
    qnn_v2_weights  = np.load(os.path.join(MODELS_DIR, 'qnn_reupload_weights.npy'))
    qnn_v2_scaler   = joblib.load(os.path.join(MODELS_DIR, 'qnn_reupload_scaler.pkl'))
    qnn_v2_features = joblib.load(os.path.join(MODELS_DIR, 'qnn_reupload_features.pkl'))

    _NQ_QNN_V2 = len(qnn_v2_features)
    _dev_qnn_v2 = qml.device("default.qubit", wires=_NQ_QNN_V2)

    @qml.qnode(_dev_qnn_v2)
    def _qnn_v2_circuit(inputs, weights):
        for layer in range(weights.shape[0]):
            qml.AngleEmbedding(inputs, wires=range(_NQ_QNN_V2), rotation='X')
            for qubit in range(_NQ_QNN_V2):
                qml.Rot(weights[layer, qubit, 0],
                        weights[layer, qubit, 1],
                        weights[layer, qubit, 2], wires=qubit)
            for qubit in range(_NQ_QNN_V2):
                qml.CNOT(wires=[qubit, (qubit + 1) % _NQ_QNN_V2])
        return qml.expval(qml.PauliZ(0))

    QNN_V2_OK = True
    print(f"[QNN v2] Loaded — features: {qnn_v2_features}")
except Exception as e:
    print(f"[QNN v2] Not available: {e}")


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
        'Uses HTTPS':         {'value': 'Yes' if features['https_token'] == 1 else 'No', 'suspicious': features['https_token'] == 0},
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


def build_quantum_features(content_vals, content_estimated):
    return {
        'Hyperlinks (total)':  {'value': int(content_vals['nb_hyperlinks']),              'estimated': 'nb_hyperlinks' in content_estimated},
        'Internal link ratio': {'value': round(content_vals['ratio_intHyperlinks'], 3),   'estimated': 'ratio_intHyperlinks' in content_estimated},
        'External link ratio': {'value': round(content_vals['ratio_extHyperlinks'], 3),   'estimated': 'ratio_extHyperlinks' in content_estimated},
        'Domain age (days)':   {'value': int(content_vals['domain_age']),                 'estimated': 'domain_age' in content_estimated},
        'Google indexed':      {'value': 'Yes' if content_vals['google_index'] else 'No', 'estimated': 'google_index' in content_estimated},
        'Web traffic rank':    {'value': int(content_vals['web_traffic']),                'estimated': 'web_traffic' in content_estimated},
    }


# ── Per-model prediction helpers ────────────────────────────────────────────────
# Each takes the already-fetched content feature dict and returns a prediction
# dict, or None if that model's artifacts failed to load. Shared by /analyse
# (one model at a time) and /analyse-all (all models on one fetch).

def predict_classical(content_vals):
    x_raw      = np.array([[content_vals[f] for f in fair_svm_features]])
    x_scaled   = fair_svm_scaler.transform(x_raw)
    pred_enc   = fair_svm.predict(x_scaled)[0]
    label      = 'phishing' if pred_enc == 1 else 'legitimate'
    score      = fair_svm.decision_function(x_scaled)[0]
    confidence = round(min(abs(score) / 3 * 100, 99), 1)
    return {
        'model_key':      'classical',
        'model_used':     'Classical SVM (Fair)',
        'model_detail':   '6 features · 200 training samples · RBF kernel',
        'model_accuracy': '86.00%',
        'label':          label,
        'confidence':     confidence,
    }


def predict_qsvm(content_vals):
    if not QSVM_OK:
        return None
    x_raw      = np.array([[content_vals[f] for f in qsvm_features]])
    x_scaled   = qsvm_scaler.transform(x_raw)[0]
    kernel_row = np.array([_qk(x_scaled, xt) for xt in qsvm_X_train])
    prediction = qsvm_model.predict(kernel_row.reshape(1, -1))[0]
    decision   = float(qsvm_model.decision_function(kernel_row.reshape(1, -1))[0])
    label      = 'phishing' if prediction == 1 else 'legitimate'
    confidence = round(min(abs(decision) / 2 * 100, 99), 1)
    return {
        'model_key':      'qsvm',
        'model_used':     'QSVM',
        'model_detail':   f'Quantum Kernel · {_NQ} qubits · {len(qsvm_X_train)} training samples',
        'model_accuracy': '86.00%',
        'label':          label,
        'confidence':     confidence,
    }


def predict_qnn(content_vals):
    if not QNN_OK:
        return None
    x_raw    = np.array([[content_vals[f] for f in qnn_features]])
    x_scaled = np.clip(qnn_scaler.transform(x_raw)[0], 0, np.pi)
    raw_pred = float(_qnn_circuit(x_scaled, qnn_weights))
    label      = 'phishing' if raw_pred >= 0 else 'legitimate'
    confidence = round(min(abs(raw_pred) * 100, 99), 1)
    qnn_acc = results.get('qnn', {}).get('accuracy', 0)
    return {
        'model_key':      'qnn',
        'model_used':     'QNN',
        'model_detail':   f'Variational · {_NQ_QNN} qubits · 3 layers · Adam trained',
        'model_accuracy': f'{qnn_acc * 100:.2f}%',
        'label':          label,
        'confidence':     confidence,
    }


def predict_qsvm_v2(content_vals):
    if not QSVM_V2_OK:
        return None
    x_raw      = np.array([[content_vals[f] for f in qsvm_v2_features]])
    x_scaled   = qsvm_v2_scaler.transform(x_raw)[0]
    kernel_row = np.array([_qk_v2(x_scaled, xt) for xt in qsvm_v2_X_train])
    prediction = qsvm_v2_model.predict(kernel_row.reshape(1, -1))[0]
    decision   = float(qsvm_v2_model.decision_function(kernel_row.reshape(1, -1))[0])
    label      = 'phishing' if prediction == 1 else 'legitimate'
    confidence = round(min(abs(decision) / 2 * 100, 99), 1)
    v2_acc = results.get('qsvm_reupload', {}).get('accuracy', 0)
    return {
        'model_key':      'qsvm_v2',
        'model_used':     'QSVM v2',
        'model_detail':   f'Quantum Kernel · {_NQ_V2} qubits · {_N_REPS}x data re-uploading',
        'model_accuracy': f'{v2_acc * 100:.2f}%',
        'label':          label,
        'confidence':     confidence,
    }


def predict_qnn_v2(content_vals):
    if not QNN_V2_OK:
        return None
    x_raw    = np.array([[content_vals[f] for f in qnn_v2_features]])
    x_scaled = np.clip(qnn_v2_scaler.transform(x_raw)[0], 0, np.pi)
    raw_pred = float(_qnn_v2_circuit(x_scaled, qnn_v2_weights))
    label      = 'phishing' if raw_pred >= 0 else 'legitimate'
    confidence = round(min(abs(raw_pred) * 100, 99), 1)
    v2_acc = results.get('qnn_reupload', {}).get('accuracy', 0)
    return {
        'model_key':      'qnn_v2',
        'model_used':     'QNN v2',
        'model_detail':   f'Variational · {_NQ_QNN_V2} qubits · 3 layers · re-upload/layer',
        'model_accuracy': f'{v2_acc * 100:.2f}%',
        'label':          label,
        'confidence':     confidence,
    }


PREDICTORS = {
    'classical': predict_classical,
    'qsvm':      predict_qsvm,
    'qnn':       predict_qnn,
    'qsvm_v2':   predict_qsvm_v2,
    'qnn_v2':    predict_qnn_v2,
}

# Feature list each model's live fetch should populate (identical set today,
# but kept per-model so a future model with different features stays correct).
_MODEL_FEATURE_LISTS = {
    'classical': fair_svm_features,
    'qsvm':      qsvm_features if QSVM_OK else fair_svm_features,
    'qnn':       qnn_features if QNN_OK else fair_svm_features,
    'qsvm_v2':   qsvm_v2_features if QSVM_V2_OK else fair_svm_features,
    'qnn_v2':    qnn_v2_features if QNN_V2_OK else fair_svm_features,
}

# Models whose result includes a "Quantum Model Inputs" panel in the UI
_MODELS_WITH_QUANTUM_PANEL = {'classical', 'qsvm', 'qsvm_v2'}


# ── Site health (liveness / SSL) helpers ────────────────────────────────────────
def get_ssl_info(hostname, timeout=6):
    """
    Connects to hostname:443 and reads the TLS certificate. Tries a verified
    handshake first; if verification fails, retries unverified purely to read
    certificate metadata (a failed/self-signed cert is itself a useful signal,
    not something to hide from the user).
    """
    verified = True
    cert = None
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
    except ssl.SSLCertVerificationError:
        verified = False
        try:
            ctx2 = ssl._create_unverified_context()
            with socket.create_connection((hostname, 443), timeout=timeout) as sock:
                with ctx2.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
        except Exception as e:
            return {'available': False, 'error': str(e)}
    except Exception as e:
        return {'available': False, 'error': str(e)}

    if not cert:
        return {'available': False, 'error': 'No certificate returned'}

    try:
        not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        not_after  = datetime.strptime(cert['notAfter'],  '%b %d %H:%M:%S %Y %Z')
        issuer_dict = dict(x[0] for x in cert.get('issuer', []))
        age_days = (datetime.utcnow() - not_before).days
        return {
            'available':       True,
            'verified':        verified,
            'issuer':          issuer_dict.get('organizationName') or issuer_dict.get('commonName', 'Unknown'),
            'issued_days_ago': age_days,
            'valid_until':     not_after.strftime('%Y-%m-%d'),
            'new_cert_flag':   age_days < 30,
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


def check_site_health(url):
    full_url = url if url.startswith('http') else 'http://' + url
    hostname = urlparse(full_url).hostname or full_url

    try:
        resolved_ip = socket.gethostbyname(hostname)
    except Exception:
        return {
            'reachable': False, 'verdict': 'dead', 'dns_ok': False,
            'message': 'DNS lookup failed — this domain does not resolve to anything.',
        }

    try:
        t0 = time.time()
        resp = requests.get(
            full_url, timeout=8, allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
        )
        elapsed_ms = round((time.time() - t0) * 1000)
    except requests.exceptions.RequestException as e:
        return {
            'reachable': False, 'verdict': 'unreachable', 'dns_ok': True,
            'resolved_ip': resolved_ip,
            'message': f'DNS resolves but the server did not respond ({e.__class__.__name__}).',
        }

    redirect_chain = [{'status': h.status_code, 'url': h.url} for h in resp.history]
    redirect_chain.append({'status': resp.status_code, 'url': resp.url})
    final_url = resp.url

    ssl_info = None
    if urlparse(final_url).scheme == 'https':
        ssl_info = get_ssl_info(urlparse(final_url).hostname or hostname)

    warnings = []
    if len(redirect_chain) > 2:
        warnings.append(f'{len(redirect_chain) - 1} redirect(s) before reaching the final page.')
    if ssl_info and ssl_info.get('available'):
        if ssl_info.get('new_cert_flag'):
            warnings.append(
                f"SSL certificate was issued only {ssl_info['issued_days_ago']} day(s) ago. "
                f"This can indicate freshly-stood-up phishing infrastructure, but large "
                f"CDN-backed sites also auto-rotate certificates this often — treat this "
                f"signal as informative only for smaller/unfamiliar domains, not on its own."
            )
        if not ssl_info.get('verified'):
            warnings.append('SSL certificate failed verification (untrusted or self-signed).')

    return {
        'reachable':        True,
        'verdict':          'live',
        'dns_ok':           True,
        'resolved_ip':      resolved_ip,
        'http_status':      resp.status_code,
        'response_time_ms': elapsed_ms,
        'redirect_chain':   redirect_chain,
        'final_url':        final_url,
        'ssl':              ssl_info,
        'warnings':         warnings,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template(
        'index.html', results=results,
        qsvm_ok=QSVM_OK, qnn_ok=QNN_OK,
        qsvm_v2_ok=QSVM_V2_OK, qnn_v2_ok=QNN_V2_OK,
    )


@app.route('/site-health', methods=['POST'])
def site_health():
    url = request.form.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    return jsonify(check_site_health(url))


@app.route('/analyse', methods=['POST'])
def analyse():
    url   = request.form.get('url', '').strip()
    model = request.form.get('model', 'classical')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if model not in PREDICTORS:
        model = 'classical'

    features, base_estimated = extract_features(url)
    kf = build_key_features(features)

    content_vals, content_estimated = fetch_content_features(url, _MODEL_FEATURE_LISTS[model])

    pred = PREDICTORS[model](content_vals)
    if pred is None:
        # Requested model's artifacts aren't loaded — fall back to Classical (Fair).
        model = 'classical'
        pred = predict_classical(content_vals)

    response = {
        'label':           pred['label'],
        'confidence':      pred['confidence'],
        'model_used':      pred['model_used'],
        'model_detail':    pred['model_detail'],
        'model_accuracy':  pred['model_accuracy'],
        'key_features':    kf,
        'estimated_count': len(content_estimated),
        'estimated_list':  sorted(content_estimated),
    }
    if model in _MODELS_WITH_QUANTUM_PANEL:
        response['quantum_features'] = build_quantum_features(content_vals, content_estimated)

    return jsonify(response)


@app.route('/analyse-all', methods=['POST'])
def analyse_all():
    url = request.form.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    features, base_estimated = extract_features(url)
    kf = build_key_features(features)

    # All 5 models currently share the same 6 MI-selected features, so one
    # live fetch (one page request, one WHOIS lookup) covers every model
    # instead of 5 separate ones.
    content_vals, content_estimated = fetch_content_features(url, fair_svm_features)

    model_results = []
    for key, fn in PREDICTORS.items():
        pred = fn(content_vals)
        if pred is not None:
            model_results.append(pred)

    phishing_votes = sum(1 for r in model_results if r['label'] == 'phishing')
    legit_votes    = len(model_results) - phishing_votes
    majority       = 'phishing' if phishing_votes > legit_votes else 'legitimate'
    dissenters     = [r['model_used'] for r in model_results if r['label'] != majority]

    return jsonify({
        'key_features':      kf,
        'quantum_features':  build_quantum_features(content_vals, content_estimated),
        'estimated_count':   len(content_estimated),
        'estimated_list':    sorted(content_estimated),
        'models':            model_results,
        'majority_label':    majority,
        'vote_counts':       {'phishing': phishing_votes, 'legitimate': legit_votes},
        'dissenters':        dissenters,
        'unanimous':         len(dissenters) == 0,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
