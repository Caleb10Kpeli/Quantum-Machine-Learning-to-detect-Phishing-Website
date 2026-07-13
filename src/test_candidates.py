"""Quick test of candidate chip URLs across all 3 models."""
import joblib, numpy as np, requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pennylane as qml
try:
    import whois as w_lib
    from dateutil import parser as dateparser
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

BASE = "."
qnn_s  = joblib.load('models/qnn_scaler.pkl')
qnn_f  = joblib.load('models/qnn_features.pkl')
qnn_w  = np.load('models/qnn_weights.npy')
fair   = joblib.load('models/fair_svm.pkl')
fair_s = joblib.load('models/fair_svm_scaler.pkl')
fair_f = joblib.load('models/fair_svm_features.pkl')
qsvm   = joblib.load('models/qsvm.pkl')
qsvm_s = joblib.load('models/qsvm_scaler.pkl')
qsvm_f = joblib.load('models/qsvm_features.pkl')
qsvm_X = np.load('models/qsvm_X_train.npy')

NQ = len(qnn_f)
dqsvm = qml.device('default.qubit', wires=NQ)
dqnn  = qml.device('default.qubit', wires=NQ)
NEUTRAL = {'web_traffic': 799374, 'domain_age': 3893}

def fmap(x):
    qml.AngleEmbedding(x, wires=range(NQ), rotation='X')
    for i in range(NQ - 1): qml.CNOT(wires=[i, i + 1])
    qml.AngleEmbedding(x, wires=range(NQ), rotation='Z')
    for i in range(NQ - 2, -1, -1): qml.CNOT(wires=[i + 1, i])

@qml.qnode(dqsvm)
def kc(x1, x2):
    fmap(x1); qml.adjoint(fmap)(x2)
    return qml.probs(wires=range(NQ))

@qml.qnode(dqnn)
def qcirc(inp, wts):
    qml.AngleEmbedding(inp, wires=range(NQ), rotation='X')
    for l in range(wts.shape[0]):
        for q in range(NQ):
            qml.Rot(wts[l, q, 0], wts[l, q, 1], wts[l, q, 2], wires=q)
        for q in range(NQ):
            qml.CNOT(wires=[q, (q + 1) % NQ])
    return qml.expval(qml.PauliZ(0))

def qk(x1, x2):
    return float(kc(np.array(x1, dtype=float), np.array(x2, dtype=float))[0])

all_feats = list(set(list(fair_f) + list(qsvm_f) + list(qnn_f)))

CANDIDATES = [
    ('Reuters',    'https://www.reuters.com'),
    ('CNN',        'https://www.cnn.com'),
    ('Yahoo',      'https://www.yahoo.com'),
    ('Python.org', 'https://www.python.org'),
    ('NASA',       'https://www.nasa.gov'),
]

print()
print('=' * 85)
print('  CANDIDATE CHIP TEST')
print('=' * 85)

for name, url in CANDIDATES:
    parsed = urlparse(url)
    domain = parsed.netloc.lstrip('www.')
    vals = {f: float(NEUTRAL.get(f, 0.0)) for f in all_feats}

    try:
        r = requests.get(url, timeout=8,
                         headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True)
        soup  = BeautifulSoup(r.text, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]
        total = len(hrefs)
        int_l = sum(1 for h in hrefs if h.startswith('/') or domain in h)
        ext_l = total - int_l
        if 'nb_hyperlinks' in vals:       vals['nb_hyperlinks']       = total
        if 'ratio_intHyperlinks' in vals: vals['ratio_intHyperlinks'] = int_l / total if total else 0
        if 'ratio_extHyperlinks' in vals: vals['ratio_extHyperlinks'] = ext_l / total if total else 0
        if 'google_index' in vals:        vals['google_index']        = 1
        link_note = f'{total} links ({int_l} int / {ext_l} ext)'
    except Exception as e:
        link_note = f'fetch failed: {e}'

    if WHOIS_OK:
        try:
            wi = w_lib.whois(domain)
            cr = wi.creation_date
            if isinstance(cr, list): cr = cr[0]
            if cr and not isinstance(cr, datetime):
                try: cr = dateparser.parse(str(cr))
                except: cr = None
            if cr:
                now = datetime.now(tz=timezone.utc) if cr.tzinfo else datetime.now()
                vals['domain_age'] = max(0, (now - cr).days)
        except: pass

    # Classical
    xc  = fair_s.transform(np.array([[vals[f] for f in fair_f]]))
    pc  = fair.predict(xc)[0]
    cls = 'LEGIT' if pc == 0 else 'PHISH'

    # QSVM
    xs    = qsvm_s.transform(np.array([[vals[f] for f in qsvm_f]]))[0]
    kr    = np.array([qk(xs, xt) for xt in qsvm_X])
    pq    = qsvm.predict(kr.reshape(1, -1))[0]
    qs_l  = 'LEGIT' if pq == 0 else 'PHISH'

    # QNN
    xn    = np.clip(qnn_s.transform(np.array([[vals[f] for f in qnn_f]]))[0], 0, np.pi)
    raw   = float(qcirc(xn, qnn_w))
    qn_l  = 'PHISH' if raw >= 0 else 'LEGIT'

    agree = 'ALL LEGIT OK' if cls == qs_l == qn_l == 'LEGIT' else '*** PROBLEM ***'
    age   = int(vals['domain_age'])
    print(f'\n  {name} ({url})')
    print(f'    {link_note}  |  domain_age={age}d')
    print(f'    Classical={cls}  QSVM={qs_l}  QNN={qn_l} (raw={raw:+.3f})  -->  {agree}')

print()
print('=' * 85)
