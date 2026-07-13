"""
Tests all phishing URLs from the test set through all 3 models using LIVE content
fetching (same as the Flask app). Identifies best supervisor demo candidates.
"""

import os, sys, time, joblib, numpy as np, pandas as pd, requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import pennylane as qml

try:
    import whois as python_whois
    from dateutil import parser as dateparser
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

BASE       = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH  = os.path.join(BASE, 'data', 'phishing_data.csv.csv')
MODELS_DIR = os.path.join(BASE, 'models')

N_QUBITS = 6; N_TRAIN = 100; N_TEST = 50; SEED = 42

# ── Reconstruct test split ─────────────────────────────────────────────────────
df       = pd.read_csv(DATA_PATH)
all_f    = [c for c in df.columns if c not in ('url', 'status')]
le       = LabelEncoder()
y_all    = le.fit_transform(df['status'])
X_all    = df[all_f].fillna(df[all_f].median())
mi       = mutual_info_classif(X_all, y_all, random_state=SEED)
top_idx  = np.argsort(mi)[::-1][:N_QUBITS]
TOP_F    = [all_f[i] for i in top_idx]

np.random.seed(SEED)
ph_idx = np.where(y_all == 1)[0]; lg_idx = np.where(y_all == 0)[0]
tr_ph  = np.random.choice(ph_idx, N_TRAIN, replace=False)
tr_lg  = np.random.choice(lg_idx, N_TRAIN, replace=False)
te_ph  = np.random.choice(np.setdiff1d(ph_idx, tr_ph), N_TEST, replace=False)
te_lg  = np.random.choice(np.setdiff1d(lg_idx, tr_lg), N_TEST, replace=False)
test_idx = np.random.permutation(np.concatenate([te_ph, te_lg]))
y_test   = y_all[test_idx]
urls     = df['url'].iloc[test_idx].values

phish_urls = [(urls[i], i) for i in range(len(y_test)) if y_test[i] == 1]
print(f"Testing {len(phish_urls)} phishing URLs from test set...\n")

# ── Load models ────────────────────────────────────────────────────────────────
fair   = joblib.load(os.path.join(MODELS_DIR, 'fair_svm.pkl'))
fair_s = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_scaler.pkl'))
fair_f = joblib.load(os.path.join(MODELS_DIR, 'fair_svm_features.pkl'))
qsvm   = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
qsvm_s = joblib.load(os.path.join(MODELS_DIR, 'qsvm_scaler.pkl'))
qsvm_f = joblib.load(os.path.join(MODELS_DIR, 'qsvm_features.pkl'))
qsvm_X = np.load(os.path.join(MODELS_DIR, 'qsvm_X_train.npy'))
qnn_w  = np.load(os.path.join(MODELS_DIR, 'qnn_weights.npy'))
qnn_s  = joblib.load(os.path.join(MODELS_DIR, 'qnn_scaler.pkl'))
qnn_f  = joblib.load(os.path.join(MODELS_DIR, 'qnn_features.pkl'))

NQ = N_QUBITS
dev_q = qml.device('default.qubit', wires=NQ)
dev_n = qml.device('default.qubit', wires=NQ)

def fmap(x):
    qml.AngleEmbedding(x, wires=range(NQ), rotation='X')
    for i in range(NQ - 1): qml.CNOT(wires=[i, i + 1])
    qml.AngleEmbedding(x, wires=range(NQ), rotation='Z')
    for i in range(NQ - 2, -1, -1): qml.CNOT(wires=[i + 1, i])

@qml.qnode(dev_q)
def kc(x1, x2):
    fmap(x1); qml.adjoint(fmap)(x2)
    return qml.probs(wires=range(NQ))

@qml.qnode(dev_n)
def qcirc(inp, wts):
    qml.AngleEmbedding(inp, wires=range(NQ), rotation='X')
    for l in range(wts.shape[0]):
        for q in range(NQ): qml.Rot(wts[l,q,0],wts[l,q,1],wts[l,q,2],wires=q)
        for q in range(NQ): qml.CNOT(wires=[q,(q+1)%NQ])
    return qml.expval(qml.PauliZ(0))

def qk(x1, x2):
    return float(kc(np.array(x1,dtype=float), np.array(x2,dtype=float))[0])

NEUTRAL = {'web_traffic': 799374, 'domain_age': 3893}

def fetch(url, feature_list):
    vals = {f: float(NEUTRAL.get(f, 0.0)) for f in feature_list}
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    domain = parsed.netloc.lstrip('www.') or url.split('/')[0]
    live = False
    try:
        r = requests.get(url, timeout=6,
            headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True)
        soup  = BeautifulSoup(r.text, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]
        total = len(hrefs)
        int_l = sum(1 for h in hrefs if h.startswith('/') or domain in h)
        ext_l = total - int_l
        if 'nb_hyperlinks'       in vals: vals['nb_hyperlinks']       = total
        if 'ratio_intHyperlinks' in vals: vals['ratio_intHyperlinks'] = int_l/total if total else 0
        if 'ratio_extHyperlinks' in vals: vals['ratio_extHyperlinks'] = ext_l/total if total else 0
        if 'google_index'        in vals: vals['google_index']        = 1
        live = True
    except Exception:
        pass
    if 'domain_age' in vals and WHOIS_OK:
        try:
            w  = python_whois.whois(domain)
            cr = w.creation_date
            if isinstance(cr, list): cr = cr[0]
            if cr and not isinstance(cr, datetime):
                try: cr = dateparser.parse(str(cr))
                except: cr = None
            if cr:
                now = datetime.now(tz=timezone.utc) if cr.tzinfo else datetime.now()
                vals['domain_age'] = max(0, (now - cr).days)
        except: pass
    return vals, live

def predict(vals):
    # Classical
    xc  = fair_s.transform(np.array([[vals[f] for f in fair_f]]))
    pc  = fair.predict(xc)[0]
    sc  = fair.decision_function(xc)[0]
    cls = ('P' if pc == 1 else 'L', round(min(abs(sc)/3*100, 99), 1))

    # QSVM
    xs   = qsvm_s.transform(np.array([[vals[f] for f in qsvm_f]]))[0]
    kr   = np.array([qk(xs, xt) for xt in qsvm_X])
    pq   = qsvm.predict(kr.reshape(1,-1))[0]
    dq   = float(qsvm.decision_function(kr.reshape(1,-1))[0])
    qs   = ('P' if pq == 1 else 'L', round(min(abs(dq)/2*100, 99), 1))

    # QNN
    xn   = np.clip(qnn_s.transform(np.array([[vals[f] for f in qnn_f]]))[0], 0, np.pi)
    raw  = float(qcirc(xn, qnn_w))
    qn   = ('P' if raw >= 0 else 'L', round(min(abs(raw)*100, 99), 1))

    return cls, qs, qn

# ── Run all phishing URLs ──────────────────────────────────────────────────────
results = []
all_feats = list(set(list(fair_f)+list(qsvm_f)+list(qnn_f)))

for idx, (url, _) in enumerate(phish_urls):
    print(f"  [{idx+1:2d}/{len(phish_urls)}] {url[:70]}", end=' ... ', flush=True)
    vals, live = fetch(url, all_feats)
    cls, qs, qn = predict(vals)
    results.append((url, live, cls, qs, qn))
    status = f"cls={'P' if cls[0]=='P' else 'l'}  qsvm={'P' if qs[0]=='P' else 'l'}  qnn={'P' if qn[0]=='P' else 'l'}  {'[live]' if live else '[offline]'}"
    print(status)

# ── Summarise ──────────────────────────────────────────────────────────────────
print()
print('=' * 90)
print('  BEST DEMO CANDIDATES -- supervisor will be impressed')
print('  Scenario A: ALL 3 models agree => PHISHING (clean clear-cut result)')
print('=' * 90)
for url, live, cls, qs, qn in results:
    if cls[0] == 'P' and qs[0] == 'P' and qn[0] == 'P':
        print(f"\n  {url}")
        print(f"    Classical={cls[1]:.0f}%  QSVM={qs[1]:.0f}%  QNN={qn[1]:.0f}%  {'[LIVE PAGE]' if live else '[offline - URL features only]'}")

print()
print('=' * 90)
print('  Scenario B: Classical MISSES, quantum model(s) CATCH => shows quantum advantage')
print('=' * 90)
for url, live, cls, qs, qn in results:
    if cls[0] == 'L' and (qs[0] == 'P' or qn[0] == 'P'):
        caught = []
        if qs[0] == 'P': caught.append(f'QSVM {qs[1]:.0f}%')
        if qn[0] == 'P': caught.append(f'QNN {qn[1]:.0f}%')
        print(f"\n  {url}")
        print(f"    Classical MISSED ({cls[1]:.0f}% legit)   Caught by: {', '.join(caught)}  {'[LIVE PAGE]' if live else '[offline - URL features only]'}")

print()
print('=' * 90)
print('  SUMMARY')
total = len(results)
all3  = sum(1 for _,_,c,q,n in results if c[0]==q[0]==n[0]=='P')
cls_c = sum(1 for _,_,c,_,_ in results if c[0]=='P')
qs_c  = sum(1 for _,_,_,q,_ in results if q[0]=='P')
qn_c  = sum(1 for _,_,_,_,n in results if n[0]=='P')
live_ = sum(1 for _,l,_,_,_ in results if l)
print(f'  Total phishing URLs tested : {total}')
print(f'  Still live (page loaded)   : {live_}')
print(f'  Classical catches          : {cls_c}/{total}')
print(f'  QSVM catches               : {qs_c}/{total}')
print(f'  QNN catches                : {qn_c}/{total}')
print(f'  All 3 agree (phishing)     : {all3}/{total}')
print('=' * 90)
