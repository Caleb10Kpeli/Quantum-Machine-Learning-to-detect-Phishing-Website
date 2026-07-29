const form          = document.getElementById('analyse-form');
const urlInput      = document.getElementById('url-input');
const btn           = document.getElementById('analyse-btn');
const panel         = document.getElementById('result-panel');
const banner        = document.getElementById('result-banner');
const grid          = document.getElementById('features-grid');
const loading       = document.getElementById('loading');
const loadingText   = loading.querySelector('span');
const errorMsg      = document.getElementById('error-msg');
const estNote       = document.getElementById('estimated-note');
const estList       = document.getElementById('estimated-list');
const confWrap      = document.getElementById('confidence-wrap');
const confFill      = document.getElementById('confidence-fill');
const confLabel     = document.getElementById('confidence-label-pct');
const modelBadgeRow = document.getElementById('model-badge-row');
const modelBadgeDot = document.getElementById('model-badge-dot');
const modelBadgeTxt = document.getElementById('model-badge-text');
const quantumPanel  = document.getElementById('quantum-panel');
const quantumGrid   = document.getElementById('quantum-grid');
const selectedModel = document.getElementById('selected-model');
const qsvmNote      = document.getElementById('qsvm-note');
const confFlag      = document.getElementById('confidence-flag');

const checkHealthBtn  = document.getElementById('check-health-btn');
const compareAllBtn   = document.getElementById('compare-all-btn');
const healthPanel     = document.getElementById('health-panel');
const healthVerdict   = document.getElementById('health-verdict');
const healthDetails   = document.getElementById('health-details');
const healthWarnings  = document.getElementById('health-warnings');
const consensusPanel  = document.getElementById('consensus-panel');
const consensusSummary = document.getElementById('consensus-summary');
const consensusTable  = document.getElementById('consensus-table');
const siteWarning     = document.getElementById('site-warning');
const consensusSiteWarning = document.getElementById('consensus-site-warning');

const LOW_CONFIDENCE_THRESHOLD = 25; // below this, flag as borderline

const MODEL_LOADING = {
  classical: 'Fetching page · running Classical SVM (Fair)… (~10s)',
  qsvm:      'Fetching page · running quantum kernel… (~10s)',
  qnn:       'Fetching page · running variational circuit… (~10s)',
  qsvm_v2:   'Fetching page · running quantum kernel (data re-uploading)… (~15s)',
  qnn_v2:    'Fetching page · running variational circuit (data re-uploading)… (~10s)',
};

const MODEL_DOT_CLASS = {
  'Classical SVM': 'dot-classical',
  'QSVM':          'dot-qsvm',
  'QNN':           'dot-qnn',
  'QSVM v2':       'dot-qsvm-v2',
  'QNN v2':        'dot-qnn-v2',
};

// ── Bar chart animation ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    document.querySelectorAll('.bar-fill[data-target]').forEach(el => {
      el.style.width = parseFloat(el.dataset.target) + '%';
    });
  }, 120);

  // Sample URL chips
  document.querySelectorAll('.chip[data-url]').forEach(chip => {
    chip.addEventListener('click', () => {
      urlInput.value = chip.dataset.url;
      urlInput.focus();
    });
  });
});

// ── Model tab switching ───────────────────────────────────────────────────────
document.querySelectorAll('.model-tab:not(.tab-disabled)').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    selectedModel.value = tab.dataset.model;

    // Hide previous result when switching models
    panel.classList.add('hidden');
    errorMsg.classList.add('hidden');
    healthPanel.classList.add('hidden');
    consensusPanel.classList.add('hidden');
  });
});

// ── Form submit ───────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const url   = urlInput.value.trim();
  const model = selectedModel.value;
  if (!url) return;

  panel.classList.add('hidden');
  errorMsg.classList.add('hidden');
  healthPanel.classList.add('hidden');
  consensusPanel.classList.add('hidden');
  loadingText.textContent = MODEL_LOADING[model] || MODEL_LOADING.classical;
  loading.classList.remove('hidden');
  btn.disabled = true;

  try {
    const fd = new FormData(form);

    const res  = await fetch('/analyse', { method: 'POST', body: fd });
    const data = await res.json();

    loading.classList.add('hidden');
    btn.disabled = false;

    if (!res.ok) {
      showError(data.error || 'Something went wrong.');
      return;
    }

    renderResult(data);
  } catch (err) {
    loading.classList.add('hidden');
    btn.disabled = false;
    showError('Network error — is the Flask server running?');
  }
});

// ── Site status warning (dead/unreachable URL detected during analysis) ────────
function applySiteStatus(el, siteStatus) {
  if (siteStatus && siteStatus.verdict !== 'live') {
    const icon = siteStatus.verdict === 'dead' ? '✕' : '⚠';
    el.textContent = `${icon} ${siteStatus.message}`;
    el.className = 'site-warning site-warning-' + siteStatus.verdict;
  } else {
    el.textContent = '';
    el.className = 'site-warning hidden';
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  applySiteStatus(siteWarning, data.site_status);
  const isPhishing = data.label === 'phishing';

  // Banner
  banner.className   = isPhishing ? 'banner-phishing' : 'banner-legitimate';
  banner.textContent = isPhishing
    ? `⚠ PHISHING DETECTED  (confidence ≈${data.confidence}%)`
    : `✓ LEGITIMATE WEBSITE  (confidence ≈${data.confidence}%)`;

  // Model badge
  modelBadgeDot.className = 'model-badge-dot';
  modelBadgeDot.classList.add(MODEL_DOT_CLASS[data.model_used] || 'dot-classical');
  modelBadgeTxt.textContent = `${data.model_used}  ·  ${data.model_detail}  ·  ${data.model_accuracy} accuracy`;
  modelBadgeRow.classList.remove('hidden');

  // Confidence bar
  confFill.className = 'confidence-fill ' + (isPhishing ? 'conf-phishing' : 'conf-legit');
  confFill.style.width = '0%';
  confLabel.textContent = data.confidence + '%';
  confWrap.classList.remove('hidden');
  setTimeout(() => { confFill.style.width = data.confidence + '%'; }, 60);
  confFlag.classList.toggle('hidden', data.confidence >= LOW_CONFIDENCE_THRESHOLD);

  // URL feature breakdown
  grid.innerHTML = '';
  for (const [name, info] of Object.entries(data.key_features)) {
    const row = document.createElement('div');
    row.className = 'feature-row';
    row.innerHTML = `
      <div class="feature-dot ${info.suspicious ? 'dot-bad' : 'dot-ok'}"></div>
      <span class="feature-name">${name}</span>
      <span class="feature-value ${info.suspicious ? 'suspicious' : ''}">${info.value}</span>
    `;
    grid.appendChild(row);
  }

  // Quantum features panel (QSVM only)
  if (data.quantum_features) {
    quantumGrid.innerHTML = '';
    for (const [name, info] of Object.entries(data.quantum_features)) {
      const row = document.createElement('div');
      row.className = 'quantum-row' + (info.estimated ? ' estimated' : '');
      row.innerHTML = `
        <span class="quantum-name">${name}</span>
        <span class="quantum-value">${info.value}</span>
        ${info.estimated ? '<span class="quantum-est-tag">estimated</span>' : ''}
      `;
      quantumGrid.appendChild(row);
    }
    quantumPanel.classList.remove('hidden');
  } else {
    quantumPanel.classList.add('hidden');
  }

  // Estimated note
  const sum = estNote.querySelector('summary');
  sum.textContent = `${data.estimated_count} features used default values`;
  estList.textContent = data.estimated_list.join(', ');

  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove('hidden');
}

// ── Site Health check ─────────────────────────────────────────────────────────
checkHealthBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) { showError('Enter a URL first.'); return; }

  errorMsg.classList.add('hidden');
  panel.classList.add('hidden');
  consensusPanel.classList.add('hidden');
  healthPanel.classList.add('hidden');
  loadingText.textContent = 'Checking DNS, HTTP response and SSL certificate…';
  loading.classList.remove('hidden');
  checkHealthBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append('url', url);
    const res  = await fetch('/site-health', { method: 'POST', body: fd });
    const data = await res.json();
    loading.classList.add('hidden');
    checkHealthBtn.disabled = false;

    if (!res.ok) { showError(data.error || 'Something went wrong.'); return; }
    renderHealth(data);
  } catch (err) {
    loading.classList.add('hidden');
    checkHealthBtn.disabled = false;
    showError('Network error — is the Flask server running?');
  }
});

function renderHealth(data) {
  const verdictMap = {
    live:        { cls: 'verdict-live',        text: '✓ LIVE — site responded' },
    dead:        { cls: 'verdict-dead',        text: '✕ DEAD — domain does not resolve' },
    unreachable: { cls: 'verdict-unreachable', text: '⚠ UNREACHABLE — DNS resolves but server did not respond' },
  };
  const v = verdictMap[data.verdict] || verdictMap.unreachable;
  healthVerdict.className   = 'health-verdict ' + v.cls;
  healthVerdict.textContent = v.text;

  const rows = [];
  if (data.resolved_ip)      rows.push(['Resolved IP', data.resolved_ip]);
  if (data.http_status)      rows.push(['HTTP status', data.http_status]);
  if (data.response_time_ms != null) rows.push(['Response time', data.response_time_ms + ' ms']);
  if (data.redirect_chain && data.redirect_chain.length > 1) {
    rows.push(['Redirect chain', data.redirect_chain.map(h => `${h.status} → ${h.url}`).join('<br>')]);
  }
  if (data.ssl && data.ssl.available) {
    rows.push(['SSL issuer', data.ssl.issuer]);
    rows.push(['SSL issued', `${data.ssl.issued_days_ago} day(s) ago (valid until ${data.ssl.valid_until})`]);
    rows.push(['SSL verified', data.ssl.verified ? 'Yes' : 'No — untrusted/self-signed']);
  }
  if (data.message) rows.push(['Note', data.message]);

  healthDetails.innerHTML = rows.map(([label, val]) => `
    <div class="health-row">
      <span class="health-label">${label}</span>
      <span class="health-value">${val}</span>
    </div>
  `).join('');

  if (data.warnings && data.warnings.length) {
    healthWarnings.innerHTML = data.warnings.map(w => `<div class="health-warning">&#9888; ${w}</div>`).join('');
  } else {
    healthWarnings.innerHTML = '';
  }

  healthPanel.classList.remove('hidden');
  healthPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Model Consensus (all 5 models on one fetch) ─────────────────────────────────
compareAllBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) { showError('Enter a URL first.'); return; }

  errorMsg.classList.add('hidden');
  panel.classList.add('hidden');
  healthPanel.classList.add('hidden');
  consensusPanel.classList.add('hidden');
  loadingText.textContent = 'Fetching page once, running all 5 models…';
  loading.classList.remove('hidden');
  compareAllBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append('url', url);
    const res  = await fetch('/analyse-all', { method: 'POST', body: fd });
    const data = await res.json();
    loading.classList.add('hidden');
    compareAllBtn.disabled = false;

    if (!res.ok) { showError(data.error || 'Something went wrong.'); return; }
    renderConsensus(data);
  } catch (err) {
    loading.classList.add('hidden');
    compareAllBtn.disabled = false;
    showError('Network error — is the Flask server running?');
  }
});

function renderConsensus(data) {
  applySiteStatus(consensusSiteWarning, data.site_status);
  const total = data.models.length;
  const agree = total - data.dissenters.length;
  consensusSummary.textContent = data.unanimous
    ? `${total}/${total} models agree: ${data.majority_label.toUpperCase()}`
    : `${agree}/${total} agree: ${data.majority_label.toUpperCase()} — ${data.dissenters.join(', ')} disagree`;
  consensusSummary.className = 'consensus-summary ' + (data.unanimous ? 'consensus-agree' : 'consensus-split');

  consensusTable.innerHTML = data.models.map(m => {
    const isDissenter = data.dissenters.includes(m.model_used);
    const isPhishing  = m.label === 'phishing';
    const lowConf     = m.confidence < LOW_CONFIDENCE_THRESHOLD;
    const dotClass    = MODEL_DOT_CLASS[m.model_used] || 'dot-classical';
    return `
      <div class="consensus-row ${isDissenter ? 'row-dissent' : ''}">
        <span class="model-badge-dot ${dotClass}"></span>
        <span class="consensus-model">${m.model_used}</span>
        <span class="consensus-label ${isPhishing ? 'label-phishing' : 'label-legit'}">
          ${isPhishing ? 'PHISHING' : 'LEGITIMATE'}
        </span>
        <span class="consensus-confidence">${m.confidence}%${lowConf ? ' <span class="low-conf-tag">borderline</span>' : ''}</span>
      </div>
    `;
  }).join('');

  consensusPanel.classList.remove('hidden');
  consensusPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
