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

const MODEL_LOADING = {
  classical: 'Fetching page · running Classical SVM (Fair)… (~10s)',
  qsvm:      'Fetching page · running quantum kernel… (~10s)',
  qnn:       'Fetching page · running variational circuit… (~10s)',
};

const MODEL_DOT_CLASS = {
  'Classical SVM': 'dot-classical',
  'QSVM':          'dot-qsvm',
  'QNN':           'dot-qnn',
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

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
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
