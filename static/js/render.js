// ── STATE ─────────────────────────────────────────────────────────
// TICKERS is populated by market.js after /api/market/dashboard loads
let selectedTicker = null;
let filterSignal   = 'all';

// ── SIGNAL HELPER ────────────────────────────────────────────────
function getSignalDisplay(signal) {
  const sig = String(signal).toLowerCase().trim();
  if (sig === 'bullish') {
    return { text: '▲ BULLISH', class: 'bullish', arrow: '▲', color: 'pos' };
  } else if (sig === 'bearish') {
    return { text: '▼ BEARISH', class: 'bearish', arrow: '▼', color: 'neg' };
  } else {
    return { text: 'NEUTRAL', class: 'neutral', arrow: '●', color: 'neu' };
  }
}

// ── TICKER LIST ───────────────────────────────────────────────────
function renderTickerList() {
  const container = document.getElementById('tickerList');
  container.innerHTML = '';
  const list = filterSignal === 'all'
    ? TICKERS
    : TICKERS.filter(t => t.signal === filterSignal);
  
  list.forEach(t => {
    const pos = t.return_1d >= 0;
    const signalDisplay = getSignalDisplay(t.signal);
    const div = document.createElement('div');
    div.className = 'ticker-row' + (selectedTicker && t.ticker === selectedTicker.ticker ? ' selected' : '');
    div.innerHTML = `
      <div>
        <div class="ticker-sym">${t.ticker}</div>
        <div class="ticker-sector">${t.sector}</div>
      </div>
      <div class="ticker-right">
        <div class="ticker-price">$${t.close.toFixed(2)}</div>
        <div class="ticker-ret ${signalDisplay.color}-val">${signalDisplay.text}</div>
      </div>
    `;
    div.addEventListener('click', () => selectTicker(t));
    container.appendChild(div);
  });
}

function filterTickers(sig, btn) {
  filterSignal = sig;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTickerList();
}

// ── SELECT TICKER ─────────────────────────────────────────────────
function selectTicker(t) {
  selectedTicker = t;
  const signalDisplay = getSignalDisplay(t.signal);
  
  // Update chart header
  document.getElementById('chartSymbol').textContent = t.name || t.ticker;
  document.getElementById('chartSector').textContent = `${t.ticker} · ${t.sector}`;
  document.getElementById('chartPrice').textContent  = `$${t.close.toFixed(2)}`;
  
  const chg = document.getElementById('chartChange');
  chg.textContent = signalDisplay.text;
  chg.className   = 'chart-change ' + signalDisplay.color;
  
  const sig = document.getElementById('chartSignal');
  sig.textContent = signalDisplay.text;
  sig.className   = `signal-badge ${signalDisplay.class}`;
  
  document.getElementById('cRsi').textContent   = t.rsi_14.toFixed(2);
  document.getElementById('cMacd').textContent  = t.macd.toFixed(3);
  document.getElementById('cAtr').textContent   = t.atr_14.toFixed(2);
  document.getElementById('cStoch').textContent = t.stoch_k.toFixed(2);
  document.getElementById('cMom').textContent   = t.momentum_score.toFixed(3);
  
  // RSI color
  const rsiEl = document.getElementById('cRsi');
  rsiEl.className = t.rsi_14 > 70 ? 'cstat-v rsi-high' : t.rsi_14 < 30 ? 'cstat-v rsi-low' : 'cstat-v';
  
  renderTickerList();
  if (typeof loadTickerHistory === 'function') {
    loadTickerHistory(t.ticker);
  } else {
    drawPriceChart();
  }
  updateSentimentPanel();
}

function onChartRangeChange() {
  if (selectedTicker && typeof loadTickerHistory === 'function') {
    loadTickerHistory(selectedTicker.ticker);
  } else if (typeof drawPriceChart === 'function') {
    drawPriceChart();
  }
}

// ── SIGNAL SUMMARY ────────────────────────────────────────────────
function renderSignalSummary() {
  const n = TICKERS.length;
  const bull = TICKERS.filter(t => t.signal === 'bullish').length;
  const neu  = TICKERS.filter(t => t.signal === 'neutral').length;
  const bear = TICKERS.filter(t => t.signal === 'bearish').length;
}

// ── FINANCIAL MATRIX ──────────────────────────────────────────────
function renderMatrix() {
  const tbody = document.getElementById('matrixBody');
  tbody.innerHTML = '';
  TICKERS.forEach(t => {
    const rsiClass = t.rsi_14 > 70 ? 'rsi-high' : t.rsi_14 < 30 ? 'rsi-low' : 'rsi-mid';
    const retColor = t.exp_return >= 0 ? 'pos-val' : 'neg-val';
    const signalDisplay = getSignalDisplay(t.signal);
    const macdColor = t.macd >= 0 ? 'pos-val' : 'neg-val';
    const histColor = t.macd_hist >= 0 ? 'pos-val' : 'neg-val';
    
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${t.ticker}</td>
      <td>$${t.close.toFixed(2)}</td>
      <td class="${rsiClass}">${t.rsi_14.toFixed(2)}</td>
      <td class="${macdColor}">${t.macd.toFixed(3)}</td>
      <td class="${histColor}">${t.macd_hist.toFixed(3)}</td>
      <td>${(t.bb_percent * 100).toFixed(1)}%</td>
      <td>${t.atr_14.toFixed(2)}</td>
      <td>${t.stoch_k.toFixed(1)}</td>
      <td>${t.trend_strength.toFixed(3)}</td>
      <td>${t.momentum_score.toFixed(3)}</td>
      <td class="neg-val">${t.dist_52w_high.toFixed(2)}%</td>
      <td class="pos-val">+${t.dist_52w_low.toFixed(2)}%</td>
      <td class="${retColor}">${t.exp_return > 0 ? '+' : ''}${t.exp_return.toFixed(3)}%</td>
      <td>
        <span class="signal-badge-sm sig-${signalDisplay.class}">
          ${signalDisplay.text}
        </span>
      </td>
    `;
    tr.addEventListener('click', () => selectTicker(t));
    tbody.appendChild(tr);
  });
}

// ── SENTIMENT PANEL ───────────────────────────────────────────────
function populateSentimentSelect() {
  const sel = document.getElementById('sentimentTicker');
  if (!sel) return;
  TICKERS.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t.ticker;
    opt.textContent = `${t.ticker} — ${t.sector}`;
    sel.appendChild(opt);
  });
}

function updateSentimentPanel() {
  const tickerEl = document.getElementById('sentimentTicker');
  const body     = document.getElementById('sentimentBody');
  if (!tickerEl || !body) return;  // sentiment panel not in DOM — skip
  const ticker = tickerEl.value || selectedTicker.ticker;
  const s = SENTIMENT[ticker];
  if (!s) return;
  const scoreColor = s.score > 0.05 ? 'var(--green)' : s.score < -0.05 ? 'var(--red)' : 'var(--text-secondary)';
  const scoreLabel = s.score > 0.05 ? 'Positive' : s.score < -0.05 ? 'Negative' : 'Neutral';
  const scoreLabelClass = s.score > 0.05 ? 'sig-bullish' : s.score < -0.05 ? 'sig-bearish' : 'sig-neutral';
  const fillWidth = Math.min(Math.abs(s.score) * 100, 100);
  body.innerHTML = `
    <div class="sent-detail">
      <div class="sent-detail-title">📰 ${ticker} — Sentiment (Last 7 Days)</div>
      <!-- Score bar -->
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
        <div class="sent-score-bar" style="flex:1;height:8px">
          <div class="sent-fill" style="width:${fillWidth}%;background:${scoreColor}"></div>
        </div>
        <span class="sent-score-val" style="color:${scoreColor}">${s.score > 0 ? '+' : ''}${s.score.toFixed(3)}</span>
        <span class="sent-label ${scoreLabelClass}">${scoreLabel}</span>
      </div>

      <!-- Gauge -->
      <div class="sent-gauge">
        <div class="gauge-track">
          <div class="gauge-neg" style="width:${(s.neg*100).toFixed(1)}%"></div>
          <div class="gauge-neu" style="width:${(s.neu*100).toFixed(1)}%"></div>
          <div class="gauge-pos" style="width:${(s.pos*100).toFixed(1)}%"></div>
        </div>
        <div class="gauge-labels">
          <span>▼ Neg ${(s.neg*100).toFixed(0)}%</span>
          <span>Neu ${(s.neu*100).toFixed(0)}%</span>
          <span>Pos ${(s.pos*100).toFixed(0)}% ▲</span>
        </div>
      </div>

      <!-- Stats grid -->
      <div class="sent-stat-grid">
        <div class="sent-stat">
          <span class="sent-stat-l">Total Articles</span>
          <span class="sent-stat-v">${s.articles}</span>
        </div>
        <div class="sent-stat">
          <span class="sent-stat-l">Confidence</span>
          <span class="sent-stat-v">${(s.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="sent-stat">
          <span class="sent-stat-l">✅ Positive</span>
          <span class="sent-stat-v" style="color:var(--green)">${Math.round(s.pos * s.articles)}</span>
        </div>
        <div class="sent-stat">
          <span class="sent-stat-l">❌ Negative</span>
          <span class="sent-stat-v" style="color:var(--red)">${Math.round(s.neg * s.articles)}</span>
        </div>
        <div class="sent-stat">
          <span class="sent-stat-l">⚪ Neutral</span>
          <span class="sent-stat-v" style="color:var(--text-secondary)">${Math.round(s.neu * s.articles)}</span>
        </div>
        <div class="sent-stat">
          <span class="sent-stat-l">Pos Ratio</span>
          <span class="sent-stat-v" style="color:var(--green)">${(s.pos*100).toFixed(1)}%</span>
        </div>
      </div>

      <!-- Model signal link -->
      <div style="margin-top:8px;padding:6px 8px;background:var(--bg-hover);border-radius:4px;display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:9.5px;color:var(--text-muted)">Model Signal</span>
        <span class="signal-badge-sm sig-${getSignalDisplay(TICKERS.find(t => t.ticker===ticker)?.signal || 'neutral').class}">
          ${getSignalDisplay(TICKERS.find(t => t.ticker===ticker)?.signal || 'neutral').text}
        </span>
        <span style="font-size:9.5px;color:var(--text-muted)">Exp. Return</span>
        <span style="font-size:10px;font-weight:700;color:${TICKERS.find(t => t.ticker===ticker)?.exp_return >= 0 ? 'var(--green)' : 'var(--red)'}">
          ${TICKERS.find(t => t.ticker===ticker)?.exp_return > 0 ? '+' : ''}${TICKERS.find(t => t.ticker===ticker)?.exp_return?.toFixed(3) || '0'}%
        </span>
      </div>
    </div>
  `;
}

// ── INIT — called by market.js after API dashboard loads ────────
function initDashboard() {
  if (!TICKERS.length) return;
  selectedTicker = TICKERS[0];
  renderTickerList();
  renderSignalSummary();
  renderMatrix();
  selectTicker(selectedTicker);
}