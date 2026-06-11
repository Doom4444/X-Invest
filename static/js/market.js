// static/js/market.js — API-driven data loader
let TICKERS        = [];
let TICKER_HISTORY = {};
let MACRO          = {};
const SENTIMENT    = {};

function showMarketLoading(msg) {
  const list = document.getElementById('tickerList');
  if (list) {
    list.innerHTML = `<div class="market-loading">${msg || 'Loading market data…'}</div>`;
  }
}

function showMarketError(message) {
  const strip = document.querySelector('.macro-strip');
  if (strip) {
    strip.innerHTML =
      `<div class="market-error">⚠ ${message}</div>`;
  }
  const list = document.getElementById('tickerList');
  if (list) {
    list.innerHTML = `<div class="market-error">${message}</div>`;
  }
}

async function loadDashboardData() {
  showMarketLoading('Loading market data…');
  try {
    const res = await fetch('/api/market/dashboard', { cache: 'no-store' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();

    MACRO   = data.macro || {};
    TICKERS = data.tickers || [];

    updateMacroStrip();
    console.log(`[market.js] ${TICKERS.length} tickers loaded from API`);

    if (typeof initDashboard === 'function') {
      initDashboard();
    }
  } catch (err) {
    console.error('[market.js]', err);
    showMarketError(err.message || 'Failed to load market data');
  }
}

async function loadTickerHistory(ticker) {
  const yearFrom = parseInt(document.getElementById('yearFrom')?.value, 10) || 2020;
  const yearTo   = parseInt(document.getElementById('yearTo')?.value, 10)   || 2026;

  try {
    const url = `/api/market/${encodeURIComponent(ticker)}/history?from=${yearFrom}&to=${yearTo}`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    TICKER_HISTORY[ticker] = data.history || [];
    if (typeof drawPriceChart === 'function') drawPriceChart();
  } catch (err) {
    console.error('[market.js] history', ticker, err);
    TICKER_HISTORY[ticker] = [];
    if (typeof drawPriceChart === 'function') drawPriceChart();
  }
}

function formatChg(val) {
  if (val == null || Number.isNaN(val)) return '';
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(2)}%`;
}

function updateMacroStrip() {
  const set = (id, val, chg, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    const chgStr = chg != null ? ` ${formatChg(chg)}` : '';
    el.textContent = val + chgStr;
    el.className = 'macro-val' + (cls ? ' ' + cls : '');
  };

  const cls = (chg) => (chg >= 0 ? 'pos' : 'neg');

  set('m-vix',   MACRO.vix?.toFixed(2),           MACRO.vix_chg,       MACRO.vix > 20 ? 'neg' : 'pos');
  set('m-sp500', MACRO.sp500?.toLocaleString(),   MACRO.sp500_chg,     cls(MACRO.sp500_chg || 0));
  set('m-10y',   `${MACRO.tnx_10y?.toFixed(2)}%`, MACRO.tnx_10y_chg,   cls(MACRO.tnx_10y_chg || 0));
  set('m-oil',   `$${MACRO.oil?.toFixed(2)}`,     MACRO.oil_chg,       cls(MACRO.oil_chg || 0));
  set('m-gold',  `$${MACRO.gold?.toLocaleString()}`, MACRO.gold_chg,     cls(MACRO.gold_chg || 0));
  set('m-btc',   `$${MACRO.btc?.toLocaleString()}`, MACRO.btc_chg,      cls(MACRO.btc_chg || 0));
  set('m-dxy',   MACRO.dxy?.toFixed(2),           MACRO.dxy_chg,       cls(MACRO.dxy_chg || 0));
  set('m-fed',   `${MACRO.fed_rate?.toFixed(2)}%`, null,               null);
  set('m-cpi',   `${MACRO.cpi?.toFixed(1)}%`,     null,               MACRO.cpi > 3 ? 'neg' : 'pos');
}

loadDashboardData();
