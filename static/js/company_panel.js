// ── COMPANY PANEL — live API fetch ────────────────────────────────
// GET /api/market/{ticker}/forecast

function getCompanySignalDisplay(signal) {
  const sig = String(signal).toLowerCase().trim();
  if (sig === 'buy') {
    return { text: '▲ BUY', class: 'sig-bullish' };
  } else if (sig === 'sell') {
    return { text: '▼ SELL', class: 'sig-bearish' };
  } else {
    return { text: '● HOLD', class: 'sig-neutral' };
  }
}

function showForecastLoading(ticker) {
  const body  = document.getElementById('companyPanelBody');
  const label = document.getElementById('cpTickerLabel');
  if (label) label.textContent = ticker;
  if (body) {
    body.innerHTML = `<div class="cp-no-data">Loading forecast for <strong>${ticker}</strong>…</div>`;
  }
}

async function renderCompanyPanel(ticker) {
  const body  = document.getElementById('companyPanelBody');
  const label = document.getElementById('cpTickerLabel');
  if (!body) return;

  showForecastLoading(ticker);

  try {
    const res = await fetch(`/api/market/${encodeURIComponent(ticker)}/forecast`, { cache: 'no-store' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const { meta, forecasts } = await res.json();
    if (label) label.textContent = meta.ticker;

    const signalDisplay = getCompanySignalDisplay(meta.signal);
    const dirUp      = meta.direction === 'UP';
    const dirNeutral = meta.direction === 'NEUTRAL';
    const dirArrow   = dirUp ? '▲' : dirNeutral ? '●' : '▼';
    const dirClass   = dirUp ? 'cp-dir-up' : dirNeutral ? 'cp-dir-neu' : 'cp-dir-dn';

    const targetDiff  = meta.price_target - meta.current_price;
    const targetPct   = (targetDiff / meta.current_price) * 100;
    const targetPos   = targetDiff >= 0;
    const targetColor = targetPos ? 'var(--green)' : 'var(--red)';

    const lastForecast = forecasts[forecasts.length - 1]?.forecast_price ?? meta.current_price;
    const forecastDiff = lastForecast - meta.current_price;
    const forecastPct  = (forecastDiff / meta.current_price) * 100;
    const forecastPos  = forecastDiff >= 0;

    const dayCount = forecasts.length;
    const svgChart   = buildForecastSVG(meta.current_price, forecasts);

    body.innerHTML = `
      <div class="cp-meta-card">
        <div class="cp-sig-row">
          <span class="cp-sig-badge ${signalDisplay.class}">${signalDisplay.text}</span>
          <span class="cp-dir ${dirClass}">${dirArrow} ${meta.direction}</span>
          <span class="cp-sector-pill">${meta.sector}</span>
        </div>

        <div class="cp-price-block">
          <div class="cp-current-price">$${meta.current_price.toFixed(2)}</div>
          <div class="cp-price-lbl">Expected Price</div>
        </div>

        <div class="cp-kv-grid">
          <div class="cp-kv">
            <span class="cp-k">Price Target</span>
            <span class="cp-v" style="color:${targetColor}">$${meta.price_target.toFixed(2)}</span>
          </div>
          <div class="cp-kv">
            <span class="cp-k">Target Return</span>
            <span class="cp-v" style="color:${targetColor}">
              ${targetPos ? '+' : ''}${targetPct.toFixed(2)}%
            </span>
          </div>
          <div class="cp-kv">
            <span class="cp-k">End Forecast</span>
            <span class="cp-v" style="color:${forecastPos ? 'var(--green)' : 'var(--red)'}">
              $${lastForecast.toFixed(2)}
            </span>
          </div>
          <div class="cp-kv">
            <span class="cp-k">Forecast Δ</span>
            <span class="cp-v" style="color:${forecastPos ? 'var(--green)' : 'var(--red)'}">
              ${forecastPos ? '+' : ''}${forecastPct.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <div class="cp-section-title">${dayCount}-Day Forecast with Bands</div>
      <div class="cp-chart-wrap">${svgChart}</div>

      <div class="cp-section-title" style="margin-top:8px">Forecast Table</div>
      <div class="cp-table-wrap">
        <table class="cp-table">
          <thead>
            <tr><th>Date</th><th>Forecast</th><th>Upper</th><th>Lower</th></tr>
          </thead>
          <tbody>
            ${forecasts.map(f => {
              const up = f.forecast_price >= meta.current_price;
              return `<tr>
                <td>${f.date.slice(5)}</td>
                <td class="${up ? 'pos-val' : 'neg-val'}">$${f.forecast_price.toFixed(2)}</td>
                <td style="color:var(--green);opacity:0.75">$${f.upper_band.toFixed(2)}</td>
                <td style="color:var(--red);opacity:0.75">$${f.lower_band.toFixed(2)}</td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (err) {
    console.error('[company_panel]', err);
    body.innerHTML = `
      <div class="cp-error">
        <div class="cp-error-icon">⚠️</div>
        <div class="cp-error-title">Could not load forecast</div>
        <div class="cp-error-msg">${ticker}<br><br>${err.message}</div>
      </div>`;
  }
}

function buildForecastSVG(currentPrice, forecasts) {
  if (!forecasts.length) {
    return `<div class="cp-no-data">No forecast points</div>`;
  }

  const W = 220, H = 110;
  const padL = 36, padR = 6, padT = 8, padB = 18;
  const cW = W - padL - padR, cH = H - padT - padB;
  const allP = forecasts.flatMap(f => [f.forecast_price, f.upper_band, f.lower_band]);
  allP.push(currentPrice);
  const minP = Math.min(...allP) * 0.997;
  const maxP = Math.max(...allP) * 1.003;
  const rng  = maxP - minP;
  const py = p => padT + cH - ((p - minP) / rng) * cH;
  const px = i => padL + (i / Math.max(forecasts.length - 1, 1)) * cW;
  const upperPts = forecasts.map((f, i) => `${px(i)},${py(f.upper_band)}`).join(' ');
  const lowerPts = [...forecasts].reverse().map((f, i) =>
    `${px(forecasts.length - 1 - i)},${py(f.lower_band)}`).join(' ');
  const fLine = forecasts.map((f, i) =>
    `${i === 0 ? 'M' : 'L'}${px(i)},${py(f.forecast_price)}`).join(' ');
  const refY = py(currentPrice);
  const yLabels = Array.from({ length: 5 }, (_, i) => {
    const price = minP + (rng / 4) * i;
    return `<text x="${padL - 3}" y="${py(price) + 3}" text-anchor="end" fill="#334466" font-size="7" font-family="monospace">$${price.toFixed(0)}</text>`;
  }).join('');
  const xLabels = `<text x="${padL}" y="${H - 4}" text-anchor="middle" fill="#334466" font-size="7" font-family="monospace">${forecasts[0].date.slice(5)}</text>` +
                  `<text x="${padL + cW}" y="${H - 4}" text-anchor="middle" fill="#334466" font-size="7" font-family="monospace">${forecasts[forecasts.length - 1].date.slice(5)}</text>`;
  const isUp = forecasts[forecasts.length - 1].forecast_price >= currentPrice;
  const col  = isUp ? '#22c55e' : '#ef4444';
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">` +
    `<rect width="${W}" height="${H}" fill="#080b12" rx="4"/>` +
    `<polygon points="${upperPts} ${lowerPts}" fill="${col}" fill-opacity="0.08"/>` +
    `<polyline points="${forecasts.map((f, i) => `${px(i)},${py(f.upper_band)}`).join(' ')}" fill="none" stroke="#22c55e" stroke-width="0.8" stroke-opacity="0.4" stroke-dasharray="3,3"/>` +
    `<polyline points="${forecasts.map((f, i) => `${px(i)},${py(f.lower_band)}`).join(' ')}" fill="none" stroke="#ef4444" stroke-width="0.8" stroke-opacity="0.4" stroke-dasharray="3,3"/>` +
    `<line x1="${padL}" y1="${refY}" x2="${padL + cW}" y2="${refY}" stroke="#f59e0b" stroke-width="0.7" stroke-dasharray="4,3" stroke-opacity="0.6"/>` +
    `<path d="${fLine}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/>` +
    `<circle cx="${px(forecasts.length - 1)}" cy="${py(forecasts[forecasts.length - 1].forecast_price)}" r="2.5" fill="${col}"/>` +
    yLabels + xLabels +
    `</svg>`;
}

(function () {
  const _orig = window.selectTicker;
  window.selectTicker = function (t) {
    if (_orig) _orig(t);
    renderCompanyPanel(t.ticker);
  };
})();
