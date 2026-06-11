// ── PRICE CHART ───────────────────────────────────────────────────
const canvas = document.getElementById('mainChart');
const ctx    = canvas.getContext('2d');

function drawPriceChart() {
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  if (!W || !H) return;

  canvas.width  = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  ctx.clearRect(0, 0, W, H);

  if (!selectedTicker) return;

  // ── Apply year range filter from UI inputs ──────────────────────
  const yearFrom = parseInt(document.getElementById('yearFrom')?.value) || 2018;
  const yearTo   = parseInt(document.getElementById('yearTo')?.value)   || 2026;

  const allHistory = TICKER_HISTORY[selectedTicker.ticker] || [];
  const history = allHistory.filter(d => {
    const yr = parseInt((d.date || '').slice(0, 4));
    return yr >= yearFrom && yr <= yearTo;
  });
  if (!history.length) return;

  const prices  = history.map(d => d.close);
  const vols    = history.map(d => d.vol);
  const dates   = history.map(d => d.date || '');

  const type    = document.getElementById('chartTypeSelect')?.value || 'line';

  const padL = 52, padR = 16, padT = 16, padB = 46;
  const cW = W - padL - padR;
  const cH = H - padT - padB - 36; // reserve bottom for volume

  const minP   = Math.min(...prices) * 0.998;
  const maxP   = Math.max(...prices) * 1.002;
  const range  = maxP - minP;
  const px2y   = p => padT + cH - ((p - minP) / range) * cH;
  const ix2x   = i => padL + (i / (prices.length - 1)) * cW;

  const isUp   = prices[prices.length - 1] >= prices[0];
  const mainColor = isUp ? '#22c55e' : '#ef4444';

  // ── Background
  ctx.fillStyle = '#080b12';
  ctx.fillRect(0, 0, W, H);

  // ── Grid
  const gridCount = 5;
  ctx.setLineDash([2, 5]);
  ctx.lineWidth   = 0.5;
  for (let i = 0; i <= gridCount; i++) {
    const y     = padT + (cH / gridCount) * i;
    const price = maxP - (range / gridCount) * i;
    ctx.strokeStyle = '#1a2844';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + cW, y); ctx.stroke();
    ctx.fillStyle = '#334466'; ctx.font = '8.5px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText('$' + price.toFixed(2), padL - 4, y + 3);
  }
  ctx.setLineDash([]);

  // ── Date labels (every ~10 points)
  ctx.fillStyle  = '#334466';
  ctx.font       = '8.5px JetBrains Mono';
  ctx.textAlign  = 'center';
  const step = Math.ceil(prices.length / 8);
  prices.forEach((_, i) => {
    if (i % step === 0 && dates[i]) {
      const x   = ix2x(i);
      const lbl = dates[i].slice(5); // "MM-DD"
      ctx.fillText(lbl, x, padT + cH + 14);
    }
  });

  // ── Area fill
  if (type === 'area') {
    const grad = ctx.createLinearGradient(0, padT, 0, padT + cH);
    grad.addColorStop(0, isUp ? '#22c55e28' : '#ef444428');
    grad.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.moveTo(ix2x(0), px2y(prices[0]));
    prices.forEach((p, i) => { if (i > 0) ctx.lineTo(ix2x(i), px2y(p)); });
    ctx.lineTo(ix2x(prices.length - 1), padT + cH);
    ctx.lineTo(ix2x(0), padT + cH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // ── Price line
  ctx.beginPath();
  ctx.strokeStyle = mainColor;
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  prices.forEach((p, i) => {
    i === 0 ? ctx.moveTo(ix2x(0), px2y(p)) : ctx.lineTo(ix2x(i), px2y(p));
  });
  ctx.stroke();

  // ── Last price dot
  const lastX = ix2x(prices.length - 1);
  const lastY = px2y(prices[prices.length - 1]);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = mainColor;
  ctx.fill();

  // ── Last price tag
  ctx.fillStyle = mainColor;
  ctx.beginPath();
  const tagW = 56, tagH = 16;
  ctx.roundRect(lastX - tagW - 4, lastY - tagH / 2, tagW, tagH, 3);
  ctx.fill();
  ctx.fillStyle  = 'white';
  ctx.font       = 'bold 9px JetBrains Mono';
  ctx.textAlign  = 'center';
  ctx.fillText('$' + prices[prices.length - 1].toFixed(2), lastX - tagW / 2 - 4, lastY + 3.5);

  // ── Volume bars
  const volBase  = padT + cH + 22;
  const volH     = 22;
  const maxVol   = Math.max(...vols);
  const barW     = Math.max(1, cW / prices.length - 1);
  vols.forEach((v, i) => {
    const x   = padL + (i / prices.length) * cW;
    const bh  = (v / maxVol) * volH;
    const isGreen = i === 0 ? true : prices[i] >= prices[i - 1];
    ctx.fillStyle = isGreen ? '#22c55e44' : '#ef444444';
    ctx.fillRect(x, volBase + volH - bh, barW, bh);
  });

  // ── Vol label
  ctx.fillStyle = '#334466'; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'left';
  ctx.fillText('VOL', padL + 4, volBase + 10);
}

function resizeChart() { drawPriceChart(); }
window.addEventListener('resize', resizeChart);
setTimeout(resizeChart, 60);

// Timeframe buttons (visual only — data is fixed)
document.querySelectorAll('.time-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    drawPriceChart();
  });
});