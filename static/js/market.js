// Owner: market.js
// Frontend for the X‑Invest Markets page.
// Talks to:
//   GET /api/market/companies -> list of companies
//   GET /api/market/{ticker}  -> dashboard data for one company

const companyListEl = document.getElementById("company-list");
const dashboardPanelEl = document.getElementById("dashboard-panel");
const marketContainerEl = document.getElementById("market-container");
let sidebarToggleBtn = null;

function ensureSidebarToggle() {
  if (!marketContainerEl || sidebarToggleBtn) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "company-toggle";
  btn.className = "company-toggle";
  btn.setAttribute("aria-label", "Toggle companies list");
  btn.setAttribute("aria-expanded", "true");
  btn.innerHTML = "<span>&laquo;</span>";
  btn.addEventListener("click", () => {
    if (!marketContainerEl) return;
    const collapsed = marketContainerEl.classList.toggle("sidebar-collapsed");
    btn.setAttribute("aria-expanded", String(!collapsed));
    btn.innerHTML = collapsed ? "<span>&raquo;</span>" : "<span>&laquo;</span>";
  });
  marketContainerEl.appendChild(btn);
  sidebarToggleBtn = btn;
}

async function loadCompanies() {
  if (!companyListEl) return;
  companyListEl.innerHTML = `
    <div class="company-list-header">
      <div>
        <strong>Companies</strong>
      </div>
      <span>Loading curated tickers…</span>
    </div>
  `;

  try {
    const res = await fetch("/api/market/companies");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    renderCompanyCards(Array.isArray(data) ? data : []);
  } catch (e) {
    companyListEl.innerHTML = `
      <div class="company-list-header">
        <div><strong>Companies</strong></div>
      </div>
      <p style="font-size:0.85rem;color:#f87171;">
        Could not load companies: ${(e && e.message) || "Unknown error"}
      </p>
    `;
  }
}

function renderCompanyCards(companies) {
  if (!companyListEl) return;

  const header = document.createElement("div");
  header.className = "company-list-header";
  header.innerHTML = `
    <div>
      <strong>Companies</strong>
    </div>
    <span>${companies.length} symbols</span>
  `;

  const grid = document.createElement("div");
  grid.className = "company-grid";

  companies.forEach((c) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "company-card";
    card.innerHTML = `
      <div class="ticker-row">
        <span class="ticker">${c.ticker}</span>
        <span class="flag">${c.flag || ""}</span>
      </div>
      <div class="name">${c.name_en || c.name_ar || c.ticker}</div>
      <div class="meta">
        ${c.market || ""} • ${c.sector || ""}
      </div>
    `;
    card.addEventListener("click", () => showDashboard(c.ticker));
    grid.appendChild(card);
  });

  companyListEl.innerHTML = "";
  companyListEl.appendChild(header);
  companyListEl.appendChild(grid);
}

async function showDashboard(ticker) {
  if (!dashboardPanelEl) return;
  if (marketContainerEl) {
    marketContainerEl.classList.add("has-dashboard");
    ensureSidebarToggle();
  }
  dashboardPanelEl.classList.remove("hidden");
  dashboardPanelEl.innerHTML = `
    <div class="dashboard-empty">
      Loading dashboard for <strong style="margin-inline:0.2rem;">${ticker}</strong>…
    </div>
  `;

  try {
    const res = await fetch(`/api/market/${encodeURIComponent(ticker)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const msg = err.error || `HTTP ${res.status}`;
      dashboardPanelEl.innerHTML = `
        <div class="dashboard-empty">
          Could not load data for <strong style="margin-inline:0.2rem;">${ticker}</strong>.<br/>
          <span style="color:#f87171;font-size:0.85rem;">${msg}</span>
        </div>
      `;
      return;
    }
    const data = await res.json();
    renderDashboard(data);
  } catch (e) {
    dashboardPanelEl.innerHTML = `
      <div class="dashboard-empty">
        Network error while loading <strong style="margin-inline:0.2rem;">${ticker}</strong>.<br/>
        <span style="color:#f87171;font-size:0.85rem;">
          ${(e && e.message) || "Unknown error"}
        </span>
      </div>
    `;
  }
}

function formatNumber(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1_000_000_000) return (n / 1_000_000_000).toFixed(digits) + "B";
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(digits) + "M";
  if (Math.abs(n) >= 1_000) return (n / 1_000).toFixed(digits) + "K";
  return n.toFixed(digits);
}

function renderDashboard(d) {
  if (!dashboardPanelEl) return;
  const change = d.change;
  const changePct = d.change_pct;
  const positive = typeof change === "number" && change > 0;
  const negative = typeof change === "number" && change < 0;

  const changeClass = positive ? "positive" : negative ? "negative" : "";
  const changeText =
    change == null && changePct == null
      ? "—"
      : `${change != null ? change.toFixed(2) : "—"} (${changePct != null ? changePct.toFixed(2) + "%" : "—"})`;

  const news = Array.isArray(d.news) ? d.news : [];

  dashboardPanelEl.innerHTML = `
    <div class="dashboard-header">
      <div class="dashboard-title">
        <h2>${d.name_en || d.ticker}</h2>
        ${
          d.name_ar
            ? `<span>${d.name_ar}</span>`
            : ""
        }
      </div>
      <div class="dashboard-price">
        <div class="dashboard-price-main">
          ${d.price != null ? d.price.toFixed(2) : "—"} ${d.currency || ""}
        </div>
        <div class="dashboard-change ${changeClass}">
          ${changeText}
        </div>
      </div>
    </div>

    <div class="dashboard-meta-grid">
      <div class="dashboard-meta-item">
        <span>Ticker</span>
        <span>${d.ticker}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Market</span>
        <span>${d.market || "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Sector</span>
        <span>${d.sector || "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Market cap</span>
        <span>${formatNumber(d.market_cap)}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>P/E ratio</span>
        <span>${d.pe_ratio != null ? d.pe_ratio.toFixed(2) : "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>P/B ratio</span>
        <span>${d.pb_ratio != null ? d.pb_ratio.toFixed(2) : "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>EPS</span>
        <span>${d.eps != null ? d.eps.toFixed(2) : "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Dividend yield</span>
        <span>${d.dividend != null ? (d.dividend * 100).toFixed(2) + "%" : "—"}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>52‑week range</span>
        <span>${d.week52_low != null ? d.week52_low.toFixed(2) : "—"} – ${
          d.week52_high != null ? d.week52_high.toFixed(2) : "—"
        }</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Day range</span>
        <span>${d.day_low != null ? d.day_low.toFixed(2) : "—"} – ${
          d.day_high != null ? d.day_high.toFixed(2) : "—"
        }</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Volume (avg)</span>
        <span>${formatNumber(d.volume)} / ${formatNumber(d.avg_volume)}</span>
      </div>
      <div class="dashboard-meta-item">
        <span>Website</span>
        <span>
          ${
            d.website
              ? `<a href="${d.website}" target="_blank" rel="noopener noreferrer">${d.website}</a>`
              : "—"
          }
        </span>
      </div>
    </div>

    <div class="dashboard-news">
      <h3>Latest news</h3>
      ${
        news.length === 0
          ? `<p style="font-size:0.8rem;color:var(--text-muted);">No recent news available.</p>`
          : `<ul>${news
              .map(
                (n) => `
                  <li>
                    <a href="${n.link || "#"}" target="_blank" rel="noopener noreferrer">
                      ${n.title}
                    </a>
                  </li>
                `
              )
              .join("")}</ul>`
      }
    </div>
  `;
}

if (companyListEl) {
  window.addEventListener("DOMContentLoaded", loadCompanies);
}

