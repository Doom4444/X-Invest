

let equityChart = null;


document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('endDate').value = today;

    
    document.getElementById('startDate').max = today;
    document.getElementById('endDate').max = today;
});

async function runBacktest() {
    const ticker = document.getElementById('ticker').value.trim();
    const initialCapital = parseFloat(document.getElementById('initialCapital').value);
    const startDate = document.getElementById('startDate').value;

    if (!ticker) {
        alert('⚠️ Please select a ticker symbol from the list.');
        return;
    }

    const btn = document.getElementById('runBtn');
    const resultsContainer = document.getElementById('resultsContainer');

    
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>Running Backtest...';
    resultsContainer.classList.remove('show');

    try {
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: ticker,
                initial_capital: initialCapital,
                start: startDate
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server responded with HTTP ${response.status}`);
        }

        const data = await response.json();
        console.log("Backtest response:", data); // للـ debugging

        
        if (!data.success) {
            throw new Error(data.error || 'Backtest simulation failed');
        }

        if (!data.equity || !data.metrics) {
            throw new Error('Invalid response: missing equity or metrics');
        }

       
        if (!data.trades) {
            // console.warn('Warning: trades is missing from response');
            data.trades = []; // تعيين قيمة افتراضية
        }

        
        displayResults(data);
        resultsContainer.classList.add('show');

        
        setTimeout(() => {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

    } catch (error) {
        alert(`❌ Error running backtest:\n${error.message}`);
        console.error('Backtest Error:', error);
    } finally {
        
        btn.disabled = false;
        btn.innerHTML = '🚀 Run Backtest';
    }
}

function displayResults(data) {
    const equity = data.equity || [];
    const trades = data.trades || [];
    const metrics = data.metrics || {};

    
    displayKPIs(metrics, equity, trades);
    drawEquityCurve(equity);
    displayTrades(trades);
}

function displayKPIs(metrics, equity,trades) {
    const kpiGrid = document.getElementById('kpiGrid');
    const initialCapital = equity[0] || 10000;
    const finalCapital = equity[equity.length - 1] || initialCapital;
    const totalReturn = ((finalCapital - initialCapital) / initialCapital) * 100;

    const kpis = [
        {
            label: 'Total Return',
            value: `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`,
            delta: `$${(finalCapital - initialCapital).toFixed(2)}`,
            className: totalReturn >= 0 ? 'positive' : 'negative',
            info: 'Overall profit/loss as a percentage of initial capital.'
        },
        {
            label: 'Win Rate',
            value: `${(metrics.win_rate * 100).toFixed(1)}%`,
            delta: `${trades.length} trades`,
            className: metrics.win_rate >= 0.5 ? 'positive' : 'negative',
            info: 'Percentage of profitable trades out of total executed.'
        },
        {
            label: 'Profit Factor',
            value: metrics.profit_factor.toFixed(2),
            delta: metrics.profit_factor >= 1.3 ? '✅ Good' : '⚠️ Low',
            className: metrics.profit_factor >= 1.5 ? 'positive' : metrics.profit_factor >= 1.0 ? 'neutral' : 'negative',
            info: 'Gross Profit ÷ Gross Loss. Ratio > 1.5 is considered excellent.'
        },
        {
            label: 'Max Drawdown',
            value: `${(metrics.max_dd * 100).toFixed(1)}%`,
            delta: metrics.max_dd <= 0.2 ? '✅ Low Risk' : '⚠️ High Risk',
            className: metrics.max_dd <= 0.15 ? 'positive' : metrics.max_dd <= 0.25 ? 'neutral' : 'negative',
            info: 'Largest peak-to-trough decline. Measures worst-case temporary loss.'
        },
        {
            label: 'Sharpe Ratio',
            value: metrics.sharpe.toFixed(2),
            delta: metrics.sharpe >= 1.0 ? '✅ Good' : metrics.sharpe >= 0.5 ? '⚠️ Moderate' : '❌ Poor',
            className: metrics.sharpe >= 1.5 ? 'positive' : metrics.sharpe >= 0.5 ? 'neutral' : 'negative',
            info: 'Risk-adjusted return. >1.0 means return justifies the risk taken.'
        },
        {
            label: 'Final Equity',
            value: `$${finalCapital.toFixed(2)}`,
            delta: `Started: $${initialCapital.toFixed(2)}`,
            className: finalCapital >= initialCapital ? 'positive' : 'negative',
            info: 'Total portfolio value at the end of the backtest period.'
        }
    ];

    kpiGrid.innerHTML = kpis.map(kpi => `
        <div class="kpi-card ${kpi.className}">
            <div class="kpi-label">
                ${kpi.label}
                <span class="tooltip">?
                    <span class="tooltip-text">${kpi.info}</span>
                </span>
            </div>
            <div class="kpi-value">${kpi.value}</div>
            <div class="kpi-delta ${kpi.className}">${kpi.delta}</div>
        </div>
    `).join('');
}

function drawEquityCurve(equity) {
    const ctx = document.getElementById('equityChart').getContext('2d');

    // تدمير الشارت القديم إن وُجد لتجنب التداخل
    if (equityChart) {
        equityChart.destroy();
    }

    const labels = equity.map((_, i) => `Day ${i + 1}`);

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Portfolio Value',
                data: equity,
                borderColor: '#ffa502',
                backgroundColor: 'rgba(255, 165, 2, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    labels: { color: '#dfe6e9', font: { family: 'system-ui' } }
                },
                tooltip: {
                    backgroundColor: 'rgba(7, 26, 47, 0.95)',
                    titleColor: '#ffa502',
                    bodyColor: '#dfe6e9',
                    borderColor: 'rgba(255, 165, 2, 0.3)',
                    borderWidth: 1,
                    callbacks: {
                        label: function (context) {
                            return `Value: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#c7d0d9', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#c7d0d9',
                        callback: function (value) { return '$' + value.toFixed(0); }
                    }
                }
            }
        }
    });
}

function displayTrades(trades) {
    const tbody = document.getElementById('tradesTableBody');

    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8">No trades executed</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${new Date(trade.entry).toLocaleDateString()}</td>
            <td>${new Date(trade.exit).toLocaleDateString()}</td>
            <td>$${trade.entry_price.toFixed(2)}</td>
            <td>$${trade.exit_price.toFixed(2)}</td>
            <td>${trade.shares}</td>
            <td>${trade.hold_days} days</td>
            <td style="color: ${trade.pnl >= 0 ? 'green' : 'red'}">
                $${trade.pnl.toFixed(2)}
            </td>
            <td style="color: ${trade.pnl_pct >= 0 ? 'green' : 'red'}">
                ${trade.pnl_pct.toFixed(2)}%
            </td>
        </tr>
    `).join('');
}