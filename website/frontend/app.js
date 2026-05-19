/**
 * Route Forecaster — Simplified Rendering Logic
 */

const ROUTE_NAMES = {
    'XSICFENE_FarEast_NorthEurope': 'Far East → N. Europe',
    'XSICNEFE_NorthEurope_FarEast': 'N. Europe → Far East',
    'XSICFEUW_FarEast_USWestCoast': 'Far East → US West',
    'XSICUWFE_USWestCoast_FarEast': 'US West → Far East'
};

const COLORS = ['#00ccff', '#aa77ff', '#00ff88', '#ffaa00'];
let chart = null;
let currentRange = 'all';
let streamData = null;

// Terminal Log
const feed = document.getElementById('terminal-feed');
function addLog(msg, type='info') {
    const div = document.createElement('div');
    div.className = 'log-line';
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
    
    let colorClass = 'log-sys';
    if (type === 'data') colorClass = 'log-lane';
    if (type === 'err') colorClass = 'log-err';
    
    div.innerHTML = `<span class="log-ts">${ts}</span> <span class="${colorClass}">${msg}</span>`;
    feed.insertBefore(div, feed.firstChild);
    if (feed.children.length > 50) feed.removeChild(feed.lastChild);
}

function fmt(n) { return Number(n).toLocaleString(undefined, {minimumFractionDigits:0,maximumFractionDigits:0}); }
function fmtd(iso) {
    if(!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
}

async function api(path) {
    try {
        const res = await fetch(path);
        if(!res.ok) throw new Error('HTTP ' + res.status);
        return await res.json();
    } catch(e) {
        addLog(`[API ERROR] Fetching ${path} failed`, 'err');
        throw e;
    }
}

async function fullRefresh() {
    addLog('[SYS] Fetching latest forecasts...', 'info');
    document.getElementById('last-updated').textContent = 'Updating...';
    
    try {
        const [stream, today, hist] = await Promise.all([
            api('/api/predictions/stream'),
            api('/api/predictions/today'),
            api('/api/historical-summary')
        ]);
        
        streamData = stream;
        
        // Update KPIs
        document.getElementById('kpi-cutoff').textContent = fmtd(stream.training_cutoff);
        document.getElementById('kpi-today').textContent = fmtd(stream.today);
        document.getElementById('gap-counter').textContent = stream.days_gap + ' days gap';
        document.getElementById('kpi-preds').textContent = stream.total_predictions.toLocaleString();
        document.getElementById('pred-counter').textContent = stream.total_predictions + ' preds';
        
        let todaySum = 0;
        let todayCount = 0;
        if(today && today.predictions) {
            today.predictions.forEach(p => {
                todaySum += p.predicted_price;
                todayCount++;
            });
        }
        if(todayCount > 0) {
            document.getElementById('kpi-avg').textContent = '$' + fmt(todaySum / todayCount);
        }

        // Ticker & Table
        updateTable(today.predictions, hist);
        updateTicker(today.predictions);
        
        // Chart
        renderChart();
        
        document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
        addLog(`[DATA] Loaded ${stream.total_predictions} predictions across 4 routes.`, 'data');
    } catch(e) {
        addLog('[SYS] Refresh failed.', 'err');
        document.getElementById('last-updated').textContent = 'Error';
    }
}

function updateTable(todayPreds, hist) {
    const tbody = document.getElementById('forecast-body');
    if(!todayPreds) {
        tbody.innerHTML = '<tr><td colspan="6">No predictions available.</td></tr>';
        return;
    }
    
    tbody.innerHTML = todayPreds.map(p => {
        const name = ROUTE_NAMES[p.route] || p.route;
        const mean = (hist && hist[p.route]) ? hist[p.route].mean : null;
        let changeHtml = '--';
        if(mean) {
            const chg = ((p.predicted_price - mean) / mean) * 100;
            const cls = chg >= 0 ? 'price-up' : 'price-down';
            const arrow = chg >= 0 ? '▲' : '▼';
            changeHtml = `<span class="${cls}">${arrow} ${Math.abs(chg).toFixed(1)}%</span>`;
        }
        
        return `<tr>
            <td style="color:var(--text-0);font-weight:500">${name}</td>
            <td><span class="tag tag-route">40FT CONTAINER</span></td>
            <td style="font-weight:700;color:var(--green)">$${fmt(p.predicted_price)}</td>
            <td style="color:var(--text-2)">${(p.confidence*100).toFixed(0)}%</td>
            <td style="color:var(--text-2)">${mean ? '$'+fmt(mean) : '--'}</td>
            <td>${changeHtml}</td>
        </tr>`;
    }).join('');
}

function updateTicker(todayPreds) {
    const scroll = document.getElementById('ticker-scroll');
    if(!todayPreds) return;
    scroll.innerHTML = todayPreds.map(p => {
        const name = ROUTE_NAMES[p.route] || p.route;
        return `<div class="ticker-item">
            <span class="ticker-name">${name}</span>
            <span class="ticker-price">$${fmt(p.predicted_price)}</span>
        </div>`;
    }).join('');
}

function setRange(val) {
    currentRange = val;
    document.querySelectorAll('.chart-controls .btn').forEach(b => {
        b.classList.toggle('btn-active', String(b.dataset.range) === String(val));
    });
    renderChart();
}

function renderChart() {
    if (!streamData || !streamData.predictions_by_route) return;

    const byRoute = streamData.predictions_by_route;
    const datasets = [];

    Object.keys(byRoute).forEach((route, idx) => {
        const points = byRoute[route]
            .sort((a,b) => new Date(a.date) - new Date(b.date))
            .filter(p => {
                if (currentRange === 'all') return true;
                const d = (Date.now() - new Date(p.date).getTime()) / 86400000;
                return d <= currentRange;
            });
        if (!points.length) return;

        const visualOffset = (idx % 2 === 0) ? 0 : 40;
        
        datasets.push({
            label: ROUTE_NAMES[route] || route,
            data: points.map(p => ({ 
                x: new Date(p.date), 
                y: p.predicted_price + visualOffset,
                _real_price: p.predicted_price 
            })),
            borderColor: COLORS[idx % COLORS.length],
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2,
            fill: false
        });
    });

    if (chart) chart.destroy();

    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Default config matching the dark aesthetic
    Chart.defaults.color = '#6a6a7e';
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
    Chart.defaults.font.size = 10;

    chart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#b0b0be', boxWidth: 10, usePointStyle: true }
                },
                tooltip: {
                    backgroundColor: '#16161e',
                    titleColor: '#ffffff',
                    bodyColor: '#b0b0be',
                    borderColor: '#2a2a35',
                    borderWidth: 1,
                    callbacks: { label: c => '  ' + c.dataset.label + ': $' + fmt(c.raw._real_price) }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    time: { unit: 'day' }
                },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { callback: v => '$' + fmt(v) }
                }
            }
        }
    });
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
    addLog('[SYS] Booting frontend dashboard...', 'info');
    fullRefresh();
    setInterval(fullRefresh, 5 * 60 * 1000); // 5 min
});
