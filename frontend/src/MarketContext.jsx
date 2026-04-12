import React, { useEffect, useState } from 'react';

export default function MarketContext() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch('http://localhost:5001/market-context')
            .then(r => r.json())
            .then(d => { setData(d); setLoading(false); })
            .catch(() => { setError('Could not load market data.'); setLoading(false); });
    }, []);

    if (loading) return <div className="mc-wrap"><div className="mc-header">Loading market data...</div></div>;
    if (error) return <div className="mc-wrap"><div className="mc-header" style={{ color: '#f87171' }}>{error}</div></div>;

    const fmt = (val, decimals = 2) => val != null ? Number(val).toFixed(decimals) : '—';
    const pct = val => val != null ? `${val >= 0 ? '+' : ''}${fmt(val)}%` : '—';
    const color = val => val > 0 ? 'up' : val < 0 ? 'down' : 'neutral';

    const vixColor = data.vix_change > 0 ? 'down' : 'up';   // VIX up = bad
    const trendText = data.spy_vs_sma20 > 1 ? 'Above 20d avg' : 'Below 20d avg';
    const trendDir = data.spy_vs_sma20 > 1 ? 'Bullish' : 'Bearish';
    const trendCol = data.spy_vs_sma20 > 1 ? 'up' : 'down';

    return (
        <div className="mc-wrap">
            <div className="mc-header">Market context — today</div>
            <div className="mc-grid">

                <div className="mc-card">
                    <div className="mc-label">S&P 500 (SPY)</div>
                    <div className={`mc-value ${color(data.spy_return)}`}>${fmt(data.spy_price)}</div>
                    <div className={`mc-change ${color(data.spy_return)}`}>
                        <span className="mc-dot" style={{ background: data.spy_return >= 0 ? '#4ade80' : '#f87171' }}></span>
                        {pct(data.spy_return * 100)}
                    </div>
                </div>

                <div className="mc-card">
                    <div className="mc-label">Tech (XLK)</div>
                    <div className={`mc-value ${color(data.xlk_return)}`}>${fmt(data.xlk_price)}</div>
                    <div className={`mc-change ${color(data.xlk_return)}`}>
                        <span className="mc-dot" style={{ background: data.xlk_return >= 0 ? '#4ade80' : '#f87171' }}></span>
                        {pct(data.xlk_return * 100)}
                    </div>
                </div>

                <div className="mc-card">
                    <div className="mc-label">VIX (Fear)</div>
                    <div className="mc-value neutral">{fmt(data.vix_level, 1)}</div>
                    <div className={`mc-change ${vixColor}`}>
                        <span className="mc-dot" style={{ background: data.vix_change > 0 ? '#f87171' : '#4ade80' }}></span>
                        {data.vix_change >= 0 ? '+' : ''}{fmt(data.vix_change, 1)} pts
                    </div>
                </div>

                <div className="mc-card">
                    <div className="mc-label">Market trend</div>
                    <div className={`mc-value ${trendCol}`} style={{ fontSize: 15, paddingTop: 4 }}>{trendText}</div>
                    <div className={`mc-change ${trendCol}`}>
                        <span className="mc-dot" style={{ background: trendCol === 'up' ? '#4ade80' : '#f87171' }}></span>
                        {trendDir}
                    </div>
                </div>

            </div>
            <div className="mc-footer">~15 min delay · yfinance</div>
        </div>
    );
}