import React, { useState, useCallback, forwardRef, useImperativeHandle } from "react";

const GOOGL = forwardRef((props, ref) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const fetchData = useCallback(() => {
        setLoading(true);
        const host = window.location.hostname || 'localhost';
        fetch(`http://${host}:5001/googl`)
            .then(r => r.json())
            .then(d => { setData(d); setLoading(false); })
            .catch(() => { setError('Could not load GOOGL prediction.'); setLoading(false); });
    }, []);

    useImperativeHandle(ref, () => ({
        fetchData
    }));

    return(
        <div> 
            {loading && <div className="prediction">Loading GOOGL prediction...</div>}
            {error && <div className="prediction" style={{ color: '#f87171' }}>{error}</div>}
            {data && (
                <div className="prediction">
                    <div className="prediction-header">GOOGL Prediction</div>
                    <div className="prediction-details">
                        <p><strong>Ticker:</strong> {data.ticker}</p>
                        <p><strong>Date:</strong> {data.date}</p>
                        <p><strong>Current Price:</strong> ${data.current_price}</p>
                        <p><strong>Predicted 5D Return:</strong> {data.predicted_5d_return}</p>
                        <p><strong>Predicted 5D %:</strong> {data.predicted_5d_pct}%</p>
                        <p><strong>Direction:</strong> {data.direction}</p>
                    </div>
                    <button className="refresh-button" onClick={fetchData}>Refresh</button>
                </div>
            )}
        </div>

    )
});

export default GOOGL;
