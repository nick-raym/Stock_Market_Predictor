import './App.css'
import React, { useState, useRef } from 'react';
import MarketContext from './MarketContext';
import AAPL from './AAPL';

function App() {
  const [activeTicker, setActiveTicker] = useState('AAPL');
  const aaplRef = useRef();

  const handlePredict = () => {
    if (activeTicker === 'AAPL' && aaplRef.current) {
      aaplRef.current.fetchData();
    }
    // Add logic for other tickers when implemented
  };

  return (
    <>
      <section id="center">
        <h1 className="text-predictor">Stock Market Predictor</h1>

        <div class="predictor-box">
          <h2 className="text-model">Choose a Model</h2>

          <ul className="predictor-options">
            <button
              className={`ticker-button ${activeTicker === 'AAPL' ? 'active' : ''}`}
              onClick={() => setActiveTicker('AAPL')}
            >
              APPLE PREDICTOR
            </button>
            <button
              className={`ticker-button ${activeTicker === 'GOOGL' ? 'active' : ''}`}
              onClick={() => setActiveTicker('GOOGL')}
            >
              GOOGLE PREDICTOR
            </button>
            <button
              className={`ticker-button ${activeTicker === 'MARKET' ? 'active' : ''}`}
              onClick={() => setActiveTicker('MARKET')}
            >
              MARKET PREDICTOR
            </button>
          </ul>

          <div className="predictor-search">
            {activeTicker === 'MARKET' ? (
              <>
                <input className="input-box" placeholder="Enter ticker e.g. TSLA" />
                <button className="search-button">Predict</button>
              </>
            ) : (
              <button className="search-button search-button--wide" onClick={handlePredict}>Predict</button>
            )}
          </div>
          <div className="prediction-display">
            {activeTicker === 'AAPL' && <AAPL ref={aaplRef} />}
          </div>
        </div>
        <MarketContext />
        <h6>For educational purposes only. Not financial advice.</h6>

      </section>

      <div className="ticks"></div>
    </>
  )
}

export default App