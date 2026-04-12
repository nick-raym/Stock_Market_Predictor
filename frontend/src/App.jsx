import './App.css'
import React, { useState } from 'react';

function App() {
  const [activeTicker, setActiveTicker] = useState('AAPL');

  return (
    <>
      <section id="center">
        <h1 className="text-predictor">Stock Market Predictor</h1>

        <div>
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
            <input className="input-box" placeholder="Enter ticker e.g. AAPL" />
            <button className="search-button">Predict</button>
          </div>
        </div>
      </section>

      <div className="ticks"></div>
    </>
  )
}

export default App