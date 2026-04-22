import './App.css'
import React, { useState, useRef } from 'react';
import MarketContext from './MarketContext';
import AAPL from './AAPL';
import GOOGL from './GOOGL';
import Market from './Market';

function App() {
  const [activeTicker, setActiveTicker] = useState('');
  const aaplRef = useRef();
  const googlRef = useRef();

  const handlePredict = () => {
    if (activeTicker === 'AAPL' && aaplRef.current) {
      aaplRef.current.fetchData();
    } else if (activeTicker === 'GOOGL' && googlRef.current) {
      googlRef.current.fetchData();
    }
  }

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
                <Market />
              ) : (
                <button className="search-button search-button--wide" onClick={handlePredict}>Predict</button>
              )}
            </div>
            <div className="prediction-display">
              {activeTicker === 'AAPL' && <AAPL ref={aaplRef} />}
              {activeTicker === 'GOOGL' && <GOOGL ref={googlRef} />}
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