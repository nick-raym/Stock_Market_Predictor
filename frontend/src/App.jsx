
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

import React, { useEffect, useState } from 'react';
// import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
// import Navbar from './Navbar';

function App() {
  // const [count, setCount] = useState(0)

  return (
    <>
      <section id="center">
        <div className="hero">
          <img src={heroImg} className="base" width="170" height="179" alt="" />
          <img src={reactLogo} className="framework" alt="React logo" />
          <img src={viteLogo} className="vite" alt="Vite logo" />
        </div>
 
        <div>
          <h1>Stock Market Predictor
          </h1>
        </div>

        <div>
          <h3>Choose a stock to predict if the price will go up or down tommorow</h3>
          <input class='input-box'></input>
          <br></br>
          <button class='search-button'>Predict</button>
        </div>
      </section>
      <div className="ticks"></div>
    </>
  )
}

export default App
