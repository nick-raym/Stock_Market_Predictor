from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # allows your React app to call this API

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("model.pkl not found. Run predictor.py first.")

model = joblib.load(MODEL_PATH)
print("Model loaded.")

FEATURES = [
    "SMA_10",
    "SMA_50",
    "EMA_10",
    "RSI",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "Lag_1",
    "Lag_5",
]

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    df["RSI"] = compute_rsi(df["Close"])

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(window=20).std()

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_5"] = df["Close"].shift(5)

    return df


# ── ROUTES ────────────────────────────────────────────────────────────────────


@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "").upper().strip()

    if not ticker:
        return jsonify({"error": "Missing 'ticker' query parameter"}), 400

    # Fetch last 90 days (enough for all rolling windows)
    raw = yf.download(
        ticker, period="90d", interval="1d", auto_adjust=True, progress=False
    )

    if raw.empty:
        return (
            jsonify(
                {"error": f"No data found for ticker '{ticker}'. Check the symbol."}
            ),
            404,
        )

    df = build_features(raw)
    df = df.dropna()

    if df.empty or len(df) < 1:
        return (
            jsonify({"error": f"Not enough data to compute features for '{ticker}'."}),
            422,
        )

    latest = df[FEATURES].iloc[-1:]
    latest_close = float(df["Close"].iloc[-1])
    latest_date = str(df.index[-1].date())

    prediction = int(model.predict(latest)[0])
    probabilities = model.predict_proba(latest)[0]
    confidence = float(probabilities[prediction])

    return jsonify(
        {
            "ticker": ticker,
            "date": latest_date,
            "current_price": round(latest_close, 2),
            "prediction": prediction,  # 1 = UP, 0 = DOWN
            "direction": "UP" if prediction == 1 else "DOWN",
            "confidence": round(confidence * 100, 1),  # e.g. 63.4
            "prob_up": round(float(probabilities[1]) * 100, 1),
            "prob_down": round(float(probabilities[0]) * 100, 1),
            "features": {k: round(float(v), 4) for k, v in latest.iloc[0].items()},
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier"})


@app.route("/market-context", methods=["GET"])
def market_context():
    try:
        spy = yf.download(
            "SPY", period="5d", interval="1d", auto_adjust=True, progress=False
        )
        xlk = yf.download(
            "XLK", period="5d", interval="1d", auto_adjust=True, progress=False
        )
        vix = yf.download("^VIX", period="5d", interval="1d", progress=False)

        def flatten(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()

        spy = flatten(spy)
        xlk = flatten(xlk)
        vix = flatten(vix)

        spy_price = float(spy["Close"].iloc[-1])
        spy_prev = float(spy["Close"].iloc[-2])
        spy_return = (spy_price - spy_prev) / spy_prev

        xlk_price = float(xlk["Close"].iloc[-1])
        xlk_prev = float(xlk["Close"].iloc[-2])
        xlk_return = (xlk_price - xlk_prev) / xlk_prev

        vix_level = float(vix["Close"].iloc[-1])
        vix_prev = float(vix["Close"].iloc[-2])
        vix_change = vix_level - vix_prev

        # SPY vs its 20-day average — tells us if we're in a bullish regime
        spy_long = yf.download(
            "SPY", period="60d", interval="1d", auto_adjust=True, progress=False
        )
        spy_long = flatten(spy_long)
        spy_sma20 = float(spy_long["Close"].rolling(20).mean().iloc[-1])
        spy_vs_sma20 = spy_price / spy_sma20

        return jsonify(
            {
                "spy_price": round(spy_price, 2),
                "spy_return": round(spy_return, 6),
                "xlk_price": round(xlk_price, 2),
                "xlk_return": round(xlk_return, 6),
                "vix_level": round(vix_level, 2),
                "vix_change": round(vix_change, 2),
                "spy_vs_sma20": round(spy_vs_sma20, 4),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
