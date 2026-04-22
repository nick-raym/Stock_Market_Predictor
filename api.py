from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────

AAPL_MODEL_PATH = "model_aapl.pkl"
GOOGL_MODEL_PATH = "model_googl.pkl"

if not os.path.exists(AAPL_MODEL_PATH):
    raise RuntimeError(f"{AAPL_MODEL_PATH} not found. Run predictor.py first.")
if not os.path.exists(GOOGL_MODEL_PATH):
    raise RuntimeError(f"{GOOGL_MODEL_PATH} not found. Run googlePredictor.py first.")

aapl_model = joblib.load(AAPL_MODEL_PATH)
googl_model = joblib.load(GOOGL_MODEL_PATH)
print("Models loaded: AAPL, GOOGL")

# ── AAPL FEATURES (matches predictor.py) ─────────────────────────────────────

AAPL_FEATURES = [
    'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',
    'RSI',
    'Return_1', 'Return_5', 'Return_10', 'Return_20',
    'Momentum_3', 'Momentum_10', 'Momentum_20',
    'Volatility_10', 'Volatility_20', 'Volatility_Ratio',
    'Volume_Change', 'Volume_SMA_Ratio',
    'Trend_Strength',
    'Market_Return', 'Market_Return_5', 'Market_vs_SMA20',
    'Sector_Return', 'Sector_vs_Market',
    'VIX_Level', 'VIX_Change', 'VIX_vs_SMA20',
    'Days_to_Earnings', 'Days_since_Earnings',
]

# ── GOOGL FEATURES (matches googlePredictor.py) ─────────────────────────────

GOOGL_FEATURES = [
    # Price / trend
    'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',

    # Momentum — short, medium, long
    'RSI',
    'Return_1', 'Return_5', 'Return_10', 'Return_20',
    'Momentum_3', 'Momentum_10', 'Momentum_20',

    # Volatility
    'Volatility_10', 'Volatility_20', 'Volatility_Ratio',

    # Volume
    'Volume_Change', 'Volume_SMA_Ratio', 'Trend_Strength',

    # Broad market
    'Market_Return', 'Market_Return_5', 'Market_vs_SMA20',

    # Dual sector (GOOGL lives in both XLK and XLC)
    'Sector_XLK_Return', 'Sector_XLC_Return',
    'XLK_vs_Market', 'XLC_vs_Market', 'XLC_vs_XLK',

    # Ad revenue proxy (META only)
    'Meta_Return', 'Meta_Return_5',

    # AI competition proxies
    'MSFT_Return', 'MSFT_Return_5',
    'NVDA_Return', 'NVDA_Return_5',

    # GOOGL vs competitors — relative strength
    'GOOGL_vs_MSFT',
    'GOOGL_vs_XLC',
    'GOOGL_vs_XLK',

    # Macro / fear
    'VIX_Level', 'VIX_Change', 'VIX_vs_SMA20',

    # Earnings proximity
    'Days_to_Earnings', 'Days_since_Earnings',
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def download(ticker, period="120d"):
    raw = yf.download(ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.dropna()


def compute_rsi(series, window=14):
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def scalar_value(value):
    if isinstance(value, pd.DataFrame) and value.size == 1:
        return value.iloc[0, 0]
    if isinstance(value, pd.Series) and value.size == 1:
        return value.iloc[0]
    return value


def get_earnings_features(ticker, index):
    """Return (days_to_earnings, days_since_earnings) aligned to index."""
    try:
        t   = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is None or cal.empty:
            raise ValueError("No earnings data")
        earn_dates = sorted(
            pd.DatetimeIndex(cal.index.normalize()).tz_localize(None).unique()
        )
        days_to, days_since = [], []
        for dt in index:
            future = [e for e in earn_dates if e >= dt]
            past   = [e for e in earn_dates if e < dt]
            days_to.append(min((future[0] - dt).days if future else 90, 90))
            days_since.append(min((dt - past[-1]).days if past else 90, 90))
        return (
            pd.Series(days_to,    index=index, name='Days_to_Earnings'),
            pd.Series(days_since, index=index, name='Days_since_Earnings'),
        )
    except Exception:
        neutral = pd.Series(45, index=index)
        return neutral.rename('Days_to_Earnings'), neutral.rename('Days_since_Earnings')


def safe_reindex(series, index):
    """Reindex and fill NaN with 0 — safe for tickers with partial history."""
    return series.reindex(index).fillna(0)


def build_features_aapl(ticker_df, spy_df, xlk_df, vix_df, ticker):
    """Build features for AAPL (matches predictor.py)."""
    df = ticker_df.copy()

    # ── price features ────────────────────────────────────────────────────────
    df['SMA_10']         = df['Close'].rolling(10).mean()
    df['SMA_50']         = df['Close'].rolling(50).mean()
    df['SMA_Ratio']      = df['SMA_10'] / df['SMA_50']
    df['Price_vs_SMA10'] = df['Close'] / df['SMA_10']
    df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']

    df['RSI']         = compute_rsi(df['Close'])
    df['Return_1']    = df['Close'].pct_change(1)
    df['Return_5']    = df['Close'].pct_change(5)
    df['Return_10']   = df['Close'].pct_change(10)
    df['Return_20']   = df['Close'].pct_change(20)
    df['Momentum_3']  = df['Close'].pct_change(3)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)

    daily_ret              = df['Close'].pct_change()
    df['Volatility_10']    = daily_ret.rolling(10).std()
    df['Volatility_20']    = daily_ret.rolling(20).std()
    df['Volatility_Ratio'] = df['Volatility_10'] / (df['Volatility_20'] + 1e-9)

    df['Volume_Change']    = df['Volume'].pct_change()
    df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
    df['Trend_Strength']   = abs(df['Return_5']) / (df['Volatility_10'] + 1e-9)

    # ── market / sector / VIX ─────────────────────────────────────────────────
    spy_ret      = spy_df['Close'].pct_change()
    spy_ret5     = spy_df['Close'].pct_change(5)
    spy_vs_sma20 = spy_df['Close'] / spy_df['Close'].rolling(20).mean()
    xlk_ret      = xlk_df['Close'].pct_change()
    vix          = vix_df['Close']
    vix_chg      = vix.pct_change()
    vix_vs_sma20 = vix / vix.rolling(20).mean()

    df['Market_Return']    = spy_ret.reindex(df.index)
    df['Market_Return_5']  = spy_ret5.reindex(df.index)
    df['Market_vs_SMA20']  = spy_vs_sma20.reindex(df.index)
    df['Sector_Return']    = xlk_ret.reindex(df.index)
    df['Sector_vs_Market'] = df['Sector_Return'] - df['Market_Return']
    df['VIX_Level']        = vix.reindex(df.index)
    df['VIX_Change']       = vix_chg.reindex(df.index)
    df['VIX_vs_SMA20']     = vix_vs_sma20.reindex(df.index)

    # ── earnings proximity ────────────────────────────────────────────────────
    days_to, days_since = get_earnings_features(ticker, df.index)
    df['Days_to_Earnings']    = days_to.reindex(df.index)
    df['Days_since_Earnings'] = days_since.reindex(df.index)

    return df.dropna(subset=AAPL_FEATURES)


def build_features_googl(ticker_df, spy_df, xlk_df, xlc_df, meta_df, msft_df, nvda_df, vix_df, ticker):
    """Build features for GOOGL (matches googlePredictor.py)."""
    df = ticker_df.copy()

    # ── price features ────────────────────────────────────────────────────────
    df['SMA_10']         = df['Close'].rolling(10).mean()
    df['SMA_50']         = df['Close'].rolling(50).mean()
    df['SMA_Ratio']      = df['SMA_10'] / df['SMA_50']
    df['Price_vs_SMA10'] = df['Close'] / df['SMA_10']
    df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']

    df['RSI']         = compute_rsi(df['Close'])
    df['Return_1']    = df['Close'].pct_change(1)
    df['Return_5']    = df['Close'].pct_change(5)
    df['Return_10']   = df['Close'].pct_change(10)
    df['Return_20']   = df['Close'].pct_change(20)

    df['Momentum_3']  = df['Close'].pct_change(3)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)

    daily_ret              = df['Close'].pct_change()
    df['Volatility_10']    = daily_ret.rolling(10).std()
    df['Volatility_20']    = daily_ret.rolling(20).std()
    df['Volatility_Ratio'] = df['Volatility_10'] / (df['Volatility_20'] + 1e-9)

    df['Volume_Change']    = df['Volume'].pct_change()
    df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
    df['Trend_Strength']   = abs(df['Return_5']) / (df['Volatility_10'] + 1e-9)

    # ── market ────────────────────────────────────────────────────────────────
    spy_ret      = spy_df['Close'].pct_change()
    spy_ret5     = spy_df['Close'].pct_change(5)
    spy_vs_sma20 = spy_df['Close'] / spy_df['Close'].rolling(20).mean()

    df['Market_Return']   = safe_reindex(spy_ret, df.index)
    df['Market_Return_5'] = safe_reindex(spy_ret5, df.index)
    df['Market_vs_SMA20'] = safe_reindex(spy_vs_sma20, df.index)

    # ── Sector (XLK + XLC) ────────────────────────────────────────────────────
    xlk_ret = xlk_df['Close'].pct_change()
    xlc_ret = xlc_df['Close'].pct_change()

    df['Sector_XLK_Return'] = safe_reindex(xlk_ret, df.index)
    df['Sector_XLC_Return'] = safe_reindex(xlc_ret, df.index)

    df['XLK_vs_Market']     = df['Sector_XLK_Return'] - df['Market_Return']
    df['XLC_vs_Market']     = df['Sector_XLC_Return'] - df['Market_Return']
    df['XLC_vs_XLK']        = df['Sector_XLC_Return'] - df['Sector_XLK_Return']

    # ── Ad revenue proxy (META) ────────────────────────────────────────────────
    meta_ret   = meta_df['Close'].pct_change()
    meta_ret5  = meta_df['Close'].pct_change(5)

    df['Meta_Return']   = safe_reindex(meta_ret, df.index)
    df['Meta_Return_5'] = safe_reindex(meta_ret5, df.index)

    # ── AI competition (MSFT, NVDA) ───────────────────────────────────────────
    msft_ret   = msft_df['Close'].pct_change()
    msft_ret5  = msft_df['Close'].pct_change(5)
    nvda_ret   = nvda_df['Close'].pct_change()
    nvda_ret5  = nvda_df['Close'].pct_change(5)

    df['MSFT_Return']   = safe_reindex(msft_ret, df.index)
    df['MSFT_Return_5'] = safe_reindex(msft_ret5, df.index)
    df['NVDA_Return']   = safe_reindex(nvda_ret, df.index)
    df['NVDA_Return_5'] = safe_reindex(nvda_ret5, df.index)

    # ── GOOGL relative strength vs competitors ────────────────────────────────
    df['GOOGL_vs_MSFT'] = df['Return_5'] - safe_reindex(msft_ret5, df.index)
    df['GOOGL_vs_XLC']  = df['Return_5'] - safe_reindex(xlc_ret.rolling(5).sum(), df.index)
    df['GOOGL_vs_XLK']  = df['Return_5'] - safe_reindex(xlk_ret.rolling(5).sum(), df.index)

    # ── VIX ───────────────────────────────────────────────────────────────────
    vix          = vix_df['Close']
    vix_chg      = vix.pct_change()
    vix_vs_sma20 = vix / vix.rolling(20).mean()

    df['VIX_Level']    = safe_reindex(vix, df.index)
    df['VIX_Change']   = safe_reindex(vix_chg, df.index)
    df['VIX_vs_SMA20'] = safe_reindex(vix_vs_sma20, df.index)

    # ── earnings proximity ────────────────────────────────────────────────────
    days_to, days_since = get_earnings_features(ticker, df.index)
    df['Days_to_Earnings']    = days_to.reindex(df.index)
    df['Days_since_Earnings'] = days_since.reindex(df.index)

    return df.dropna(subset=GOOGL_FEATURES)


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "AAPL").upper().strip()

    try:
        # Need 120 days to fill all rolling windows (SMA_50, etc.)
        ticker_df = download(ticker, period="120d")
        spy_df    = download("SPY",  period="120d")
        xlk_df    = download("XLK",  period="120d")
        vix_df    = yf.download("^VIX", period="120d", interval="1d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        if ticker_df.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404

        # Use AAPL features by default (for backward compatibility)
        df = build_features_aapl(ticker_df, spy_df, xlk_df, vix_df, ticker)

        if df.empty:
            return jsonify({"error": "Not enough data to compute features"}), 422

        latest       = df[AAPL_FEATURES].iloc[-1:]
        latest_close = float(scalar_value(df['Close'].iloc[-1]))
        latest_date  = str(df.index[-1].date())

        predicted_return = float(scalar_value(aapl_model.predict(latest)[0]))

        return jsonify({
            "ticker":           ticker,
            "date":             latest_date,
            "current_price":    round(latest_close, 2),
            "predicted_5d_return": round(predicted_return, 6),
            "predicted_5d_pct":    round(predicted_return * 100, 3),
            "direction":        "UP" if predicted_return > 0 else "DOWN",
            "signal_strength":  round(abs(predicted_return) * 100, 3),
            "features":         {k: round(float(scalar_value(v)), 6) for k, v in latest.iloc[0].items()},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/aapl", methods=["GET"])
def aapl_summary():
    """Convenience: run /predict but always for AAPL, no params needed."""
    try:
        aapl_df = download("AAPL", period="120d")
        spy_df  = download("SPY",  period="120d")
        xlk_df  = download("XLK",  period="120d")
        vix_df  = yf.download("^VIX", period="120d", interval="1d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        df = build_features_aapl(aapl_df, spy_df, xlk_df, vix_df, "AAPL")
        latest       = df[AAPL_FEATURES].iloc[-1:]
        latest_close = float(scalar_value(df['Close'].iloc[-1]))
        latest_date  = str(df.index[-1].date())
        predicted_return = float(scalar_value(aapl_model.predict(latest)[0]))

        return jsonify({
            "ticker":              "AAPL",
            "date":                latest_date,
            "current_price":       round(latest_close, 2),
            "predicted_5d_return": round(predicted_return, 6),
            "predicted_5d_pct":    round(predicted_return * 100, 3),
            "direction":           "UP" if predicted_return > 0 else "DOWN",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/googl", methods=["GET"])
def googl_summary():
    """GOOGL prediction using GOOGL-specific features and model."""
    try:
        googl_df = download("GOOGL", period="120d")
        spy_df   = download("SPY",   period="120d")
        xlk_df   = download("XLK",   period="120d")
        xlc_df   = download("XLC",   period="120d")
        meta_df  = download("META",  period="120d")
        msft_df  = download("MSFT",  period="120d")
        nvda_df  = download("NVDA",  period="120d")
        vix_df   = yf.download("^VIX", period="120d", interval="1d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        df = build_features_googl(googl_df, spy_df, xlk_df, xlc_df, meta_df, msft_df, nvda_df, vix_df, "GOOGL")
        latest       = df[GOOGL_FEATURES].iloc[-1:]
        latest_close = float(scalar_value(df['Close'].iloc[-1]))
        latest_date  = str(df.index[-1].date())
        predicted_return = float(scalar_value(googl_model.predict(latest)[0]))

        return jsonify({
            "ticker":              "GOOGL",
            "date":                latest_date,
            "current_price":       round(latest_close, 2),
            "predicted_5d_return": round(predicted_return, 6),
            "predicted_5d_pct":    round(predicted_return * 100, 3),
            "direction":           "UP" if predicted_return > 0 else "DOWN",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/market-context", methods=["GET"])
def market_context():
    try:
        spy = download("SPY",  period="60d")
        xlk = download("XLK",  period="5d")
        vix = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix.dropna()

        if len(spy) < 2 or len(xlk) < 2 or len(vix) < 2:
            return jsonify({"error": "Not enough market context data."}), 422

        spy_price    = float(scalar_value(spy['Close'].iloc[-1]))
        spy_prev     = float(scalar_value(spy['Close'].iloc[-2]))
        spy_return   = (spy_price - spy_prev) / spy_prev
        spy_sma20    = float(scalar_value(spy['Close'].rolling(20).mean().iloc[-1]))
        spy_vs_sma20 = spy_price / spy_sma20

        xlk_price  = float(scalar_value(xlk['Close'].iloc[-1]))
        xlk_prev   = float(scalar_value(xlk['Close'].iloc[-2]))
        xlk_return = (xlk_price - xlk_prev) / xlk_prev

        vix_level  = float(scalar_value(vix['Close'].iloc[-1]))
        vix_prev   = float(scalar_value(vix['Close'].iloc[-2]))
        vix_change = vix_level - vix_prev

        return jsonify({
            "spy_price":    round(spy_price, 2),
            "spy_return":   round(spy_return, 6),
            "spy_vs_sma20": round(spy_vs_sma20, 4),
            "xlk_price":    round(xlk_price, 2),
            "xlk_return":   round(xlk_return, 6),
            "vix_level":    round(vix_level, 2),
            "vix_change":   round(vix_change, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "AAPL": {"path": AAPL_MODEL_PATH, "target": "5d_forward_return"},
            "GOOGL": {"path": GOOGL_MODEL_PATH, "target": "5d_forward_return"},
        }
    })


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
