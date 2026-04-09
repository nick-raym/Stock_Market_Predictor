import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2010-01-01"
MODEL_OUTPUT = "model.pkl"

FEATURES = [
    'SMA_Ratio', 'Price_vs_SMA10',
    'RSI',
    'Return_1', 'Return_5',
    'Volume_Change', 'Volume_SMA_Ratio',
    'Volatility_10', 'Volatility_20',
    'Momentum_3', 'Momentum_10',
    'Market_Return',
    'Trend_Strength'
]

# ── MARKET DATA (SPY as benchmark) ────────────────────────────────────────────

print("Downloading SPY benchmark...")
spy_raw = yf.download("SPY", start=START_DATE, auto_adjust=True, progress=False)

if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_raw.columns = spy_raw.columns.get_level_values(0)

# Store as a plain Series indexed by date for clean merging
spy_returns = spy_raw['Close'].pct_change().rename('Market_Return')

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

    # Price-based features (ratios, not raw prices — generalise across stocks)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_Ratio'] = df['SMA_10'] / df['SMA_50']
    df['Price_vs_SMA10'] = df['Close'] / df['SMA_10']

    # Momentum
    df['RSI'] = compute_rsi(df['Close'])
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Momentum_3'] = df['Close'].pct_change(3)
    df['Momentum_10'] = df['Close'].pct_change(10)

    # Volume (ratio vs rolling mean — avoids raw volume scale differences)
    df['Volume_Change'] = df['Volume'].pct_change()
    vol_sma = df['Volume'].rolling(10).mean()
    df['Volume_SMA_Ratio'] = df['Volume'] / vol_sma  # FIX: was storing raw SMA

    # Volatility
    daily_returns = df['Close'].pct_change()
    df['Volatility_10'] = daily_returns.rolling(10).std()
    df['Volatility_20'] = daily_returns.rolling(20).std()

    # Trend strength: how large is the move relative to recent noise
    df['Trend_Strength'] = abs(df['Return_5']) / (df['Volatility_10'] + 1e-9)  # FIX: avoid div/0

    # Market context: merge SPY returns by date
    df['Market_Return'] = spy_returns.reindex(df.index)  # FIX: was direct assignment (misaligns on gaps)

    return df


# ── COLLECT TRAINING DATA ─────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1: Collecting training data (2010 → today)")
print("=" * 60)

all_dfs = []
for ticker in TICKERS:
    print(f"  Downloading {ticker}...")
    raw = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  WARNING: No data for {ticker}, skipping.")
        continue

    df = build_features(raw)

    # TARGET: 1 if next day return > 0 (simple direction)
    # Removed the 0.5% threshold — it creates class imbalance and
    # makes the model predict "flat" days as DOWN, hurting accuracy
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Store next-day return for simulation (before dropping NaNs)
    df['Next_Return'] = df['Close'].pct_change().shift(-1)

    df['Ticker'] = ticker
    df = df.dropna()

    print(f"  {ticker}: {len(df)} rows")
    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError("No data collected.")

combined = pd.concat(all_dfs).sort_index().reset_index(drop=True)
print(f"\nTotal training rows: {len(combined)} across {len(all_dfs)} tickers")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2: Training model")
print("=" * 60)

X = combined[FEATURES]
y = combined['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"  Target balance (test): {y_test.mean():.2%} UP days")

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print("  Training complete.")

# ── EVALUATE ──────────────────────────────────────────────────────────────────

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.39).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Probability distribution — tells you if model is well-calibrated
print("\nProbability distribution:")
print(pd.Series(y_prob).describe())

# Calibration check: does high confidence = higher accuracy?
bins = pd.cut(y_prob, bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
print("\nAccuracy by confidence bucket:")
print(pd.crosstab(bins, y_test, normalize='index').round(3))

importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# ── TRADING SIMULATION ────────────────────────────────────────────────────────
# FIX: Use Next_Return (pre-computed, aligned) instead of recalculating
# pct_change().shift(-1) on a sliced dataframe, which caused misalignment

print("\n" + "=" * 60)
print("PHASE 3: Trading Simulation")
print("=" * 60)

test_df = combined.iloc[len(X_train):len(X_train) + len(y_prob)].copy()
test_df['Prob'] = y_prob
test_df['Pred'] = y_pred
test_df = test_df.dropna(subset=['Next_Return']).reset_index(drop=True)

# Realign after dropna
aligned_preds = test_df['Pred'].values
aligned_probs = test_df['Prob'].values
returns = test_df['Next_Return'].values  # real next-day returns, properly aligned

# Buy-and-hold benchmark
bnh_equity = (1 + returns).cumprod()

# Basic strategy: buy when model predicts UP
strategy_returns = np.where(aligned_preds == 1, returns, 0)
strategy_equity = (1 + strategy_returns).cumprod()

print(f"\nBuy & Hold Final Return:      {bnh_equity[-1]:.4f}x")
print(f"Basic Strategy Final Return:  {strategy_equity[-1]:.4f}x")
print(f"Basic Win Rate:               {(strategy_returns > 0).mean():.2%}")
print(f"Trades taken:                 {(aligned_preds == 1).sum()} / {len(aligned_preds)}")

wins = strategy_returns[strategy_returns > 0]
losses = strategy_returns[strategy_returns < 0]
print(f"Avg Win:                      {wins.mean():.4f}")
print(f"Avg Loss:                     {losses.mean():.4f}")
print(f"Win/Loss Ratio:               {abs(wins.mean() / losses.mean()):.2f}")

# Threshold sweep
print("\nThreshold sweep:")
print(f"{'Threshold':<12} {'Final Return':<16} {'Win Rate':<12} {'Trades'}")
for t in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
    tp = (aligned_probs > t)
    tr = np.where(tp, returns, 0)
    eq = (1 + tr).cumprod()
    print(f"{t:<12} {eq[-1]:<16.4f} {(tr > 0).mean():<12.2%} {tp.sum()}")

# ── EXPORT MODEL ──────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_OUTPUT)
print(f"\nModel saved → {MODEL_OUTPUT}")