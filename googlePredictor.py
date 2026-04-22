import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER       = "GOOGL"
START_DATE   = "2010-01-01"
MODEL_OUTPUT = "model_googl.pkl"
FORWARD_DAYS = 5   # Shorter horizon for better accuracy

FEATURES = [
    'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',

    'RSI',
    'Return_1', 'Return_5', 'Return_10', 'Return_20',
    'Momentum_3', 'Momentum_10', 'Momentum_20',

    'Volatility_10', 'Volatility_20', 'Volatility_Ratio',

    'Volume_Change', 'Volume_SMA_Ratio', 'Trend_Strength',

    'Market_Return', 'Market_Return_5', 'Market_vs_SMA20',

    'Sector_XLK_Return', 'Sector_XLC_Return',
    'XLK_vs_Market', 'XLC_vs_Market', 'XLC_vs_XLK',

    'Meta_Return', 'Meta_Return_5',

    'MSFT_Return', 'MSFT_Return_5',
    'NVDA_Return', 'NVDA_Return_5',

    'GOOGL_vs_MSFT',   # when GOOGL underperforms MSFT, AI narrative hurting it
    'GOOGL_vs_XLC',
    'GOOGL_vs_XLK',
    'VIX_Level', 'VIX_Change', 'VIX_vs_SMA20',
    'Days_to_Earnings', 'Days_since_Earnings',
]

# ── DOWNLOAD EXTERNAL DATA ────────────────────────────────────────────────────

print("Downloading external data...")

def download(ticker, start):
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw

spy_raw  = download("SPY",  START_DATE)
xlk_raw  = download("XLK",  START_DATE)
xlc_raw  = download("XLC",  START_DATE)
meta_raw = download("META", START_DATE)
msft_raw = download("MSFT", START_DATE)
nvda_raw = download("NVDA", START_DATE)
# TTD REMOVED — IPO'd 2016, was silently cutting 8 years of GOOGL training data

vix_raw = yf.download("^VIX", start=START_DATE, progress=False)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

spy_close  = spy_raw['Close']
xlk_close  = xlk_raw['Close']
xlc_close  = xlc_raw['Close']
meta_close = meta_raw['Close']
msft_close = msft_raw['Close']
nvda_close = nvda_raw['Close']
vix_close  = vix_raw['Close']

spy_returns  = spy_close.pct_change()
spy_ret5     = spy_close.pct_change(5)
spy_vs_sma20 = spy_close / spy_close.rolling(20).mean()
xlk_returns  = xlk_close.pct_change()
xlc_returns  = xlc_close.pct_change()
meta_returns = meta_close.pct_change()
meta_ret5    = meta_close.pct_change(5)
msft_returns = msft_close.pct_change()
msft_ret5    = msft_close.pct_change(5)
nvda_returns = nvda_close.pct_change()
nvda_ret5    = nvda_close.pct_change(5)
vix_level    = vix_close
vix_change   = vix_close.pct_change()
vix_vs_sma20 = vix_close / vix_close.rolling(20).mean()

print("  SPY, XLK, XLC, META, MSFT, NVDA, VIX downloaded.")

# ── EARNINGS DATES ────────────────────────────────────────────────────────────

def get_earnings_features(ticker, index):
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
    except Exception as e:
        print(f"  WARNING: Could not fetch earnings dates ({e}). Using neutral values.")
        neutral = pd.Series(45, index=index)
        return neutral.rename('Days_to_Earnings'), neutral.rename('Days_since_Earnings')

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def compute_rsi(data, window=14):
    delta    = data.diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def safe_reindex(series, index):
    """Reindex and fill NaN with 0 — safe for tickers with partial history."""
    return series.reindex(index).fillna(0)


def build_features(df, days_to_earn, days_since_earn):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    # Price / trend
    df['SMA_10']         = df['Close'].rolling(10).mean()
    df['SMA_50']         = df['Close'].rolling(50).mean()
    df['SMA_Ratio']      = df['SMA_10'] / df['SMA_50']
    df['Price_vs_SMA10'] = df['Close'] / df['SMA_10']
    df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']

    # Momentum
    df['RSI']         = compute_rsi(df['Close'])
    df['Return_1']    = df['Close'].pct_change(1)
    df['Return_5']    = df['Close'].pct_change(5)
    df['Return_10']   = df['Close'].pct_change(10)
    df['Return_20']   = df['Close'].pct_change(20)
    df['Momentum_3']  = df['Close'].pct_change(3)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)

    # Volatility
    daily_ret              = df['Close'].pct_change()
    df['Volatility_10']    = daily_ret.rolling(10).std()
    df['Volatility_20']    = daily_ret.rolling(20).std()
    df['Volatility_Ratio'] = df['Volatility_10'] / (df['Volatility_20'] + 1e-9)

    # Volume
    df['Volume_Change']    = df['Volume'].pct_change()
    df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
    df['Trend_Strength']   = abs(df['Return_5']) / (df['Volatility_10'] + 1e-9)

    # Broad market
    df['Market_Return']   = safe_reindex(spy_returns,  df.index)
    df['Market_Return_5'] = safe_reindex(spy_ret5,     df.index)
    df['Market_vs_SMA20'] = safe_reindex(spy_vs_sma20, df.index)

    # Sector
    df['Sector_XLK_Return'] = safe_reindex(xlk_returns, df.index)
    df['Sector_XLC_Return'] = safe_reindex(xlc_returns, df.index)
    df['XLK_vs_Market']     = df['Sector_XLK_Return'] - df['Market_Return']
    df['XLC_vs_Market']     = df['Sector_XLC_Return'] - df['Market_Return']
    df['XLC_vs_XLK']        = df['Sector_XLC_Return'] - df['Sector_XLK_Return']

    # Ad revenue proxy
    df['Meta_Return']   = safe_reindex(meta_returns, df.index)
    df['Meta_Return_5'] = safe_reindex(meta_ret5,    df.index)

    # AI competition
    df['MSFT_Return']   = safe_reindex(msft_returns, df.index)
    df['MSFT_Return_5'] = safe_reindex(msft_ret5,    df.index)
    df['NVDA_Return']   = safe_reindex(nvda_returns, df.index)
    df['NVDA_Return_5'] = safe_reindex(nvda_ret5,    df.index)

    # GOOGL relative strength vs competitors and sectors
    msft_ret5_idx = safe_reindex(msft_ret5, df.index)
    df['GOOGL_vs_MSFT'] = df['Return_5'] - msft_ret5_idx
    df['GOOGL_vs_XLC']  = df['Return_5'] - df['Sector_XLC_Return'].rolling(5).sum()
    df['GOOGL_vs_XLK']  = df['Return_5'] - df['Sector_XLK_Return'].rolling(5).sum()

    # VIX
    df['VIX_Level']    = safe_reindex(vix_level,    df.index)
    df['VIX_Change']   = safe_reindex(vix_change,   df.index)
    df['VIX_vs_SMA20'] = safe_reindex(vix_vs_sma20, df.index)

    # Earnings
    df['Days_to_Earnings']    = days_to_earn.reindex(df.index)
    df['Days_since_Earnings'] = days_since_earn.reindex(df.index)

    return df

# ── DOWNLOAD GOOGL & BUILD FEATURES ──────────────────────────────────────────

print("=" * 60)
print(f"PHASE 1: Downloading {TICKER} (2010 → today)")
print("=" * 60)

raw = download(TICKER, START_DATE)
if raw.empty:
    raise RuntimeError(f"No data returned for {TICKER}.")

print(f"  Raw data starts: {raw.index[0].date()}")

print(f"  Fetching {TICKER} earnings dates...")
days_to_earn, days_since_earn = get_earnings_features(TICKER, raw.index)

df = build_features(raw, days_to_earn, days_since_earn)
df['Next_Return_1d'] = df['Close'].pct_change().shift(-1)
df[f'Next_Return_{FORWARD_DAYS}d'] = df['Close'].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
df = df.dropna()

print(f"  After features data starts: {df.index[0].date()}  ← should be close to raw start")
print(f"  {TICKER}: {len(df)} rows")
print(f"  1d  return — Mean: {df['Next_Return_1d'].mean():.4%}  Std: {df['Next_Return_1d'].std():.4%}  % UP: {(df['Next_Return_1d'] > 0).mean():.2%}")
print(f"  {FORWARD_DAYS}d return — Mean: {df[f'Next_Return_{FORWARD_DAYS}d'].mean():.4%}  Std: {df[f'Next_Return_{FORWARD_DAYS}d'].std():.4%}  % UP: {(df[f'Next_Return_{FORWARD_DAYS}d'] > 0).mean():.2%}")

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"PHASE 2: Training XGBoost Regressor (target = {FORWARD_DAYS}-day return)")
print("=" * 60)

X = df[FEATURES]
y = df[f'Next_Return_{FORWARD_DAYS}d']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train: {len(X_train)} rows ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"  Test:  {len(X_test)} rows ({X_test.index[0].date()} → {X_test.index[-1].date()})")

# ── TRAIN ─────────────────────────────────────────────────────────────────────

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.1,  # Higher learning rate
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,  # Lower
    gamma=0.0,
    reg_alpha=0.0,       # No L1
    reg_lambda=0.1,      # Lower L2
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("  Training complete.")

# ── EVALUATE ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
actual = y_test.values

mae           = mean_absolute_error(actual, y_pred)
r2            = r2_score(actual, y_pred)
pearson_r,  p_pearson  = pearsonr(actual, y_pred)
spearman_r, p_spearman = spearmanr(actual, y_pred)
direction_acc = (np.sign(y_pred) == np.sign(actual)).mean()

print(f"\n  MAE:                {mae:.6f}")
print(f"  R²:                 {r2:.6f}  (>0 = better than mean)")
print(f"  Pearson r:          {pearson_r:.4f}  (p={p_pearson:.4f})")
print(f"  Spearman r:         {spearman_r:.4f}  (p={p_spearman:.4f})")
print(f"  Direction accuracy: {direction_acc:.2%}")

print(f"\n  Prediction distribution:")
print(pd.Series(y_pred).describe())

pred_series   = pd.Series(y_pred, index=X_test.index, name='pred')
actual_series = pd.Series(actual, index=X_test.index, name='actual')

try:
    buckets   = pd.qcut(pred_series, q=5, duplicates='drop')
    bucket_df = pd.DataFrame({'pred': pred_series, 'actual': actual_series, 'bucket': buckets})
    print(f"\n  Quintile table (Q1=most bearish → Q5=most bullish):")
    print(f"  {'Bucket':<34} {'Avg Predicted':>15} {'Avg Actual':>12} {'% UP':>8}")
    for name, group in bucket_df.groupby('bucket', observed=True):
        print(f"  {str(name):<34} {group['pred'].mean():>14.4%} {group['actual'].mean():>11.4%} {(group['actual'] > 0).mean():>8.2%}")
except Exception as e:
    print(f"  (Could not compute quintiles: {e})")

importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\n  Feature Importances:")
print(importance_df.to_string(index=False))

# ── TRADING SIMULATION ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"PHASE 3: Trading Simulation ({FORWARD_DAYS}-day signal → daily P&L)")
print("=" * 60)

test_df = df.iloc[len(X_train):len(X_train) + len(y_pred)].copy()
test_df['Pred'] = y_pred
test_df = test_df.dropna(subset=['Next_Return_1d']).reset_index(drop=True)

daily_returns = test_df['Next_Return_1d'].values
pred_nd       = test_df['Pred'].values
n             = len(daily_returns)

def build_position_series(signals_raw, hold=FORWARD_DAYS):
    positions = np.zeros(n)
    counts    = np.zeros(n)
    for i, sig in enumerate(signals_raw):
        end = min(i + hold, n)
        positions[i:end] += sig
        counts[i:end]    += 1
    counts = np.where(counts == 0, 1, counts)
    return positions / counts

COST      = 0.0005
bnh_equity = (1 + daily_returns).cumprod()
print(f"\n  Buy & Hold Final Return: {bnh_equity[-1]:.4f}x")
print(f"\n  {'Strategy':<36} {'Final Return':<16} {'Win Rate':<12} {'Max DD':<10} {'Trades'}")
print("  " + "-" * 82)

best_result = None

for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
    for mode in ['long_only', 'long_short', 'scaled_ls']:

        if mode == 'long_only':
            raw_signals = np.where(pred_nd > threshold, 1.0, 0.0)
            label = f"t={threshold:.3f} long-only"
        elif mode == 'long_short':
            raw_signals = np.where(pred_nd > threshold, 1.0,
                          np.where(pred_nd < -threshold, -1.0, 0.0))
            label = f"t={threshold:.3f} long/short"
        else:
            scaled = np.clip(pred_nd / 0.02, -1.0, 1.0)
            raw_signals = np.where(pred_nd > threshold, scaled,
                          np.where(pred_nd < -threshold, scaled, 0.0))
            label = f"t={threshold:.3f} scaled L/S"

        position      = build_position_series(raw_signals)
        pos_change    = np.abs(np.diff(np.concatenate([[0], position])))
        strat_returns = position * daily_returns - pos_change * COST
        equity        = (1 + strat_returns).cumprod()
        cummax        = np.maximum.accumulate(equity)
        drawdown      = ((equity / cummax) - 1).min()
        wins          = strat_returns[strat_returns > 0]
        losses        = strat_returns[strat_returns < 0]
        trades        = (pos_change > 0.05).sum()

        print(f"  {label:<36} {equity[-1]:<16.4f} {(strat_returns > 0).mean():<12.2%} {drawdown:<10.2%} {trades}")

        if best_result is None or equity[-1] > best_result['equity']:
            best_result = {
                'label': label, 'equity': equity[-1],
                'drawdown': drawdown,
                'win_rate': (strat_returns > 0).mean(),
                'trades': trades,
                'avg_win':  wins.mean()  if len(wins)   else 0,
                'avg_loss': losses.mean() if len(losses) else 0,
            }

print(f"\n  Best: {best_result['label']}")
print(f"    Final Return: {best_result['equity']:.4f}x  (B&H: {bnh_equity[-1]:.4f}x)")
print(f"    Max Drawdown: {best_result['drawdown']:.2%}")
print(f"    Win Rate:     {best_result['win_rate']:.2%}")
print(f"    Trades:       {best_result['trades']}")
if best_result['avg_loss'] != 0:
    print(f"    Win/Loss:     {abs(best_result['avg_win'] / best_result['avg_loss']):.2f}")

# ── EXPORT ────────────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_OUTPUT)
print(f"\nModel saved → {MODEL_OUTPUT}")
print("Done.")