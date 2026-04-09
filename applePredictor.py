import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER = "AAPL"
START_DATE = "2010-01-01"
MODEL_OUTPUT = "model_aapl.pkl"

FEATURES = [
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

# ── DOWNLOAD EXTERNAL DATA ────────────────────────────────────────────────────

print("Downloading external data...")

def download(ticker, start):
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw

spy_raw = download("SPY", START_DATE)
xlk_raw = download("XLK", START_DATE)
vix_raw = yf.download("^VIX", start=START_DATE, progress=False)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

spy_close = spy_raw['Close']
xlk_close = xlk_raw['Close']
vix_close = vix_raw['Close']

spy_returns  = spy_close.pct_change().rename('spy_ret')
spy_ret5     = spy_close.pct_change(5).rename('spy_ret5')
spy_vs_sma20 = (spy_close / spy_close.rolling(20).mean()).rename('spy_vs_sma20')
xlk_returns  = xlk_close.pct_change().rename('xlk_ret')
vix_level    = vix_close.rename('vix')
vix_change   = vix_close.pct_change().rename('vix_chg')
vix_vs_sma20 = (vix_close / vix_close.rolling(20).mean()).rename('vix_vs_sma20')

print("  SPY, XLK, VIX downloaded.")

# ── EARNINGS DATES ────────────────────────────────────────────────────────────

def get_earnings_features(ticker, index):
    try:
        t = yf.Ticker(ticker)
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


def build_features(df, days_to_earn, days_since_earn):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

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

    df['Market_Return']    = spy_returns.reindex(df.index)
    df['Market_Return_5']  = spy_ret5.reindex(df.index)
    df['Market_vs_SMA20']  = spy_vs_sma20.reindex(df.index)
    df['Sector_Return']    = xlk_returns.reindex(df.index)
    df['Sector_vs_Market'] = df['Sector_Return'] - df['Market_Return']
    df['VIX_Level']        = vix_level.reindex(df.index)
    df['VIX_Change']       = vix_change.reindex(df.index)
    df['VIX_vs_SMA20']     = vix_vs_sma20.reindex(df.index)

    df['Days_to_Earnings']    = days_to_earn.reindex(df.index)
    df['Days_since_Earnings'] = days_since_earn.reindex(df.index)

    return df

# ── DOWNLOAD AAPL & BUILD FEATURES ───────────────────────────────────────────

print("=" * 60)
print(f"PHASE 1: Downloading {TICKER} (2010 → today)")
print("=" * 60)

raw = download(TICKER, START_DATE)
if raw.empty:
    raise RuntimeError(f"No data returned for {TICKER}.")

print(f"  Fetching {TICKER} earnings dates...")
days_to_earn, days_since_earn = get_earnings_features(TICKER, raw.index)

df = build_features(raw, days_to_earn, days_since_earn)
df['Next_Return'] = df['Close'].pct_change().shift(-1)
df = df.dropna()

print(f"  {TICKER}: {len(df)} rows")
print(f"  Next_Return — Mean: {df['Next_Return'].mean():.4%}  Std: {df['Next_Return'].std():.4%}  % UP: {(df['Next_Return'] > 0).mean():.2%}")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2: Training XGBoost Regressor")
print("=" * 60)

X = df[FEATURES]
y = df['Next_Return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train: {len(X_train)} rows ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"  Test:  {len(X_test)} rows ({X_test.index[0].date()} → {X_test.index[-1].date()})")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
# KEY FIX: Relaxed regularisation so the model actually splits and learns.
# Previous params (min_child_weight=15, gamma=0.2, reg_alpha=0.1, reg_lambda=1.5)
# were so aggressive that XGBoost refused to make any splits → constant prediction.
# For regression on noisy financial data, let the model find weak signals first,
# then tune regularisation only if you see overfitting (train >> test performance).

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,   # FIX: was 15 — too high, blocked all splits
    gamma=0.0,            # FIX: was 0.2 — removed min-split-gain requirement
    reg_alpha=0.0,        # FIX: was 0.1 — removed L1
    reg_lambda=1.0,       # default L2, mild
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("  Training complete.")

# ── EVALUATE ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
actual = y_test.values

mae        = mean_absolute_error(actual, y_pred)
r2         = r2_score(actual, y_pred)
pearson_r,  p_pearson  = pearsonr(actual, y_pred)
spearman_r, p_spearman = spearmanr(actual, y_pred)
direction_accuracy = (np.sign(y_pred) == np.sign(actual)).mean()

print(f"\n  MAE:                {mae:.6f}")
print(f"  R²:                 {r2:.6f}  (>0 = better than predicting the mean)")
print(f"  Pearson r:          {pearson_r:.4f}  (p={p_pearson:.4f})")
print(f"  Spearman r:         {spearman_r:.4f}  (p={p_spearman:.4f})")
print(f"  Direction accuracy: {direction_accuracy:.2%}")

print(f"\n  Prediction distribution:")
print(pd.Series(y_pred).describe())

# Quintile table — the key diagnostic
# Q1 should have negative actual returns, Q5 positive, if model has signal
pred_series   = pd.Series(y_pred, index=X_test.index, name='pred')
actual_series = pd.Series(actual, index=X_test.index, name='actual')

try:
    buckets = pd.qcut(pred_series, q=5, duplicates='drop')
    bucket_df = pd.DataFrame({'pred': pred_series, 'actual': actual_series, 'bucket': buckets})
    print(f"\n  Predicted return quintiles vs actual returns:")
    print(f"  {'Bucket':<30} {'Avg Predicted':>15} {'Avg Actual':>12} {'% UP Days':>10}")
    for name, group in bucket_df.groupby('bucket', observed=True):
        print(f"  {str(name):<30} {group['pred'].mean():>14.4%} {group['actual'].mean():>11.4%} {(group['actual'] > 0).mean():>10.2%}")
except Exception as e:
    print(f"  (Could not compute quintiles: {e})")

# Feature importances
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\n  Feature Importances:")
print(importance_df.to_string(index=False))

# ── TRADING SIMULATION ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3: Trading Simulation")
print("=" * 60)

test_df = df.iloc[len(X_train):len(X_train) + len(y_pred)].copy()
test_df['Pred_Return'] = y_pred
test_df = test_df.dropna(subset=['Next_Return']).reset_index(drop=True)

actual_returns = test_df['Next_Return'].values
pred_returns   = test_df['Pred_Return'].values

bnh_equity = (1 + actual_returns).cumprod()
print(f"\n  Buy & Hold Final Return: {bnh_equity[-1]:.4f}x")

print(f"\n  {'Strategy':<26} {'Final Return':<16} {'Win Rate':<12} {'Max DD':<10} {'Trades'}")
print("  " + "-" * 75)

best_result = None

for threshold in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
    for sizing in ['fixed', 'scaled']:
        signals = pred_returns > threshold

        if sizing == 'fixed':
            position = np.where(signals, 1.0, 0.0)
        else:
            raw_size = np.clip(pred_returns / 0.01, 0, 1.0)
            position = np.where(signals, raw_size, 0.0)

        label = f">={threshold:.3f} {sizing}"

        COST = 0.0005
        trade_happens = np.diff(np.concatenate([[0], signals.astype(int)])) != 0
        strat_returns = position * actual_returns - (trade_happens * COST)

        equity   = (1 + strat_returns).cumprod()
        cummax   = np.maximum.accumulate(equity)
        drawdown = ((equity / cummax) - 1).min()

        wins   = strat_returns[strat_returns > 0]
        losses = strat_returns[strat_returns < 0]

        print(f"  {label:<26} {equity[-1]:<16.4f} {(strat_returns > 0).mean():<12.2%} {drawdown:<10.2%} {signals.sum()}")

        if best_result is None or equity[-1] > best_result['equity']:
            best_result = {
                'threshold': threshold, 'sizing': sizing,
                'equity': equity[-1], 'drawdown': drawdown,
                'win_rate': (strat_returns > 0).mean(),
                'trades': signals.sum(),
                'avg_win': wins.mean() if len(wins) else 0,
                'avg_loss': losses.mean() if len(losses) else 0,
            }

print(f"\n  Best configuration:")
print(f"    Strategy:    >={best_result['threshold']} {best_result['sizing']}")
print(f"    Final Return:{best_result['equity']:.4f}x  (B&H: {bnh_equity[-1]:.4f}x)")
print(f"    Max Drawdown:{best_result['drawdown']:.2%}")
print(f"    Win Rate:    {best_result['win_rate']:.2%}")
print(f"    Trades:      {best_result['trades']}")
if best_result['avg_loss'] != 0:
    print(f"    Win/Loss:    {abs(best_result['avg_win'] / best_result['avg_loss']):.2f}")

# ── EXPORT ────────────────────────────────────────────────────────────────────

# joblib.dump(model, MODEL_OUTPUT)
# print(f"\nModel saved → {MODEL_OUTPUT}")
# print("Done.")