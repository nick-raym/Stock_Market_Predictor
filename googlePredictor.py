import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER = "GOOGL"
START_DATE = "2010-01-01"
MODEL_OUTPUT = "model_googl.pkl"
FORWARD_DAYS = 5

FEATURES = [
    # Price / trend
    'SMA_Ratio', 'Price_vs_SMA10', 'Price_vs_SMA50',

    # Momentum
    'RSI',
    'Return_1', 'Return_5', 'Return_10', 'Return_20',
    'Momentum_3', 'Momentum_10', 'Momentum_20',

    # Volatility
    'Volatility_10', 'Volatility_20', 'Volatility_Ratio',

    # Volume
    'Volume_Change', 'Volume_SMA_Ratio',
    'Trend_Strength',

    # Broad market
    'Market_Return', 'Market_Return_5', 'Market_vs_SMA20',

    # GOOGL-specific: two sectors (XLK tech + XLC communication services)
    'Sector_XLK_Return', 'Sector_XLC_Return',
    'XLK_vs_Market', 'XLC_vs_Market',
    'XLC_vs_XLK',           # communication services relative to tech

    # Ad revenue proxies — correlate with GOOGL's core business
    'Meta_Return',          # META moves with digital ad market
    'Meta_Return_5',
    'TTD_Return',           # The Trade Desk — pure-play programmatic ads
    'TTD_Return_5',

    # Macro / fear
    'VIX_Level', 'VIX_Change', 'VIX_vs_SMA20',

    # Earnings proximity
    'Days_to_Earnings', 'Days_since_Earnings',

    # GOOGL vs its own sector — relative strength
    'GOOGL_vs_XLC',
    'GOOGL_vs_XLK',
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
xlc_raw  = download("XLC",  START_DATE)   # communication services ETF
meta_raw = download("META", START_DATE)   # ad revenue proxy
ttd_raw  = download("TTD",  START_DATE)   # programmatic ads proxy

vix_raw = yf.download("^VIX", start=START_DATE, progress=False)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

spy_close  = spy_raw['Close']
xlk_close  = xlk_raw['Close']
xlc_close  = xlc_raw['Close']
meta_close = meta_raw['Close']
ttd_close  = ttd_raw['Close']
vix_close  = vix_raw['Close']

spy_returns    = spy_close.pct_change().rename('spy_ret')
spy_ret5       = spy_close.pct_change(5).rename('spy_ret5')
spy_vs_sma20   = (spy_close / spy_close.rolling(20).mean()).rename('spy_vs_sma20')

xlk_returns    = xlk_close.pct_change().rename('xlk_ret')
xlc_returns    = xlc_close.pct_change().rename('xlc_ret')

meta_returns   = meta_close.pct_change().rename('meta_ret')
meta_ret5      = meta_close.pct_change(5).rename('meta_ret5')
ttd_returns    = ttd_close.pct_change().rename('ttd_ret')
ttd_ret5       = ttd_close.pct_change(5).rename('ttd_ret5')

vix_level      = vix_close.rename('vix')
vix_change     = vix_close.pct_change().rename('vix_chg')
vix_vs_sma20   = (vix_close / vix_close.rolling(20).mean()).rename('vix_vs_sma20')

print("  SPY, XLK, XLC, META, TTD, VIX downloaded.")

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
    df['Market_Return']   = spy_returns.reindex(df.index)
    df['Market_Return_5'] = spy_ret5.reindex(df.index)
    df['Market_vs_SMA20'] = spy_vs_sma20.reindex(df.index)

    # Dual sector context
    df['Sector_XLK_Return'] = xlk_returns.reindex(df.index)
    df['Sector_XLC_Return'] = xlc_returns.reindex(df.index)
    df['XLK_vs_Market']     = df['Sector_XLK_Return'] - df['Market_Return']
    df['XLC_vs_Market']     = df['Sector_XLC_Return'] - df['Market_Return']
    df['XLC_vs_XLK']        = df['Sector_XLC_Return'] - df['Sector_XLK_Return']

    # Ad revenue proxies
    df['Meta_Return']   = meta_returns.reindex(df.index)
    df['Meta_Return_5'] = meta_ret5.reindex(df.index)
    df['TTD_Return']    = ttd_returns.reindex(df.index)
    df['TTD_Return_5']  = ttd_ret5.reindex(df.index)

    # VIX
    df['VIX_Level']    = vix_level.reindex(df.index)
    df['VIX_Change']   = vix_change.reindex(df.index)
    df['VIX_vs_SMA20'] = vix_vs_sma20.reindex(df.index)

    # Earnings proximity
    df['Days_to_Earnings']    = days_to_earn.reindex(df.index)
    df['Days_since_Earnings'] = days_since_earn.reindex(df.index)

    # GOOGL relative strength vs its own sectors
    df['GOOGL_vs_XLC'] = df['Return_5'] - df['Sector_XLC_Return'].rolling(5).sum()
    df['GOOGL_vs_XLK'] = df['Return_5'] - df['Sector_XLK_Return'].rolling(5).sum()

    return df

# ── DOWNLOAD GOOGL & BUILD FEATURES ──────────────────────────────────────────

print("=" * 60)
print(f"PHASE 1: Downloading {TICKER} (2010 → today)")
print("=" * 60)

raw = download(TICKER, START_DATE)
if raw.empty:
    raise RuntimeError(f"No data returned for {TICKER}.")

print(f"  Fetching {TICKER} earnings dates...")
days_to_earn, days_since_earn = get_earnings_features(TICKER, raw.index)

df = build_features(raw, days_to_earn, days_since_earn)
df['Next_Return_1d'] = df['Close'].pct_change().shift(-1)
df['Next_Return_5d'] = df['Close'].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
df = df.dropna()

print(f"  {TICKER}: {len(df)} rows")
print(f"  1d return — Mean: {df['Next_Return_1d'].mean():.4%}  Std: {df['Next_Return_1d'].std():.4%}  % UP: {(df['Next_Return_1d'] > 0).mean():.2%}")
print(f"  5d return — Mean: {df['Next_Return_5d'].mean():.4%}  Std: {df['Next_Return_5d'].std():.4%}  % UP: {(df['Next_Return_5d'] > 0).mean():.2%}")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"PHASE 2: Training XGBoost Regressor (target = {FORWARD_DAYS}-day return)")
print("=" * 60)

X = df[FEATURES]
y = df['Next_Return_5d']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train: {len(X_train)} rows ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"  Test:  {len(X_test)} rows ({X_test.index[0].date()} → {X_test.index[-1].date()})")

# ── TRAIN ─────────────────────────────────────────────────────────────────────

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("  Training complete.")

# ── EVALUATE ──────────────────────────────────────────────────────────────────

y_pred_5d = model.predict(X_test)
actual_5d = y_test.values

mae           = mean_absolute_error(actual_5d, y_pred_5d)
r2            = r2_score(actual_5d, y_pred_5d)
pearson_r,  p_pearson  = pearsonr(actual_5d, y_pred_5d)
spearman_r, p_spearman = spearmanr(actual_5d, y_pred_5d)
direction_acc = (np.sign(y_pred_5d) == np.sign(actual_5d)).mean()

print(f"\n  MAE:                {mae:.6f}")
print(f"  R²:                 {r2:.6f}  (>0 = better than predicting the mean)")
print(f"  Pearson r:          {pearson_r:.4f}  (p={p_pearson:.4f})")
print(f"  Spearman r:         {spearman_r:.4f}  (p={p_spearman:.4f})")
print(f"  Direction accuracy: {direction_acc:.2%}")

print(f"\n  Prediction distribution:")
print(pd.Series(y_pred_5d).describe())

# Quintile table
pred_series   = pd.Series(y_pred_5d, index=X_test.index, name='pred')
actual_series = pd.Series(actual_5d,  index=X_test.index, name='actual')

try:
    buckets   = pd.qcut(pred_series, q=5, duplicates='drop')
    bucket_df = pd.DataFrame({'pred': pred_series, 'actual': actual_series, 'bucket': buckets})
    print(f"\n  Quintile table (Q1=most bearish → Q5=most bullish):")
    print(f"  {'Bucket':<32} {'Avg Predicted':>15} {'Avg Actual':>12} {'% UP (5d)':>10}")
    for name, group in bucket_df.groupby('bucket', observed=True):
        print(f"  {str(name):<32} {group['pred'].mean():>14.4%} {group['actual'].mean():>11.4%} {(group['actual'] > 0).mean():>10.2%}")
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
print("PHASE 3: Trading Simulation (5-day signal → daily P&L)")
print("=" * 60)

test_df = df.iloc[len(X_train):len(X_train) + len(y_pred_5d)].copy()
test_df['Pred_5d'] = y_pred_5d
test_df = test_df.dropna(subset=['Next_Return_1d']).reset_index(drop=True)

daily_returns = test_df['Next_Return_1d'].values
pred_5d       = test_df['Pred_5d'].values
n             = len(daily_returns)

def build_position_series(signals_raw, hold=5):
    positions = np.zeros(n)
    counts    = np.zeros(n)
    for i, sig in enumerate(signals_raw):
        end = min(i + hold, n)
        positions[i:end] += sig
        counts[i:end]    += 1
    counts = np.where(counts == 0, 1, counts)
    return positions / counts

COST = 0.0005
bnh_equity = (1 + daily_returns).cumprod()
print(f"\n  Buy & Hold Final Return: {bnh_equity[-1]:.4f}x")
print(f"\n  {'Strategy':<36} {'Final Return':<16} {'Win Rate':<12} {'Max DD':<10} {'Trades'}")
print("  " + "-" * 82)

best_result = None

for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
    for mode in ['long_only', 'long_short', 'long_sit_short']:

        if mode == 'long_only':
            raw_signals = np.where(pred_5d > threshold, 1.0, 0.0)
            label = f"t={threshold:.3f} long-only"
        elif mode == 'long_short':
            raw_signals = np.where(pred_5d > threshold, 1.0,
                          np.where(pred_5d < -threshold, -1.0, 0.0))
            label = f"t={threshold:.3f} long/short"
        else:
            scaled = np.clip(pred_5d / 0.02, -1.0, 1.0)
            raw_signals = np.where(pred_5d > threshold, scaled,
                          np.where(pred_5d < -threshold, scaled, 0.0))
            label = f"t={threshold:.3f} scaled L/S"

        position      = build_position_series(raw_signals, hold=FORWARD_DAYS)
        pos_change    = np.abs(np.diff(np.concatenate([[0], position])))
        costs         = pos_change * COST
        strat_returns = position * daily_returns - costs
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

print(f"\n  Best configuration: {best_result['label']}")
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