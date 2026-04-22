import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKERS = [
    "MSFT",
    "AAPL",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "V",
    "UNH",
    "JNJ",
    "WMT",
    "PG",
    "DIS",
    "MA",
    "HD",
    "BAC",
    "VZ",
    "CMCSA",
    "NFLX",
    "CRM",
    "INTC",
]
START_DATE = "2015-01-01"
MODEL_OUTPUT = "model.pkl"
HORIZON = 5  # predict 5-day forward return

FEATURES = [
    "SMA_Ratio",
    "Price_vs_SMA10",
    "Price_vs_SMA20",
    "RSI",
    "RSI_Slope",
    "Return_1",
    "Return_5",
    "Return_10",
    "Return_21",
    "Volume_Change",
    "Volume_SMA_Ratio",
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Momentum_3",
    "Momentum_10",
    "Momentum_21",
    "Market_Return",
    "Market_Return_5",
    "Market_Return_21",
    "Trend_Strength",
    "VIX_Level",
    "VIX_Change",
    "MACD_Signal",
    "BB_Position",  # where price sits within Bollinger Bands
    "ATR_Ratio",  # normalised average true range
]

# ── MARKET DATA ───────────────────────────────────────────────────────────────

print("Downloading SPY and VIX benchmark...")
spy_raw = yf.download("SPY", start=START_DATE, auto_adjust=True, progress=False)
vix_raw = yf.download("^VIX", start=START_DATE, progress=False)

if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_raw.columns = spy_raw.columns.get_level_values(0)
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

spy_close = spy_raw["Close"]
spy_returns = spy_close.pct_change().rename("Market_Return")
spy_ret5 = spy_close.pct_change(5).rename("Market_Return_5")
spy_ret21 = spy_close.pct_change(21).rename("Market_Return_21")
vix_level = vix_raw["Close"].rename("VIX_Level")
vix_change = vix_raw["Close"].pct_change().rename("VIX_Change")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    # ── Moving averages
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["SMA_Ratio"] = sma10 / sma50
    df["Price_vs_SMA10"] = close / sma10
    df["Price_vs_SMA20"] = close / sma20

    # ── RSI + slope
    rsi = compute_rsi(close)
    df["RSI"] = rsi
    df["RSI_Slope"] = rsi.diff(3)

    # ── Returns  (NO shift — these are already in the past)
    df["Return_1"] = close.pct_change(1)
    df["Return_5"] = close.pct_change(5)
    df["Return_10"] = close.pct_change(10)
    df["Return_21"] = close.pct_change(21)

    # ── Momentum (same as returns but named distinctly for clarity)
    df["Momentum_3"] = close.pct_change(3)
    df["Momentum_10"] = close.pct_change(10)
    df["Momentum_21"] = close.pct_change(21)

    # ── Volume
    vol_sma10 = volume.rolling(10).mean()
    df["Volume_Change"] = volume.pct_change()
    df["Volume_SMA_Ratio"] = volume / vol_sma10

    # ── Volatility
    daily_ret = close.pct_change()
    df["Volatility_5"] = daily_ret.rolling(5).std()
    df["Volatility_10"] = daily_ret.rolling(10).std()
    df["Volatility_20"] = daily_ret.rolling(20).std()

    # ── Trend strength
    df["Trend_Strength"] = df["Return_5"].abs() / (df["Volatility_10"] + 1e-9)

    # ── MACD signal (12/26 EMA diff normalised by price)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = (ema12 - ema26) / close
    df["MACD_Signal"] = macd - macd.ewm(span=9).mean()

    # ── Bollinger Band position  (0 = lower band, 1 = upper band)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Position"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ── ATR ratio (normalised volatility using H/L/C)
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["ATR_Ratio"] = atr / close

    # ── Market context
    df["Market_Return"] = spy_returns.reindex(df.index)
    df["Market_Return_5"] = spy_ret5.reindex(df.index)
    df["Market_Return_21"] = spy_ret21.reindex(df.index)

    # ── VIX
    df["VIX_Level"] = vix_level.reindex(df.index).ffill().fillna(vix_level.mean())
    df["VIX_Change"] = vix_change.reindex(df.index).fillna(0)

    return df


# ── COLLECT TRAINING DATA ─────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1: Collecting training data (2015 → today)")
print("=" * 60)

all_dfs = []
for ticker in TICKERS:
    print(f"  Downloading {ticker}...")
    raw = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  WARNING: No data for {ticker}, skipping.")
        continue

    df = build_features(raw)

    # TARGET: actual forward 5-day return — shift(-5) so each row has the FUTURE return
    df["Target"] = df["Close"].pct_change(HORIZON).shift(-HORIZON)
    df["Next_Return"] = df["Target"]
    df["Ticker"] = ticker
    df = df.dropna(subset=FEATURES + ["Target"])

    print(f"  {ticker}: {len(df)} rows")
    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError("No data collected.")

combined = pd.concat(all_dfs).sort_index().reset_index(drop=True)
print(f"\nTotal training rows: {len(combined)} across {len(all_dfs)} tickers")

# ── TIME-BASED TRAIN / TEST SPLIT ────────────────────────────────────────────
# Use last ~20% of dates as test (avoids future leakage from shuffle=False on mixed tickers)

print("\n" + "=" * 60)
print("PHASE 2: Training model")
print("=" * 60)

split_idx = int(len(combined) * 0.8)
train_df = combined.iloc[:split_idx]
test_df = combined.iloc[split_idx:]

X_train = train_df[FEATURES]
y_train = train_df["Target"]
X_test = test_df[FEATURES]
y_test = test_df["Target"]

print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"  Target mean (train): {y_train.mean():.4%}")

# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,  # higher = more conservative, less overfit
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)
print("  Training complete.")

# ── EVALUATE ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
y_actual = y_test.values

mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
pearson_r, p = pearsonr(y_pred, y_actual)

print(f"\nMAE:           {mae:.6f}")
print(f"R²:            {r2:.6f}")
print(f"Pearson r:     {pearson_r:.4f}  (p={p:.4f})")

direction_acc = (np.sign(y_pred) == np.sign(y_actual)).mean()
print(f"Direction acc: {direction_acc:.2%}")

print("\nPrediction distribution:")
print(pd.Series(y_pred).describe().to_string())

# Quintile analysis — use labels to avoid duplicate-edge issues
print("\nQuintile analysis (Q1=bearish → Q5=bullish):")
pred_series = pd.Series(y_pred, index=X_test.index)
labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
buckets = pd.qcut(pred_series.rank(method="first"), q=5, labels=labels)
bucket_df = pd.DataFrame({"pred": y_pred, "actual": y_actual, "bucket": buckets})
print(f"  {'Bucket':<8} {'Avg Pred':>12} {'Avg Actual':>12} {'% UP':>8}")
for name, group in bucket_df.groupby("bucket", observed=True):
    print(
        f"  {name:<8} {group['pred'].mean():>12.4%}"
        f" {group['actual'].mean():>11.4%}"
        f" {(group['actual'] > 0).mean():>8.2%}"
    )

importance_df = pd.DataFrame(
    {"Feature": FEATURES, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)
print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# ── TRADING SIMULATION ────────────────────────────────────────────────────────
# FIX: rows are individual ticker-days.  We must not compound them as if they're
# sequential single-position days.  Instead we:
#   1. Average all predictions on each calendar date  →  one signal per day
#   2. Hold for HORIZON days, then rebalance (non-overlapping windows)
#   3. Compound those non-overlapping period returns

print("\n" + "=" * 60)
print("PHASE 3: Trading Simulation (non-overlapping 5-day windows)")
print("=" * 60)

sim_df = test_df[["Next_Return", "Ticker"]].copy()
sim_df["Pred"] = y_pred

# Get the actual dates from the original index before reset_index stripped them
sim_df["Date"] = (
    combined.iloc[split_idx:].index
    if hasattr(combined.index, "date")
    else pd.RangeIndex(len(sim_df))
)  # fallback — dates were dropped by reset_index

# Re-attach dates by rebuilding from original per-ticker data
# (combined was reset_index'd, so we use an integer step proxy instead)
# Group by every HORIZON rows to simulate non-overlapping rebalance
sim_df = sim_df.reset_index(drop=True)

# Sample every HORIZON-th row to get non-overlapping periods
non_overlap = sim_df.iloc[::HORIZON].copy()
n_periods = len(non_overlap)

print(f"  Non-overlapping {HORIZON}-day periods: {n_periods}")

preds = non_overlap["Pred"].values
returns = non_overlap["Next_Return"].values

# Buy-and-hold baseline
bnh_equity = (1 + returns).cumprod()

print(f"\nThreshold sweep  (long-only, {HORIZON}-day hold, non-overlapping):")
print(
    f"  {'Threshold':<12} {'Final Return':>14} {'Ann. Return':>12} {'Win Rate':>10} {'Trades':>8}"
)
print("  " + "-" * 60)

years = n_periods * HORIZON / 252
best = None

for thr in [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020]:
    mask = preds > thr
    strat_r = np.where(mask, returns, 0.0)
    equity = (1 + strat_r).cumprod()

    final = equity[-1]
    ann = final ** (1 / years) - 1 if years > 0 else 0
    trades = int(mask.sum())
    wr = (strat_r[mask] > 0).mean() if trades > 0 else 0

    print(f"  {thr:<12.3f} {final:>14.4f}x {ann:>11.2%} {wr:>10.2%} {trades:>8}")

    if best is None or final > best["equity"]:
        best = {
            "threshold": thr,
            "equity": final,
            "ann": ann,
            "win_rate": wr,
            "trades": trades,
        }

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKERS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "V", "UNH", "JNJ", "WMT", "PG", "DIS", "MA",
    "HD", "BAC", "VZ", "CMCSA", "NFLX", "CRM", "INTC"
]

SECTOR_ETFS = {
    'Tech':       'XLK',
    'Finance':    'XLF',
    'Health':     'XLV',
    'Consumer':   'XLY',
    'Industrial': 'XLI',
}

TICKER_SECTOR = {
    "MSFT": "Tech",    "AAPL": "Tech",     "GOOGL": "Tech",   "AMZN": "Consumer",
    "TSLA": "Consumer","META": "Tech",      "NVDA":  "Tech",   "JPM":  "Finance",
    "V":    "Finance", "UNH":  "Health",    "JNJ":   "Health", "WMT":  "Consumer",
    "PG":   "Consumer","DIS":  "Consumer",  "MA":    "Finance","HD":   "Consumer",
    "BAC":  "Finance", "VZ":   "Industrial","CMCSA": "Consumer","NFLX":"Consumer",
    "CRM":  "Tech",    "INTC": "Tech",
}

START_DATE  = "2015-01-01"
HORIZON     = 5
MODEL_PATH  = "model.pkl"
TRANS_COST  = 0.001    # 0.1% round-trip
SLIPPAGE    = 0.0005   # 0.05% slippage

WF_INITIAL_YEARS = 3
WF_STEP_MONTHS   = 6
WF_TEST_MONTHS   = 6

TICKER_FEATURES = [
    'SMA_Ratio', 'Price_vs_SMA20',
    'RSI', 'RSI_Slope',
    'Return_5', 'Return_10', 'Return_21',
    'Volume_Change', 'Volume_SMA_Ratio',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'MACD_Signal', 'BB_Position', 'ATR_Ratio',
    'Earnings_Proximity',
]

CS_FEATURES = [
    'CS_Return_5_Rank',
    'CS_Return_21_Rank',
    'CS_Volatility_10_Rank',
    'CS_RSI_Rank',
    'CS_Volume_SMA_Ratio_Rank',
    'CS_Momentum_Score',
]

MACRO_FEATURES = [
    'Market_Return', 'Market_Return_5', 'Market_Return_21',
    'VIX_Level', 'VIX_Change', 'VIX_Regime',
    'Yield_Curve',
    'DXY_Return',
    'Sector_Return_5',
    'Sector_Rel_Return',
]

ALL_FEATURES = TICKER_FEATURES + CS_FEATURES + MACRO_FEATURES

# ── DOWNLOAD MACRO DATA ───────────────────────────────────────────────────────

print("Downloading macro data...")


def dl(ticker, start=START_DATE):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close']


spy_close = dl("SPY")
vix_close = dl("^VIX")
tlt_close = dl("TLT")
shy_close = dl("SHY")
dxy_close = dl("UUP")

sector_closes = {name: dl(etf) for name, etf in SECTOR_ETFS.items()}

mkt_ret    = spy_close.pct_change()
mkt_ret5   = spy_close.pct_change(5)
mkt_ret21  = spy_close.pct_change(21)
vix_lvl    = vix_close
vix_chg    = vix_close.pct_change()
vix_regime = (vix_close > 25).astype(float)

yield_curve = (tlt_close / tlt_close.rolling(20).mean()) - \
              (shy_close / shy_close.rolling(20).mean())
dxy_ret5 = dxy_close.pct_change(5)
sector_ret5 = {name: s.pct_change(5) for name, s in sector_closes.items()}

print("Macro data ready.\n")

# ── EARNINGS PROXIMITY ────────────────────────────────────────────────────────

def get_earnings_dates(ticker, start=START_DATE):
    try:
        t   = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is None or cal.empty:
            return np.array([], dtype='datetime64[D]')
        dates = pd.DatetimeIndex(cal.index).normalize()
        dates = dates[dates >= pd.Timestamp(start)]
        return np.array(dates, dtype='datetime64[D]')
    except Exception:
        return np.array([], dtype='datetime64[D]')


def earnings_proximity(date_index, earnings_arr, window=30):
    if len(earnings_arr) == 0:
        return pd.Series(0.0, index=date_index)
    dates_np = np.array(date_index, dtype='datetime64[D]')
    result   = np.full(len(dates_np), float(window))
    for ed in earnings_arr:
        diff   = np.abs((dates_np - ed).astype(int))
        result = np.minimum(result, diff)
    return pd.Series(1.0 - np.minimum(result, window) / window, index=date_index)

# ── PER-TICKER FEATURE ENGINEERING ───────────────────────────────────────────

def compute_rsi(series, window=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_ticker_features(raw, ticker, earnings_arr):
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    close  = df['Close']
    volume = df['Volume']
    high   = df['High']
    low    = df['Low']

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df['SMA_Ratio']      = sma10 / sma50
    df['Price_vs_SMA20'] = close / sma20

    rsi = compute_rsi(close)
    df['RSI']       = rsi
    df['RSI_Slope'] = rsi.diff(3)

    df['Return_5']  = close.pct_change(5)
    df['Return_10'] = close.pct_change(10)
    df['Return_21'] = close.pct_change(21)

    vol_sma10 = volume.rolling(10).mean()
    df['Volume_Change']    = volume.pct_change()
    df['Volume_SMA_Ratio'] = volume / vol_sma10

    daily_ret = close.pct_change()
    df['Volatility_5']  = daily_ret.rolling(5).std()
    df['Volatility_10'] = daily_ret.rolling(10).std()
    df['Volatility_20'] = daily_ret.rolling(20).std()

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = (ema12 - ema26) / close
    df['MACD_Signal'] = macd - macd.ewm(span=9).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_Position'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low  - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR_Ratio'] = tr.rolling(14).mean() / close

    df['Earnings_Proximity'] = earnings_proximity(df.index, earnings_arr)

    df['Market_Return']    = mkt_ret.reindex(df.index)
    df['Market_Return_5']  = mkt_ret5.reindex(df.index)
    df['Market_Return_21'] = mkt_ret21.reindex(df.index)
    df['VIX_Level']        = vix_lvl.reindex(df.index).ffill().fillna(float(vix_lvl.mean()))
    df['VIX_Change']       = vix_chg.reindex(df.index).fillna(0)
    df['VIX_Regime']       = vix_regime.reindex(df.index).ffill().fillna(0)
    df['Yield_Curve']      = yield_curve.reindex(df.index).ffill().fillna(0)
    df['DXY_Return']       = dxy_ret5.reindex(df.index).fillna(0)

    sector = TICKER_SECTOR.get(ticker, 'Tech')
    df['Sector_Return_5']   = sector_ret5[sector].reindex(df.index).fillna(0)
    df['Sector_Rel_Return'] = df['Return_5'] - df['Sector_Return_5']

    df['Ticker'] = ticker
    return df

# ── PHASE 1: COLLECT DATA ─────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 1: Downloading & building per-ticker features")
print("=" * 60)

ticker_dfs = {}

for ticker in TICKERS:
    print(f"  {ticker}...", end=" ", flush=True)
    raw = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    if raw.empty:
        print("SKIP")
        continue
    earnings_arr = get_earnings_dates(ticker)
    df = build_ticker_features(raw, ticker, earnings_arr)
    df['Target']      = df['Close'].pct_change(HORIZON).shift(-HORIZON)
    df['Next_Return'] = df['Target']
    ticker_dfs[ticker] = df
    print(f"{len(df)} rows")

# ── CROSS-SECTIONAL RANK FEATURES ────────────────────────────────────────────

print("\nBuilding cross-sectional rank features...")

panel = pd.concat(list(ticker_dfs.values()), axis=0)
panel.index.name = 'Date'

rank_cols = {
    'CS_Return_5_Rank':         'Return_5',
    'CS_Return_21_Rank':        'Return_21',
    'CS_Volatility_10_Rank':    'Volatility_10',
    'CS_RSI_Rank':              'RSI',
    'CS_Volume_SMA_Ratio_Rank': 'Volume_SMA_Ratio',
}

for cs_name, raw_col in rank_cols.items():
    pivot  = panel.pivot_table(index=panel.index, columns='Ticker', values=raw_col)
    ranked = pivot.rank(axis=1, pct=True)
    for ticker in ticker_dfs:
        if ticker in ranked.columns:
            ticker_dfs[ticker][cs_name] = ranked[ticker].reindex(ticker_dfs[ticker].index)

for ticker in ticker_dfs:
    df = ticker_dfs[ticker]
    df['CS_Momentum_Score'] = df[['CS_Return_5_Rank','CS_Return_21_Rank']].mean(axis=1)

print("Cross-sectional features built.\n")

# ── COMBINE ───────────────────────────────────────────────────────────────────

all_dfs = []
for ticker, df in ticker_dfs.items():
    df2 = df.dropna(subset=ALL_FEATURES + ['Target'])
    all_dfs.append(df2)

combined = pd.concat(all_dfs).sort_index()
combined.index.name = 'Date'
combined = combined.reset_index()

print(f"Total rows (after dropna): {len(combined):,} across {len(all_dfs)} tickers")

# ── PHASE 2: WALK-FORWARD VALIDATION ─────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2: Walk-forward validation")
print("=" * 60)

combined['Date'] = pd.to_datetime(combined['Date'])
combined = combined.sort_values('Date').reset_index(drop=True)

min_date = combined['Date'].min()
max_date = combined['Date'].max()

fold_start = min_date + pd.DateOffset(years=WF_INITIAL_YEARS)
folds = []
while fold_start + pd.DateOffset(months=WF_TEST_MONTHS) <= max_date:
    fold_end = fold_start + pd.DateOffset(months=WF_TEST_MONTHS)
    folds.append((min_date, fold_start, fold_end))
    fold_start = fold_end

print(f"  Date range : {min_date.date()} → {max_date.date()}")
print(f"  Folds      : {len(folds)}")
print(f"  Initial training window: {WF_INITIAL_YEARS}y | "
      f"Step: {WF_STEP_MONTHS}mo | Test: {WF_TEST_MONTHS}mo\n")

fold_results = []

for i, (tr_start, tr_end, te_end) in enumerate(folds):
    tr_mask = (combined['Date'] >= tr_start) & (combined['Date'] < tr_end)
    te_mask = (combined['Date'] >= tr_end)   & (combined['Date'] < te_end)

    X_tr = combined.loc[tr_mask, ALL_FEATURES]
    y_tr = combined.loc[tr_mask, 'Target']
    X_te = combined.loc[te_mask, ALL_FEATURES]
    y_te = combined.loc[te_mask, 'Target']

    if len(X_tr) < 500 or len(X_te) < 50:
        continue

    mdl = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.75,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.2, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    mdl.fit(X_tr, y_tr, verbose=False)
    preds = mdl.predict(X_te)

    fold_df = combined.loc[te_mask, ['Date', 'Ticker', 'Next_Return']].copy()
    fold_df['Pred']   = preds
    fold_df['y_true'] = y_te.values
    fold_results.append(fold_df)

    mae  = mean_absolute_error(y_te, preds)
    pr,_ = pearsonr(preds, y_te)
    da   = (np.sign(preds) == np.sign(y_te.values)).mean()
    print(f"  Fold {i+1:02d}  {str(tr_end.date())[:7]}→{str(te_end.date())[:7]}  "
          f"train={len(X_tr):>6,}  test={len(X_te):>5,}  "
          f"MAE={mae:.5f}  r={pr:.4f}  dir={da:.2%}")

# ── FINAL MODEL (all data up to last train_end) ───────────────────────────────

last_tr_end  = folds[-1][1]
final_mask   = combined['Date'] < last_tr_end
X_final      = combined.loc[final_mask, ALL_FEATURES]
y_final      = combined.loc[final_mask, 'Target']

final_model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.75,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.2, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)
final_model.fit(X_final, y_final, verbose=False)
joblib.dump(final_model, MODEL_PATH)
print(f"\nFinal model saved → {MODEL_PATH}  ({len(X_final):,} training rows)")

# ── PHASE 3: AGGREGATE OOS METRICS ───────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3: Aggregate out-of-sample metrics")
print("=" * 60)

oos          = pd.concat(fold_results).reset_index(drop=True)
y_pred_all   = oos['Pred'].values
y_actual_all = oos['y_true'].values

mae      = mean_absolute_error(y_actual_all, y_pred_all)
r2       = r2_score(y_actual_all, y_pred_all)
pr, pval = pearsonr(y_pred_all, y_actual_all)
sr, _    = spearmanr(y_pred_all, y_actual_all)
da       = (np.sign(y_pred_all) == np.sign(y_actual_all)).mean()

print(f"  OOS rows           : {len(oos):,}")
print(f"  MAE                : {mae:.6f}")
print(f"  R²                 : {r2:.6f}")
print(f"  Pearson r          : {pr:.4f}  (p={pval:.6f})")
print(f"  Spearman r         : {sr:.4f}")
print(f"  Direction accuracy : {da:.2%}")

print("\n  Prediction distribution:")
print(pd.Series(y_pred_all).describe().to_string())

print("\n  Quintile analysis (Q1=bearish → Q5=bullish):")
pred_series = pd.Series(y_pred_all)
labels      = ['Q1','Q2','Q3','Q4','Q5']
buckets     = pd.qcut(pred_series.rank(method='first'), q=5, labels=labels)
bdf = pd.DataFrame({'pred': y_pred_all, 'actual': y_actual_all, 'bucket': buckets})
print(f"  {'Q':<4} {'Avg Pred':>10} {'Avg Actual':>12} {'% UP':>8} {'N':>7}")
for name, grp in bdf.groupby('bucket', observed=True):
    print(f"  {name:<4} {grp['pred'].mean():>10.4%}"
          f" {grp['actual'].mean():>12.4%}"
          f" {(grp['actual']>0).mean():>8.2%}"
          f" {len(grp):>7,}")

imp_df = (
    pd.DataFrame({'Feature': ALL_FEATURES, 'Importance': final_model.feature_importances_})
    .sort_values('Importance', ascending=False)
)
print("\n  Feature Importances (final model):")
print(imp_df.to_string(index=False))

# ── PHASE 4: PER-TICKER SIMULATION ───────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 4: Per-ticker simulation (with transaction costs & slippage)")
print("=" * 60)
print(f"  Transaction cost: {TRANS_COST:.2%}  |  Slippage: {SLIPPAGE:.2%}  |  Hold: {HORIZON} days\n")


def simulate_ticker(df_t, threshold=0.004):
    df_t  = df_t.sort_values('Date').reset_index(drop=True)
    rows  = df_t.iloc[::HORIZON]
    preds = rows['Pred'].values
    rets  = rows['Next_Return'].values

    equity = 1.0
    n_trades = wins = 0

    for pred, ret in zip(preds, rets):
        if pred > threshold:
            period_ret = ret - TRANS_COST - SLIPPAGE
            equity    *= (1 + period_ret)
            n_trades  += 1
            wins      += int(period_ret > 0)

    n_periods = len(rows)
    years     = n_periods * HORIZON / 252
    ann_ret   = equity ** (1 / years) - 1 if years > 0 else 0
    win_rate  = wins / n_trades if n_trades > 0 else 0

    bnh     = float((1 + rets).cumprod()[-1]) if len(rets) else 1.0
    bnh_ann = bnh ** (1 / years) - 1 if years > 0 else 0

    return dict(final_equity=equity, ann_return=ann_ret, win_rate=win_rate,
                n_trades=n_trades, bnh_equity=bnh, bnh_ann=bnh_ann)


thresholds = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010]

print(f"  {'Ticker':<8} {'Best Thr':>9} {'Strat Ann':>10} {'BnH Ann':>9} "
      f"{'Win Rate':>10} {'Trades':>8} {'Alpha':>9}")
print("  " + "-" * 72)

all_results = []
for ticker in sorted(oos['Ticker'].unique()):
    df_t = oos[oos['Ticker'] == ticker].copy()
    if len(df_t) < 20:
        continue
    best = None
    for thr in thresholds:
        res = simulate_ticker(df_t, threshold=thr)
        res['threshold'] = thr
        res['ticker']    = ticker
        if best is None or res['ann_return'] > best['ann_return']:
            best = res
    alpha = best['ann_return'] - best['bnh_ann']
    print(f"  {ticker:<8} {best['threshold']:>9.3f} {best['ann_return']:>9.2%}"
        f" {best['bnh_ann']:>9.2%} {best['win_rate']:>10.2%}"
        f" {best['n_trades']:>8} {alpha:>+9.2%}")
    all_results.append(best)

if all_results:
    ann_rets = [r['ann_return'] for r in all_results]
    bnh_rets = [r['bnh_ann']    for r in all_results]
    alphas   = [r['ann_return'] - r['bnh_ann'] for r in all_results]
    print(f"\n  Portfolio summary (equal-weight across tickers):")
    print(f"    Avg strategy ann. return : {np.mean(ann_rets):.2%}")
    print(f"    Avg buy-and-hold ann.    : {np.mean(bnh_rets):.2%}")
    print(f"    Avg alpha                : {np.mean(alphas):+.2%}")
    print(f"    Tickers beating BnH      : {sum(a > 0 for a in alphas)}/{len(alphas)}")


print(f"\nLong/Short strategy (threshold sweep):")
print(
    f"  {'Threshold':<12} {'Final Return':>14} {'Ann. Return':>12} {'Win Rate':>10} {'Trades':>8}"
)
print("  " + "-" * 60)

for thr in [0.000, 0.002, 0.004, 0.006, 0.008, 0.010]:
    mask_l = preds > thr
    mask_s = preds < -thr
    strat_r = np.where(mask_l, returns, np.where(mask_s, -returns, 0.0))
    equity = (1 + strat_r).cumprod()
    final = equity[-1]
    ann = final ** (1 / years) - 1 if years > 0 else 0
    trades = int((mask_l | mask_s).sum())
    wr = (strat_r[strat_r != 0] > 0).mean() if trades > 0 else 0

    print(f"  {thr:<12.3f} {final:>14.4f}x {ann:>11.2%} {wr:>10.2%} {trades:>8}")

bnh_ann = bnh_equity[-1] ** (1 / years) - 1 if years > 0 else 0
print(
    f"\nBest long-only : threshold={best['threshold']}, {best['equity']:.4f}x total  ({best['ann']:.2%} ann.)"
)
print(f"Buy & Hold     : {bnh_equity[-1]:.4f}x total  ({bnh_ann:.2%} ann.)")

# ── EXPORT MODEL ──────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_OUTPUT)
print(f"\nModel saved → {MODEL_OUTPUT}")
