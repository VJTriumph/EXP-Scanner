import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
LENGTH        = 90       # regression lookback bars
DAYS_IN_YEAR  = 252      # bars per year for annualization
BETA_WINDOW   = 252      # lookback for beta calculation
R2_THRESHOLD  = 0.0      # minimum R2 to colour signal
WITH_RSQUARED = True     # multiply slope by R2
WITH_ANNUALIZE= True     # annualize slope
BENCHMARK     = "^NSEI"  # Nifty 50 for beta
FETCH_PERIOD  = "2y"     # how much history to download

DATA_DIR      = "data"
INDICES_CSV   = os.path.join(DATA_DIR, "indices.csv")
STOCKS_CSV    = os.path.join(DATA_DIR, "ind_niftytotalmarket_list.csv")
OUTPUT_JSON   = os.path.join(DATA_DIR, "results.json")

# ── MATH ────────────────────────────────────────────────────────────────────
def compute_exp_slope(prices, length=LENGTH, days_in_year=DAYS_IN_YEAR,
                      with_rsquared=WITH_RSQUARED, with_annualize=WITH_ANNUALIZE,
                      r2_threshold=R2_THRESHOLD):
    if len(prices) < length:
        return None
    sl = np.array(prices[-length:], dtype=float)
    sl = sl[~np.isnan(sl)]
    if len(sl) < length * 0.8:
        return None
    log_p = np.log(sl)
    x = np.arange(len(log_p), dtype=float)

    # correlation-based OLS slope (same as Pine Script)
    mx, my = x.mean(), log_p.mean()
    sx = x.std()
    sy = log_p.std()
    if sx == 0 or sy == 0:
        return None
    c = np.corrcoef(x, log_p)[0, 1]
    slope = c * (sy / sx)

    # annualized
    annualized = (np.exp(slope) ** days_in_year - 1) * 100

    # R-squared using sequential index vs log price
    cum_x = np.arange(1, len(log_p) + 1, dtype=float)
    r2_raw = np.corrcoef(cum_x, log_p)[0, 1] ** 2
    r2 = float(r2_raw) if not np.isnan(r2_raw) else 0.0

    base  = annualized if with_annualize else slope
    final = base * (r2 if with_rsquared else 1.0)
    above = r2 > r2_threshold

    return {
        "slope":          round(float(final), 4),
        "raw_slope":      round(float(annualized), 4),
        "r2":             round(r2, 4),
        "above_threshold": above
    }


def compute_52w(prices):
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    window = arr[-252:] if len(arr) >= 252 else arr
    high52 = float(window.max())
    current = float(arr[-1])
    pct = round((current - high52) / high52 * 100, 2)
    return {
        "current":       round(current, 2),
        "high52":        round(high52, 2),
        "pct_from_high": pct
    }


def compute_beta(stock_prices, bench_prices, window=BETA_WINDOW):
    sp = np.array(stock_prices, dtype=float)
    bp = np.array(bench_prices, dtype=float)
    n  = min(len(sp), len(bp), window)
    if n < 20:
        return None
    sp = sp[-n:]
    bp = bp[-n:]
    # align & drop nans
    mask = ~(np.isnan(sp) | np.isnan(bp))
    sp, bp = sp[mask], bp[mask]
    if len(sp) < 15:
        return None
    sr = np.diff(sp) / sp[:-1]
    br = np.diff(bp) / bp[:-1]
    if br.std() == 0:
        return None
    cov    = np.cov(sr, br)[0, 1]
    var_b  = np.var(br, ddof=1)
    return round(float(cov / var_b), 3) if var_b != 0 else None


def color_signal(result, prev_slope=None):
    if not result or not result["above_threshold"]:
        return "gray"
    up     = result["slope"] > 0
    rising = (prev_slope is None) or (result["slope"] > prev_slope)
    if up and rising:     return "bright_green"
    if up and not rising: return "faded_green"
    if not up and rising: return "faded_red"
    return "bright_red"


# ── FETCH HELPER ─────────────────────────────────────────────────────────────
def fetch_closes(symbol, period=FETCH_PERIOD):
    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna().tolist()
        return closes if len(closes) >= 10 else None
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        return None


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"EXP Scanner  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Benchmark
    print(f"\nFetching benchmark {BENCHMARK} ...")
    bench_prices = fetch_closes(BENCHMARK)
    if bench_prices is None:
        print("WARNING: benchmark fetch failed — beta will be null")

    # 2. Indices
    print("\n── INDICES ─────────────────────────────────────────────")
    idx_df  = pd.read_csv(INDICES_CSV)
    idx_df.columns = [c.strip() for c in idx_df.columns]
    # support both old (single col) and new (two col) format
    name_col   = idx_df.columns[0]
    symbol_col = idx_df.columns[1] if len(idx_df.columns) > 1 else None

    idx_results = []
    for _, row in idx_df.iterrows():
        name   = str(row[name_col]).strip()
        symbol = str(row[symbol_col]).strip() if symbol_col else name
        if name.lower() in ("indices", "index", ""):
            continue
        print(f"  {symbol:30s}  {name}")
        prices = fetch_closes(symbol)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench_prices) if (prices and bench_prices) else None
        idx_results.append({
            "name":           name,
            "symbol":         symbol,
            "slope_data":     result,
            "high52_data":    h52,
            "beta":           beta,
            "signal_color":   color_signal(result)
        })

    # 3. Stocks
    print("\n── STOCKS ──────────────────────────────────────────────")
    stk_df = pd.read_csv(STOCKS_CSV)
    stk_df.columns = [c.strip() for c in stk_df.columns]
    # filter valid EQ only
    stk_df = stk_df[stk_df["Series"].str.strip() == "EQ"].copy()
    stk_df = stk_df.dropna(subset=["Symbol"])
    stk_df["Symbol"] = stk_df["Symbol"].str.strip()
    stk_df["Industry"] = stk_df["Industry"].str.strip()
    stk_df["Company Name"] = stk_df["Company Name"].str.strip()

    stk_results = []
    total = len(stk_df)
    for i, (_, row) in enumerate(stk_df.iterrows(), 1):
        symbol    = row["Symbol"] + ".NS"
        name      = row["Company Name"]
        industry  = row.get("Industry", "")
        isin      = row.get("ISIN Code", "")
        print(f"  [{i:3d}/{total}]  {symbol:20s}  {name[:35]}")
        prices = fetch_closes(symbol)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench_prices) if (prices and bench_prices) else None
        stk_results.append({
            "name":         name,
            "symbol":       row["Symbol"],
            "yahoo_symbol": symbol,
            "industry":     industry,
            "isin":         isin,
            "slope_data":   result,
            "high52_data":  h52,
            "beta":         beta,
            "signal_color": color_signal(result)
        })

    # 4. Save JSON
    output = {
        "meta": {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "length":        LENGTH,
            "days_in_year":  DAYS_IN_YEAR,
            "beta_window":   BETA_WINDOW,
            "r2_threshold":  R2_THRESHOLD,
            "with_rsquared": WITH_RSQUARED,
            "with_annualize":WITH_ANNUALIZE,
            "benchmark":     BENCHMARK,
            "total_indices": len(idx_results),
            "total_stocks":  len(stk_results)
        },
        "indices": idx_results,
        "stocks":  stk_results
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Saved {OUTPUT_JSON}")
    print(f"    Indices : {len(idx_results)}")
    print(f"    Stocks  : {len(stk_results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
