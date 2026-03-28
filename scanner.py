import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ── CONFIG ──────────────────────────────────────────────────────────────────
LENGTH        = 90
DAYS_IN_YEAR  = 252
BETA_WINDOW   = 252
R2_THRESHOLD  = 0.0
WITH_RSQUARED = True
WITH_ANNUALIZE= True
BENCHMARK     = "^NSEI"
DATA_DIR      = "data"
INDICES_CSV   = os.path.join(DATA_DIR, "indices.csv")
STOCKS_CSV    = os.path.join(DATA_DIR, "ind_niftytotalmarket_list.csv")
OUTPUT_JSON   = os.path.join(DATA_DIR, "results.json")

# date range: 2.5 years back to today
END_DATE   = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=900)

# ── MATH ────────────────────────────────────────────────────────────────────
def compute_exp_slope(prices):
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < LENGTH:
        return None
    sl = arr[-LENGTH:]
    log_p = np.log(sl)
    x = np.arange(len(log_p), dtype=float)
    mx, my = x.mean(), log_p.mean()
    sx, sy = x.std(), log_p.std()
    if sx == 0 or sy == 0:
        return None
    c = np.corrcoef(x, log_p)[0, 1]
    slope = c * (sy / sx)
    annualized = (np.exp(slope) ** DAYS_IN_YEAR - 1) * 100
    cum_x = np.arange(1, len(log_p)+1, dtype=float)
    r2_raw = np.corrcoef(cum_x, log_p)[0, 1] ** 2
    r2 = float(r2_raw) if not np.isnan(r2_raw) else 0.0
    base  = annualized if WITH_ANNUALIZE else slope
    final = base * (r2 if WITH_RSQUARED else 1.0)
    return {
        "slope": round(float(final), 4),
        "raw_slope": round(float(annualized), 4),
        "r2": round(r2, 4),
        "above_threshold": r2 > R2_THRESHOLD
    }

def compute_52w(prices):
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    window = arr[-252:] if len(arr) >= 252 else arr
    high52  = float(window.max())
    current = float(arr[-1])
    pct = round((current - high52) / high52 * 100, 2)
    return {"current": round(current, 2), "high52": round(high52, 2), "pct_from_high": pct}

def compute_beta(stock_prices, bench_prices):
    sp = np.array(stock_prices, dtype=float)
    bp = np.array(bench_prices, dtype=float)
    n  = min(len(sp), len(bp), BETA_WINDOW)
    if n < 20:
        return None
    sp, bp = sp[-n:], bp[-n:]
    mask = ~(np.isnan(sp) | np.isnan(bp))
    sp, bp = sp[mask], bp[mask]
    if len(sp) < 15:
        return None
    sr = np.diff(sp) / sp[:-1]
    br = np.diff(bp) / bp[:-1]
    if br.std() == 0:
        return None
    cov   = np.cov(sr, br)[0, 1]
    var_b = np.var(br, ddof=1)
    return round(float(cov / var_b), 3) if var_b != 0 else None

def color_signal(result):
    if not result or not result["above_threshold"]:
        return "gray"
    up = result["slope"] > 0
    # use raw_slope direction vs slope as proxy for rising
    rising = result["slope"] >= 0 if up else result["slope"] <= 0
    if up:    return "bright_green"
    return "bright_red"

# ── ROBUST FETCH ─────────────────────────────────────────────────────────────
def fetch_closes(symbol):
    """Try multiple methods to get close prices."""
    closes = None

    # Method 1: yf.download with date range (best for .NS indices)
    try:
        df = yf.download(
            symbol,
            start=str(START_DATE),
            end=str(END_DATE),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if not df.empty and "Close" in df.columns:
            c = df["Close"].dropna()
            if hasattr(c, 'iloc') and len(c) >= LENGTH:
                # handle MultiIndex columns from yf.download
                if isinstance(c.columns if hasattr(c, 'columns') else None, pd.MultiIndex):
                    c = c.iloc[:, 0]
                closes = c.tolist()
    except Exception as e:
        pass

    # Method 2: Ticker.history fallback
    if not closes or len(closes) < LENGTH:
        try:
            tk   = yf.Ticker(symbol)
            hist = tk.history(start=str(START_DATE), end=str(END_DATE), auto_adjust=True)
            if not hist.empty and "Close" in hist.columns:
                c = hist["Close"].dropna().tolist()
                if len(c) >= LENGTH:
                    closes = c
        except Exception as e:
            pass

    # Method 3: period fallback
    if not closes or len(closes) < LENGTH:
        try:
            tk   = yf.Ticker(symbol)
            hist = tk.history(period="2y", auto_adjust=True)
            if not hist.empty and "Close" in hist.columns:
                c = hist["Close"].dropna().tolist()
                if len(c) >= LENGTH:
                    closes = c
        except Exception as e:
            pass

    if closes and len(closes) >= LENGTH:
        return closes
    print(f"  WARN: {symbol} — not enough data ({len(closes) if closes else 0} bars)")
    return None

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"EXP Scanner  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Date range   :  {START_DATE} → {END_DATE}")
    print("=" * 60)

    # Benchmark
    print(f"\nFetching benchmark {BENCHMARK} ...")
    bench_prices = fetch_closes(BENCHMARK)
    if bench_prices is None:
        print("WARNING: benchmark failed — beta will be null")

    # ── INDICES ─────────────────────────────────────────────────────────────
    print("\n── INDICES ─────────────────────────────────────────────")
    idx_df = pd.read_csv(INDICES_CSV)
    idx_df.columns = [c.strip() for c in idx_df.columns]
    name_col   = idx_df.columns[0]
    symbol_col = idx_df.columns[1] if len(idx_df.columns) > 1 else None

    idx_results = []
    for _, row in idx_df.iterrows():
        name   = str(row[name_col]).strip()
        symbol = str(row[symbol_col]).strip() if symbol_col else name
        if name.lower() in ("indices", "index", ""):
            continue
        print(f"  {symbol:<35} {name}")
        prices = fetch_closes(symbol)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench_prices) if (prices and bench_prices) else None
        idx_results.append({
            "name": name, "symbol": symbol,
            "slope_data": result, "high52_data": h52,
            "beta": beta, "signal_color": color_signal(result)
        })
        status = f"slope={result['slope']:.1f}% r2={result['r2']:.2f}" if result else "NO DATA"
        print(f"    → {status}")

    # ── STOCKS ──────────────────────────────────────────────────────────────
    print("\n── STOCKS ──────────────────────────────────────────────")
    stk_df = pd.read_csv(STOCKS_CSV)
    stk_df.columns = [c.strip() for c in stk_df.columns]
    stk_df = stk_df[stk_df["Series"].str.strip() == "EQ"].copy()
    stk_df = stk_df.dropna(subset=["Symbol"])
    stk_df["Symbol"] = stk_df["Symbol"].str.strip()

    stk_results = []
    total = len(stk_df)
    for i, (_, row) in enumerate(stk_df.iterrows(), 1):
        symbol   = row["Symbol"] + ".NS"
        name     = row["Company Name"].strip()
        industry = row.get("Industry", "").strip()
        isin     = row.get("ISIN Code", "").strip()
        print(f"  [{i:3d}/{total}] {symbol:<22} {name[:35]}")
        prices = fetch_closes(symbol)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench_prices) if (prices and bench_prices) else None
        stk_results.append({
            "name": name, "symbol": row["Symbol"],
            "yahoo_symbol": symbol, "industry": industry, "isin": isin,
            "slope_data": result, "high52_data": h52,
            "beta": beta, "signal_color": color_signal(result)
        })

    # ── SAVE ────────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "generated_at":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "length":         LENGTH,
            "days_in_year":   DAYS_IN_YEAR,
            "beta_window":    BETA_WINDOW,
            "r2_threshold":   R2_THRESHOLD,
            "with_rsquared":  WITH_RSQUARED,
            "with_annualize": WITH_ANNUALIZE,
            "benchmark":      BENCHMARK,
            "total_indices":  len(idx_results),
            "total_stocks":   len(stk_results)
        },
        "indices": idx_results,
        "stocks":  stk_results
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    ok_idx  = sum(1 for r in idx_results if r["slope_data"])
    ok_stk  = sum(1 for r in stk_results if r["slope_data"])
    print(f"\n✅  Saved {OUTPUT_JSON}")
    print(f"    Indices : {ok_idx}/{len(idx_results)} loaded")
    print(f"    Stocks  : {ok_stk}/{len(stk_results)} loaded")
    print("=" * 60)

if __name__ == "__main__":
    main()
