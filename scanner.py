import yfinance as yf
import pandas as pd
import numpy as np
import json, os, time, requests
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

END_DATE   = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=900)

# NSE index name map  (Yahoo .NS symbol → NSE index name for niftyindices API)
NSE_INDEX_MAP = {
    "NIFTYSMLCAP250.NS":    "NIFTY SMLCAP 250",
    "NIFTYMIDSML400.NS":    "NIFTY MIDSML 400",
    "NIFTY_CPSE.NS":        "NIFTY CPSE",
    "NIFTYPVTBANK.NS":      "NIFTY PRIVATE BANK",
    "NIFTY_GS_COMPSITE.NS": "NIFTY GS COMPSITE",
    "NIFTY_HEALTHCARE.NS":  "NIFTY HEALTHCARE INDEX",
    "NIFTY_CONSR_DURBL.NS": "NIFTY INDIA CONSUMPTION",
    "NIFTY_OIL_AND_GAS.NS": "NIFTY OIL AND GAS",
    "NIFTY_TOTAL_MKT.NS":   "NIFTY TOTAL MARKET",
    "NIFTY_MICROCAP250.NS": "NIFTY MICROCAP 250",
    "NIFTY_IND_DIGITAL.NS": "NIFTY INDIA DIGITAL",
    "NIFTY_TATA_25_CAP.NS": "NIFTY TATA 25 CAP",
    "NIFTY_IND_DEFENCE.NS": "NIFTY INDIA DEFENCE",
    "NIFTY_IND_TOURISM.NS": "NIFTY INDIA TOURISM",
    "NIFTY_CAPITAL_MKT.NS": "NIFTY CAPITAL MARKETS",
    "NIFTY_EV.NS":          "NIFTY EV AND NEW AGE AUTOMOTIVE",
    "NIFTY_HOUSING.NS":     "NIFTY HOUSING",
    "NIFTY_IPO.NS":         "NIFTY IPO INDEX",
    "NIFTY_RURAL.NS":       "NIFTY INDIA RURAL",
    "NIFTY_INTERNET.NS":    "NIFTY INDIA INTERNET",
    "NIFTY_CHEMICALS.NS":   "NIFTY CHEMICALS",
    "NIFTY_WAVES.NS":       "NIFTY WAVES",
    "NIFTY_RAILWAYSPSU.NS": "NIFTY INDIA RAILWAYS PSU",
    "NIFTYCONGLOMERATE.NS": "NIFTY CONGLOMERATE",
}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.niftyindices.com/",
    "X-Requested-With": "XMLHttpRequest",
}

# ── MATH ────────────────────────────────────────────────────────────────────
def compute_exp_slope(prices):
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < LENGTH:
        return None
    sl    = arr[-LENGTH:]
    log_p = np.log(sl)
    x     = np.arange(len(log_p), dtype=float)
    sx, sy = x.std(), log_p.std()
    if sx == 0 or sy == 0:
        return None
    c      = np.corrcoef(x, log_p)[0, 1]
    slope  = c * (sy / sx)
    ann    = (np.exp(slope) ** DAYS_IN_YEAR - 1) * 100
    cum_x  = np.arange(1, len(log_p)+1, dtype=float)
    r2_raw = np.corrcoef(cum_x, log_p)[0, 1] ** 2
    r2     = float(r2_raw) if not np.isnan(r2_raw) else 0.0
    base   = ann if WITH_ANNUALIZE else slope
    final  = base * (r2 if WITH_RSQUARED else 1.0)
    return {"slope": round(float(final),4), "raw_slope": round(float(ann),4),
            "r2": round(r2,4), "above_threshold": r2 > R2_THRESHOLD}

def compute_52w(prices):
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    w    = arr[-252:] if len(arr)>=252 else arr
    hi   = float(w.max())
    cur  = float(arr[-1])
    return {"current": round(cur,2), "high52": round(hi,2),
            "pct_from_high": round((cur-hi)/hi*100, 2)}

def compute_beta(sp, bp):
    sp = np.array(sp, dtype=float); bp = np.array(bp, dtype=float)
    n  = min(len(sp), len(bp), BETA_WINDOW)
    if n < 20: return None
    sp, bp = sp[-n:], bp[-n:]
    mask = ~(np.isnan(sp)|np.isnan(bp))
    sp, bp = sp[mask], bp[mask]
    if len(sp) < 15: return None
    sr = np.diff(sp)/sp[:-1]; br = np.diff(bp)/bp[:-1]
    if br.std()==0: return None
    cov = np.cov(sr,br)[0,1]; var_b = np.var(br, ddof=1)
    return round(float(cov/var_b),3) if var_b!=0 else None

def color_signal(r):
    if not r or not r["above_threshold"]: return "gray"
    return "bright_green" if r["slope"]>0 else "bright_red"

# ── FETCH: Yahoo Finance ─────────────────────────────────────────────────────
def fetch_yahoo(symbol):
    for attempt in range(2):
        try:
            df = yf.download(symbol, start=str(START_DATE), end=str(END_DATE),
                             interval="1d", auto_adjust=True, progress=False, threads=False)
            if not df.empty and "Close" in df.columns:
                c = df["Close"].dropna()
                if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                vals = c.tolist()
                if len(vals) >= LENGTH:
                    return vals
        except Exception:
            pass
        try:
            tk = yf.Ticker(symbol)
            h  = tk.history(start=str(START_DATE), end=str(END_DATE), auto_adjust=True)
            if not h.empty:
                vals = h["Close"].dropna().tolist()
                if len(vals) >= LENGTH:
                    return vals
        except Exception:
            pass
        if attempt == 0:
            time.sleep(2)
    return None

# ── FETCH: NSE Indices (niftyindices.com) ───────────────────────────────────
def fetch_nse_index(index_name):
    """Fetch index history from niftyindices.com API."""
    fmt = "%d-%m-%Y"
    start_str = START_DATE.strftime(fmt)
    end_str   = END_DATE.strftime(fmt)
    url = (f"https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
           f"?name={requests.utils.quote(index_name)}&startDate={start_str}&endDate={end_str}")
    try:
        r = requests.post(
            "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString",
            json={"name": index_name, "startDate": start_str, "endDate": end_str},
            headers={**NSE_HEADERS, "Content-Type": "application/json; charset=UTF-8"},
            timeout=20
        )
        if r.status_code != 200:
            return None
        payload = r.json()
        raw = payload.get("d", "")
        if not raw:
            return None
        df = pd.read_json(raw)
        # columns: HistoricalDate, OPEN, HIGH, LOW, CLOSE
        close_col = next((c for c in df.columns if "CLOSE" in c.upper() or "Close" in c), None)
        if not close_col:
            return None
        closes = df[close_col].dropna().tolist()
        closes = [float(x) for x in reversed(closes)]  # NSE returns newest first
        return closes if len(closes) >= LENGTH else None
    except Exception as e:
        print(f"    NSE fetch error for {index_name}: {e}")
        return None

# ── COMBINED FETCH ──────────────────────────────────────────────────────────
def fetch_closes(symbol):
    # First try Yahoo
    closes = fetch_yahoo(symbol)
    if closes:
        return closes
    # Fallback: NSE indices API
    if symbol in NSE_INDEX_MAP:
        idx_name = NSE_INDEX_MAP[symbol]
        print(f"    → Yahoo failed, trying NSE API for '{idx_name}'")
        closes = fetch_nse_index(idx_name)
        if closes:
            print(f"    → NSE API: {len(closes)} bars")
            return closes
    print(f"  WARN: {symbol} — no data from any source")
    return None

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print(f"EXP Scanner  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Date range   :  {START_DATE} → {END_DATE}")
    print("="*60)

    print(f"\nFetching benchmark {BENCHMARK} ...")
    bench = fetch_closes(BENCHMARK)

    # ── INDICES ─────────────────────────────────────────────────────────────
    print("\n── INDICES ─────────────────────────────────────────────")
    idx_df = pd.read_csv(INDICES_CSV)
    idx_df.columns = [c.strip() for c in idx_df.columns]
    name_col   = idx_df.columns[0]
    symbol_col = idx_df.columns[1] if len(idx_df.columns)>1 else None
    idx_results = []
    for _, row in idx_df.iterrows():
        name   = str(row[name_col]).strip()
        symbol = str(row[symbol_col]).strip() if symbol_col else name
        if name.lower() in ("indices","index",""): continue
        print(f"  {symbol:<35} {name}")
        prices = fetch_closes(symbol)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench) if (prices and bench) else None
        status = f"slope={result['slope']:.1f}% r2={result['r2']:.2f}" if result else "NO DATA"
        print(f"    → {status}")
        idx_results.append({"name":name,"symbol":symbol,"slope_data":result,
                             "high52_data":h52,"beta":beta,"signal_color":color_signal(result)})

    # ── STOCKS ──────────────────────────────────────────────────────────────
    print("\n── STOCKS ──────────────────────────────────────────────")
    stk_df = pd.read_csv(STOCKS_CSV)
    stk_df.columns = [c.strip() for c in stk_df.columns]
    stk_df = stk_df[stk_df["Series"].str.strip()=="EQ"].dropna(subset=["Symbol"]).copy()
    stk_df["Symbol"]       = stk_df["Symbol"].str.strip()
    stk_df["Industry"]     = stk_df["Industry"].str.strip()
    stk_df["Company Name"] = stk_df["Company Name"].str.strip()
    stk_results = []
    total = len(stk_df)
    for i,(_, row) in enumerate(stk_df.iterrows(),1):
        sym  = row["Symbol"]+".NS"
        name = row["Company Name"]
        print(f"  [{i:3d}/{total}] {sym:<22} {name[:35]}")
        prices = fetch_closes(sym)
        result = compute_exp_slope(prices) if prices else None
        h52    = compute_52w(prices)       if prices else None
        beta   = compute_beta(prices, bench) if (prices and bench) else None
        stk_results.append({"name":name,"symbol":row["Symbol"],"yahoo_symbol":sym,
                             "industry":row.get("Industry",""),"isin":row.get("ISIN Code",""),
                             "slope_data":result,"high52_data":h52,"beta":beta,
                             "signal_color":color_signal(result)})

    output = {"meta":{"generated_at":datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                      "length":LENGTH,"days_in_year":DAYS_IN_YEAR,"beta_window":BETA_WINDOW,
                      "r2_threshold":R2_THRESHOLD,"with_rsquared":WITH_RSQUARED,
                      "with_annualize":WITH_ANNUALIZE,"benchmark":BENCHMARK,
                      "total_indices":len(idx_results),"total_stocks":len(stk_results)},
              "indices":idx_results,"stocks":stk_results}
    os.makedirs(DATA_DIR,exist_ok=True)
    with open(OUTPUT_JSON,"w") as f:
        json.dump(output,f,indent=2)

    ok_i = sum(1 for r in idx_results if r["slope_data"])
    ok_s = sum(1 for r in stk_results if r["slope_data"])
    print(f"\n✅  Saved {OUTPUT_JSON}")
    print(f"    Indices : {ok_i}/{len(idx_results)} loaded")
    print(f"    Stocks  : {ok_s}/{len(stk_results)} loaded")
    print("="*60)

if __name__=="__main__":
    main()
