import argparse
import time
from pathlib import Path

import pandas as pd
import requests


# ---------- Binance Spot (klines) ----------

def fetch_binance_klines(symbol: str, interval: str = "15m", limit: int = 1000, start_time=None, end_time=None) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time * 1000)
    if end_time is not None:
        params["endTime"] = int(end_time * 1000)

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_vol", "taker_quote_vol", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]


def fetch_last_n_years(symbol: str, years: int, interval: str = "15m") -> pd.DataFrame:
    interval_seconds = 15 * 60
    end_time = time.time()
    start_time = end_time - (years * 365 * 24 * 60 * 60)

    chunks = []
    while start_time < end_time:
        df = fetch_binance_klines(symbol=symbol, interval=interval, start_time=start_time, limit=1000)
        if df.empty:
            break
        chunks.append(df)
        start_time = df["timestamp"].iloc[-1].timestamp() + interval_seconds
        time.sleep(0.2)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "close"])

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


# ---------- Binance Futures (funding) ----------

def fetch_binance_funding_rate(symbol: str, start_time=None, end_time=None, limit: int = 1000) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time * 1000)
    if end_time is not None:
        params["endTime"] = int(end_time * 1000)

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])

    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "fundingRate"]].sort_values("timestamp").reset_index(drop=True)


def fetch_funding_for_years(symbol: str, years: int) -> pd.DataFrame:
    interval_seconds = 8 * 60 * 60  # funding every 8h
    end_time = time.time()
    start_time = end_time - years * 365 * 24 * 60 * 60

    chunks = []
    while start_time < end_time:
        df = fetch_binance_funding_rate(symbol, start_time=start_time, limit=1000)
        if df.empty:
            break
        chunks.append(df)
        start_time = df["timestamp"].iloc[-1].timestamp() + interval_seconds
        time.sleep(0.2)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()
    return df.resample("15min").ffill().reset_index()


# ---------- Main pipeline ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=6)
    p.add_argument("--start-date", type=str, default="2021-01-01 00:00:00")
    p.add_argument("--outdir", type=str, default="data")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    token1 = "ETHUSDT"
    token2 = "BTCUSDT"

    print("Downloading spot data...")
    t1 = fetch_last_n_years(token1, years=args.years)
    t2 = fetch_last_n_years(token2, years=args.years)

    # align start
    min_ts = max(t1["timestamp"].min(), t2["timestamp"].min())
    t1 = t1[t1["timestamp"] >= min_ts].reset_index(drop=True)
    t2 = t2[t2["timestamp"] >= min_ts].reset_index(drop=True)

    merged = pd.merge(t1, t2, on="timestamp", suffixes=(f"_{token1.lower()}", f"_{token2.lower()}"))
    pair = merged[["timestamp"]].copy()
    pair["close"] = merged[f"close_{token1.lower()}"] / merged[f"close_{token2.lower()}"]

    print("Downloading funding data...")
    f1 = resample_to_15min(fetch_funding_for_years(token1, years=args.years))
    f2 = resample_to_15min(fetch_funding_for_years(token2, years=args.years))

    # filter
    start = pd.to_datetime(args.start_date, utc=True)
    t1 = t1[t1["timestamp"] >= start]
    t2 = t2[t2["timestamp"] >= start]
    pair = pair[pair["timestamp"] >= start]
    f1 = f1[f1["timestamp"] >= start]
    f2 = f2[f2["timestamp"] >= start]

    # save
    t1.to_csv(outdir / "ethusdt_15min.csv", index=False)
    t2.to_csv(outdir / "btcusdt_15min.csv", index=False)
    pair.to_csv(outdir / "ethbtc_15min.csv", index=False)
    f1.to_csv(outdir / "eth_funding_15min.csv", index=False)
    f2.to_csv(outdir / "btc_funding_15min.csv", index=False)

    print(f"Saved CSVs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()