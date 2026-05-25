"""Download monthly closes for the four equity indices and save them to a CSV
so the deployed app does not need to call yfinance at runtime.

yfinance is rate-limited and blocked from most cloud datacenter IPs, so the
Streamlit Cloud deploy fails when it tries to fetch live. We fetch locally
(where yfinance works) and commit the CSV.

Run me when you want to refresh the historical data:

    python tools/refresh_equity_data.py

Then `git add static_data/equity_prices.csv && git commit && git push`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "static_data" / "equity_prices.csv"

TICKERS: dict[str, str] = {
    "NIFTY50":   "^NSEI",
    "SENSEX":    "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY500":  "^CRSLDX",
}

START = "1995-01-01"
END = date.today().isoformat()


def fetch(ticker: str) -> pd.Series:
    df = yf.download(
        ticker, start=START, end=END, interval="1mo",
        progress=False, auto_adjust=True, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna().astype("float64")
    close.index = pd.to_datetime(close.index).to_period("M").to_timestamp("M")
    return close.groupby(close.index).last()


def main() -> None:
    series: dict[str, pd.Series] = {}
    for key, ticker in TICKERS.items():
        print(f"Fetching {key} ({ticker}) ...", flush=True)
        s = fetch(ticker)
        print(f"  -> {len(s)} rows, {s.index.min().date()} - {s.index.max().date()}")
        series[key] = s

    panel = pd.concat(series, axis=1).sort_index()
    panel.index.name = "date"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, float_format="%.4f")
    print(f"\nWrote {len(panel)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
