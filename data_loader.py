"""Price and rate data loaders for the FIRE/SWP simulator.

Equity indices come from yfinance (monthly close, cached for 1 day).
CPI and FD rates come from static CSVs in `static_data/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

STATIC_DIR = Path(__file__).parent / "static_data"


@dataclass(frozen=True)
class AssetMeta:
    key: str
    label: str
    yf_ticker: str | None  # None for synthetic assets (FD, DEBT)
    earliest: date         # earliest start date the UI should allow


ASSETS: dict[str, AssetMeta] = {
    # earliest = realistic floor on yfinance-served data (the underlying indices have
    # longer history, but Yahoo's coverage is narrower — see static_data/README.md).
    "NIFTY50":   AssetMeta("NIFTY50",   "Nifty 50",                  "^NSEI",    date(2007, 9, 1)),
    "SENSEX":    AssetMeta("SENSEX",    "Sensex",                    "^BSESN",   date(1997, 7, 1)),
    "BANKNIFTY": AssetMeta("BANKNIFTY", "Bank Nifty",                "^NSEBANK", date(2007, 9, 1)),
    "NIFTY500":  AssetMeta("NIFTY500",  "Equity MF (Nifty 500 proxy)", "^CRSLDX", date(2007, 9, 1)),
    "DEBT":      AssetMeta("DEBT",      "Debt Funds (flat 7% proxy)", None,      date(1979, 1, 1)),
    "FD":        AssetMeta("FD",        "Fixed Deposit (SBI 1-yr)",  None,      date(1979, 1, 1)),
}

EQUITY_KEYS = [k for k, m in ASSETS.items() if m.yf_ticker is not None]
SYNTHETIC_KEYS = [k for k, m in ASSETS.items() if m.yf_ticker is None]

DEBT_ANNUAL_RETURN = 0.07  # documented v1 simplification

ASSET_TAX_CATEGORY: dict[str, str] = {
    "NIFTY50":   "equity",
    "SENSEX":    "equity",
    "BANKNIFTY": "equity",
    "NIFTY500":  "equity",
    "DEBT":      "debt",
    "FD":        "fd",
}


@dataclass(frozen=True)
class TaxRates:
    effective_date: pd.Timestamp
    equity_ltcg_rate: float
    equity_ltcg_exempt_inr: float
    debt_ltcg_rate: float
    fd_rate: float


@st.cache_data(ttl=86400, show_spinner=False)
def _download_monthly_close(ticker: str, start: date, end: date) -> pd.Series:
    """Fetch monthly-close series for a single yfinance ticker. Cached for 1 day."""
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1mo",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype="float64", name=ticker)

    # yfinance may return a MultiIndex on columns when given a single ticker in some versions.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna().astype("float64")
    close.index = pd.to_datetime(close.index).to_period("M").to_timestamp("M")
    close.name = ticker
    # Collapse any duplicate month-end rows yfinance occasionally emits.
    return close.groupby(close.index).last()


def get_equity_panel(asset_keys: list[str], start: date, end: date) -> pd.DataFrame:
    """Return a DataFrame of monthly closes for the requested equity assets.

    Columns are asset keys (not yfinance tickers). Index is month-end timestamps.
    Months where any selected series is missing are dropped.
    """
    equity = [k for k in asset_keys if ASSETS[k].yf_ticker is not None]
    if not equity:
        return pd.DataFrame()

    series_by_key: dict[str, pd.Series] = {}
    for key in equity:
        s = _download_monthly_close(ASSETS[key].yf_ticker, start, end)
        if s.empty:
            raise RuntimeError(
                f"No data returned from yfinance for {key} ({ASSETS[key].yf_ticker}). "
                "Try a later start date or check your internet connection."
            )
        series_by_key[key] = s

    panel = pd.concat(series_by_key, axis=1).sort_index()
    return panel.dropna(how="any")


@st.cache_data(show_spinner=False)
def load_cpi() -> pd.Series:
    df = pd.read_csv(STATIC_DIR / "cpi_india.csv")
    s = df.set_index("year")["cpi_yoy_pct"].astype("float64") / 100.0
    s.index = s.index.astype(int)
    return s.sort_index()


@st.cache_data(show_spinner=False)
def load_fd_rates() -> pd.Series:
    df = pd.read_csv(STATIC_DIR / "fd_rates.csv")
    s = df.set_index("year")["sbi_1yr_fd_pct"].astype("float64") / 100.0
    s.index = s.index.astype(int)
    return s.sort_index()


@st.cache_data(show_spinner=False)
def load_tax_rates() -> list[TaxRates]:
    df = pd.read_csv(STATIC_DIR / "tax_rates.csv", parse_dates=["effective_date"])
    df = df.sort_values("effective_date").reset_index(drop=True)
    return [
        TaxRates(
            effective_date=pd.Timestamp(row.effective_date),
            equity_ltcg_rate=float(row.equity_ltcg_rate),
            equity_ltcg_exempt_inr=float(row.equity_ltcg_exempt_inr),
            debt_ltcg_rate=float(row.debt_ltcg_rate),
            fd_rate=float(row.fd_rate),
        )
        for row in df.itertuples()
    ]


def tax_rates_on(rates: list[TaxRates], ts: pd.Timestamp) -> TaxRates:
    """Return the tax-rate row in effect on `ts` (latest row whose effective_date <= ts)."""
    applicable = [r for r in rates if r.effective_date <= ts]
    return applicable[-1] if applicable else rates[0]


def annual_rate_for(series: pd.Series, year: int) -> float:
    """Look up an annual rate, forward-filling from the most recent known year."""
    if year in series.index:
        return float(series.loc[year])
    known = series.index[series.index <= year]
    if len(known) == 0:
        return float(series.iloc[0])
    return float(series.loc[known.max()])
