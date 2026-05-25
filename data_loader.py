"""Price and rate data loaders for the FIRE/SWP simulator.

All historical data — equity index monthly closes, CPI, FD rates, tax rates —
comes from static CSVs in `static_data/`. The equity CSV is pre-fetched
locally via `tools/refresh_equity_data.py`; this avoids calling yfinance at
runtime, which is blocked from cloud datacenter IPs (Streamlit Cloud etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

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
    "NIFTY500":  AssetMeta("NIFTY500",  "Equity MF (Nifty 500 proxy)", "^CRSLDX", date(2005, 9, 1)),
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


@st.cache_data(show_spinner=False)
def _load_equity_panel() -> pd.DataFrame:
    """Load the bundled monthly-close CSV (one column per asset key)."""
    df = pd.read_csv(STATIC_DIR / "equity_prices.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def get_equity_panel(asset_keys: list[str], start: date, end: date) -> pd.DataFrame:
    """Return a DataFrame of monthly closes for the requested equity assets.

    Reads from static_data/equity_prices.csv. Run tools/refresh_equity_data.py
    locally to update that file with the latest monthly closes.
    """
    equity = [k for k in asset_keys if ASSETS[k].yf_ticker is not None]
    if not equity:
        return pd.DataFrame()

    full = _load_equity_panel()
    missing = [k for k in equity if k not in full.columns]
    if missing:
        raise RuntimeError(
            f"static_data/equity_prices.csv is missing columns: {missing}. "
            "Run tools/refresh_equity_data.py to regenerate it."
        )

    panel = full[equity].copy()
    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    return panel.loc[mask].dropna(how="any")


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
