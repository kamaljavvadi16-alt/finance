"""Streamlit UI for the FIRE / SWP simulator (India).

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    ASSETS,
    EQUITY_KEYS,
    get_equity_panel,
    load_cpi,
    load_fd_rates,
    load_tax_rates,
)
from simulator import simulate_swp

st.set_page_config(page_title="FIRE / SWP Simulator — India", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("FIRE / SWP Simulator — India")
st.caption(
    "Back-test a Systematic Withdrawal Plan against historical Indian market data. "
    "Find the sweet spot between corpus, allocation, and monthly burn."
)

def format_inr(x: float) -> str:
    """Format a number as rupees with Indian comma grouping (e.g. ₹1,00,00,000)."""
    n = int(round(x))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return f"₹{sign}{s}"
    last3, rest = s[-3:], s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return f"₹{sign}{','.join(groups)},{last3}"


_ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
         "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
         "Seventeen", "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]


def _words_below_crore(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ((" " + _ONES[o]) if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " Hundred" + ((" " + _words_below_crore(r)) if r else "")
    if n < 100_000:
        t, r = divmod(n, 1000)
        return _words_below_crore(t) + " Thousand" + ((" " + _words_below_crore(r)) if r else "")
    l, r = divmod(n, 100_000)
    return _words_below_crore(l) + " Lakh" + ((" " + _words_below_crore(r)) if r else "")


def num_to_words_inr(n: int) -> str:
    if n == 0:
        return "Zero Rupees"
    if n < 0:
        return "Minus " + num_to_words_inr(-n)
    if n < 10_000_000:
        return _words_below_crore(n) + " Rupees"
    c, r = divmod(n, 10_000_000)
    tail = (" " + _words_below_crore(r)) if r else ""
    return _words_below_crore(c) + " Crore" + tail + " Rupees"


def indian_ticks(max_val: float) -> tuple[list[float], list[str]]:
    """Pick ~6-8 nice round Indian-style tick values up to max_val and format them."""
    if max_val is None or max_val <= 0:
        return [0], ["₹0"]
    target = max_val / 6
    # Candidate step sizes in rupees, ascending. Denser than before so small corpora
    # (lakhs) get readable ticks too.
    candidates = [
        100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000,
        1_00_000, 2_00_000, 5_00_000, 10_00_000, 20_00_000, 50_00_000,
        1_00_00_000, 2_00_00_000, 5_00_00_000, 10_00_00_000, 20_00_00_000, 50_00_00_000,
        1_00_00_00_000, 2_00_00_00_000, 5_00_00_00_000, 10_00_00_00_000,
    ]
    step = next((c for c in candidates if c >= target), candidates[-1])
    vals: list[float] = []
    v = 0.0
    # Loop until one step beyond max so plotly's auto-range still has an upper tick.
    while v < max_val:
        vals.append(v)
        v += step
    vals.append(v)
    return vals, [format_inr(x) for x in vals]


def format_years_months(months: int) -> str:
    y, m = divmod(months, 12)
    if y == 0:
        return f"{m} month{'s' if m != 1 else ''}"
    if m == 0:
        return f"{y} year{'s' if y != 1 else ''}"
    return f"{y}y {m}m"


with st.sidebar:
    st.header("Inputs")

    corpus_words_slot = st.empty()
    corpus = st.number_input(
        "Corpus (₹)",
        min_value=10_000,
        max_value=10_00_00_00_000,
        value=1_00_00_000,
        step=1_00_000,
        help="Initial lump-sum to invest, in rupees.",
    )
    corpus_words_slot.caption(f"*{num_to_words_inr(int(corpus))} ({format_inr(int(corpus))})*")

    monthly_expense = st.number_input(
        "Monthly expense (₹)",
        min_value=1000,
        max_value=100_00_000,
        value=50_000,
        step=5_000,
        help="Starting monthly withdrawal. Grows annually with India CPI.",
    )

    st.subheader("Allocation (%)")
    st.caption("Sliders must sum to 100%.")

    cols = st.columns(2)
    alloc_pct: dict[str, int] = {}
    default_alloc = {"NIFTY50": 60, "SENSEX": 0, "BANKNIFTY": 0, "NIFTY500": 0, "DEBT": 20, "FD": 20}
    for i, (key, meta) in enumerate(ASSETS.items()):
        with cols[i % 2]:
            alloc_pct[key] = st.slider(
                meta.label,
                min_value=0, max_value=100, value=default_alloc.get(key, 0), step=5,
                key=f"alloc_{key}",
            )

    total_pct = sum(alloc_pct.values())
    if total_pct == 100:
        st.success(f"Total: {total_pct}%")
    else:
        st.error(f"Total: {total_pct}% (must be 100%)")

    selected_equity = [k for k in EQUITY_KEYS if alloc_pct[k] > 0]
    min_start = max((ASSETS[k].earliest for k in selected_equity), default=date(1979, 1, 1))
    max_start = date.today() - timedelta(days=365)

    default_start = max(min_start, date(2005, 1, 1))
    default_start = min(default_start, max_start)

    start_date = st.date_input(
        "Start date",
        value=default_start,
        min_value=min_start,
        max_value=max_start,
        help="Earliest allowed date depends on selected equity assets.",
    )
    if selected_equity:
        constraining = max(selected_equity, key=lambda k: ASSETS[k].earliest)
        st.caption(f"Earliest allowed: {min_start.isoformat()} (set by {ASSETS[constraining].label})")

    rebalance = st.radio("Rebalance", options=["annual", "never"], index=0, horizontal=True)

    run = st.button("Run simulation", type="primary", use_container_width=True, disabled=(total_pct != 100))


if not run:
    st.info("Set your inputs in the sidebar and click **Run simulation**.")
    with st.expander("How this works"):
        st.markdown(
            """
            - Your corpus is split across the chosen assets at the start date.
            - Each month: assets earn their historical return (or the FD/debt rate), then you withdraw the inflated monthly expense.
            - The expense grows yearly by India CPI.
            - Equity returns come from Yahoo Finance (yfinance). FD rates are SBI 1-year. Debt is a flat 7% proxy for v1.
            - The output shows how long your corpus lasted, plus risk metrics like max drawdown and worst-year return.
            """
        )
    st.stop()


with st.spinner("Fetching historical data and running simulation…"):
    allocation = {k: v / 100.0 for k, v in alloc_pct.items() if v > 0}

    try:
        equity_panel = (
            get_equity_panel(selected_equity, start_date, date.today())
            if selected_equity else pd.DataFrame()
        )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    cpi = load_cpi()
    fd_rates = load_fd_rates()
    tax_rates = load_tax_rates()

    try:
        result = simulate_swp(
            corpus=corpus,
            start_date=start_date,
            allocation=allocation,
            monthly_expense=float(monthly_expense),
            cpi_series=cpi,
            price_panel=equity_panel,
            fd_rate_series=fd_rates,
            tax_rates=tax_rates,
            rebalance=rebalance,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()


k1, k2, k3, k4, k5 = st.columns(5)
if result.survived:
    k1.metric("Corpus lasted", format_years_months(result.months_lasted), help="Survived to today — corpus never depleted.")
else:
    k1.metric("Corpus lasted", format_years_months(result.months_lasted), help=f"Depleted: {result.depletion_month.date()}", delta="depleted", delta_color="inverse")
k2.metric("Peak corpus", format_inr(result.peak_corpus))
k3.metric("End corpus", format_inr(result.end_corpus))
k4.metric("Max drawdown", f"{result.max_drawdown*100:.1f}%")
k5.metric("Annualized return", f"{result.annualized_return*100:.2f}%")

st.divider()

c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("Corpus over time")
    log_y = st.toggle("Log scale", value=False, key="log_corpus")
    fig = px.line(
        result.monthly_corpus.reset_index().rename(columns={"index": "date", "corpus": "Corpus (₹)"}),
        x="date", y="Corpus (₹)",
    )
    if log_y:
        fig.update_yaxes(type="log")
    else:
        tv, tt = indian_ticks(float(result.monthly_corpus.max()))
        fig.update_yaxes(tickvals=tv, ticktext=tt)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    if result.depletion_month is not None:
        dep_x = result.depletion_month.isoformat()
        fig.add_vline(x=dep_x, line_dash="dash", line_color="red")
        fig.add_annotation(x=dep_x, y=1, yref="paper", showarrow=False,
                           text="Depleted", font=dict(color="red"),
                           xanchor="left", yanchor="bottom")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Risk")
    risk_rows = [
        ("Total withdrawn",         format_inr(float(result.monthly_withdrawal.sum()))),
        ("Total tax paid",          format_inr(result.total_tax_paid)),
        ("Annualized volatility",   f"{result.annualized_volatility*100:.2f}%"),
        ("Average inflation (CPI)", f"{result.average_inflation*100:.2f}%"),
        ("Max drawdown",            f"{result.max_drawdown*100:.2f}%"),
        ("Worst calendar year",     f"{result.worst_calendar_year_return*100:.2f}%"),
        ("Longest drawdown",        f"{result.longest_drawdown_months} months"),
        ("Survived to today",       "yes" if result.survived else "no"),
    ]
    st.table(pd.DataFrame(risk_rows, columns=["Metric", "Value"]).set_index("Metric"))

st.subheader("Asset composition over time")
comp = result.per_asset_corpus.copy()
comp.columns = [ASSETS[c].label for c in comp.columns]
comp_long = comp.reset_index().melt(id_vars="index", var_name="Asset", value_name="Corpus (₹)")
comp_long = comp_long.rename(columns={"index": "date"})
fig2 = px.area(comp_long, x="date", y="Corpus (₹)", color="Asset", groupnorm=None)
tv2, tt2 = indian_ticks(float(comp.sum(axis=1).max()))
fig2.update_yaxes(tickvals=tv2, ticktext=tt2)
fig2.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Monthly withdrawal (inflation-adjusted)")
monthly_wd = result.monthly_withdrawal.iloc[1:]  # drop the t=0 zero row
wd_fig = go.Figure()
wd_fig.add_trace(go.Scatter(
    x=monthly_wd.index, y=monthly_wd.values,
    mode="lines", name="Monthly withdrawal (₹)",
    line=dict(width=2),
    hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra></extra>",
))
tv3, tt3 = indian_ticks(float(monthly_wd.max()) if len(monthly_wd) else 0)
wd_fig.update_yaxes(tickvals=tv3, ticktext=tt3, title="Withdrawal (₹)")
wd_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(wd_fig, use_container_width=True)

st.subheader("Annual portfolio return")
annual_return = (1.0 + result.monthly_return.iloc[1:]).groupby(result.monthly_return.iloc[1:].index.year).prod() - 1.0
ret_fig = go.Figure()
ret_fig.add_bar(x=list(annual_return.index), y=[v * 100 for v in annual_return.values],
                marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in annual_return.values])
ret_fig.update_yaxes(title="Return (%)")
ret_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(ret_fig, use_container_width=True)

with st.expander("Show raw monthly data"):
    table = pd.DataFrame({
        "Corpus (₹)": result.monthly_corpus,
        "Withdrawal (₹)": result.monthly_withdrawal,
        "Tax (₹)": result.monthly_tax,
        "Return (%)": (result.monthly_return * 100).round(2),
    })
    styled = table.style.format({
        "Corpus (₹)": format_inr,
        "Withdrawal (₹)": format_inr,
        "Tax (₹)": format_inr,
        "Return (%)": "{:.2f}%",
    })
    st.dataframe(styled, use_container_width=True)
