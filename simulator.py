"""Pure SWP simulation: given a corpus, allocation, and historical data,
back-test month-by-month and report how long the corpus lasted plus risk metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import (
    ASSETS,
    ASSET_TAX_CATEGORY,
    DEBT_ANNUAL_RETURN,
    TaxRates,
    annual_rate_for,
    tax_rates_on,
)


@dataclass
class SimulationResult:
    monthly_corpus: pd.Series          # total corpus at each month-end (after tax + withdrawals)
    per_asset_corpus: pd.DataFrame     # per-sleeve corpus, columns = asset keys
    monthly_withdrawal: pd.Series      # net withdrawal each month (cash to user)
    monthly_tax: pd.Series             # tax paid each month (non-zero only at FY-end / March)
    monthly_return: pd.Series          # pre-withdrawal portfolio return (asset-level, pre-tax)
    depletion_month: Optional[pd.Timestamp]
    survived: bool
    months_lasted: int                 # months elapsed since t=0
    peak_corpus: float
    end_corpus: float
    max_drawdown: float                # negative fraction, e.g. -0.34
    annualized_return: float           # geometric, decimal
    annualized_volatility: float       # stdev of monthly returns × sqrt(12)
    worst_calendar_year_return: float
    longest_drawdown_months: int
    average_inflation: float           # geometric-mean annualized CPI over the simulated months
    total_tax_paid: float              # cumulative tax (₹) deducted from corpus over the run
    inflation_index: pd.Series         # cumulative inflation factor since t=0 (starts at 1.0)
    effective_annual_return: float     # money-weighted IRR of user cashflows (after-tax)
    market_max_drawdown: float         # drawdown of a hypothetical ₹1 in this allocation (no withdrawals/tax)
    longest_market_drawdown_months: int


def _build_timeline(
    start_date: date,
    price_panel: pd.DataFrame,
    equity_keys: list[str],
) -> pd.DatetimeIndex:
    if equity_keys and not price_panel.empty:
        idx = price_panel.index[price_panel.index >= pd.Timestamp(start_date)]
        return pd.DatetimeIndex(idx)
    today = pd.Timestamp.today().normalize()
    return pd.date_range(start=pd.Timestamp(start_date), end=today, freq="ME")


def simulate_swp(
    corpus: float,
    start_date: date,
    allocation: dict[str, float],
    monthly_expense: float,
    cpi_series: pd.Series,
    price_panel: pd.DataFrame,
    fd_rate_series: pd.Series,
    tax_rates: list[TaxRates],
    rebalance: str = "annual",
) -> SimulationResult:
    alloc = {k: float(v) for k, v in allocation.items() if v > 0}
    total_alloc = sum(alloc.values())
    if abs(total_alloc - 1.0) > 1e-6:
        raise ValueError(f"Allocation must sum to 1.0, got {total_alloc:.4f}")

    equity_keys = [k for k in alloc if ASSETS[k].yf_ticker is not None]
    timeline = _build_timeline(start_date, price_panel, equity_keys)
    if len(timeline) < 2:
        raise ValueError("Timeline has fewer than 2 months — pick an earlier start date.")

    # Each sleeve tracks current market value + cost basis (₹) for capital-gain calc.
    sleeves: dict[str, dict[str, float]] = {
        k: {"value": corpus * w, "basis": corpus * w} for k, w in alloc.items()
    }
    per_asset_rows: list[dict[str, float]] = [{k: s["value"] for k, s in sleeves.items()}]
    total_rows: list[float] = [sum(s["value"] for s in sleeves.values())]
    withdrawal_rows: list[float] = [0.0]
    return_rows: list[float] = [0.0]
    tax_rows: list[float] = [0.0]

    expense = monthly_expense
    depletion_month: Optional[pd.Timestamp] = None
    inflation_factor = 1.0
    inflation_months = 0

    realized_equity_fy = 0.0
    realized_debt_fy = 0.0
    fd_interest_fy = 0.0
    total_tax_paid = 0.0

    def _withdraw_proportional(amount: float, track_realized: bool) -> float:
        """Withdraw `amount` from all sleeves proportionally to their current value.
        If track_realized, attribute realized gains to the appropriate FY bucket.
        Returns the actual amount withdrawn (may be less if corpus is short)."""
        nonlocal realized_equity_fy, realized_debt_fy
        cur_total = sum(s["value"] for s in sleeves.values())
        if cur_total <= 0 or amount <= 0:
            return 0.0
        amount = min(amount, cur_total)
        for k in sleeves:
            v = sleeves[k]["value"]
            b = sleeves[k]["basis"]
            if v <= 0:
                continue
            portion = amount * (v / cur_total)
            if track_realized:
                gain_frac = max(0.0, (v - b) / v)
                rg = portion * gain_frac
                cat = ASSET_TAX_CATEGORY[k]
                if cat == "equity":
                    realized_equity_fy += rg
                elif cat == "debt":
                    realized_debt_fy += rg
                # FD: no capital gain (interest is taxed separately)
            new_v = v - portion
            sleeves[k]["value"] = new_v
            sleeves[k]["basis"] = b * (new_v / v)
        return amount

    for t in range(1, len(timeline)):
        prev_ts = timeline[t - 1]
        ts = timeline[t]
        prev_total = sum(s["value"] for s in sleeves.values())

        for k in sleeves:
            meta = ASSETS[k]
            v_prev = sleeves[k]["value"]
            if meta.yf_ticker is not None:
                p_prev = price_panel.at[prev_ts, k]
                p_now = price_panel.at[ts, k]
                ret = (p_now / p_prev) - 1.0 if p_prev > 0 else 0.0
            elif k == "FD":
                r_annual = annual_rate_for(fd_rate_series, ts.year)
                ret = (1.0 + r_annual) ** (1.0 / 12.0) - 1.0
            elif k == "DEBT":
                ret = (1.0 + DEBT_ANNUAL_RETURN) ** (1.0 / 12.0) - 1.0
            else:
                ret = 0.0
            new_v = v_prev * (1.0 + ret)
            sleeves[k]["value"] = new_v
            if k == "FD":
                # FD interest is taxable income each year, regardless of withdrawal.
                # Basis tracks value (no capital gain on FD itself).
                fd_interest_fy += max(0.0, new_v - v_prev)
                sleeves[k]["basis"] = new_v
            # Capital sleeves: basis unchanged; the appreciation is unrealized.

        post_return_total = sum(s["value"] for s in sleeves.values())
        port_ret = (post_return_total / prev_total - 1.0) if prev_total > 0 else 0.0
        return_rows.append(port_ret)

        got = _withdraw_proportional(expense, track_realized=True)
        withdrawal_rows.append(got)

        if rebalance == "annual" and ts.month == 12:
            tot = sum(s["value"] for s in sleeves.values())
            if tot > 0:
                targets = {k: tot * w for k, w in alloc.items()}
                for k in sleeves:
                    v = sleeves[k]["value"]
                    b = sleeves[k]["basis"]
                    new_v = targets[k]
                    if v > 0 and new_v < v:
                        # Selling this sleeve down -> realize gain on the sold portion.
                        sold = v - new_v
                        gain_frac = max(0.0, (v - b) / v)
                        rg = sold * gain_frac
                        cat = ASSET_TAX_CATEGORY[k]
                        if cat == "equity":
                            realized_equity_fy += rg
                        elif cat == "debt":
                            realized_debt_fy += rg
                        new_b = b * (new_v / v)
                    elif new_v > v:
                        # Buying this sleeve up -> add to basis at purchase cost.
                        new_b = b + (new_v - v)
                    else:
                        new_b = b
                    sleeves[k]["value"] = new_v
                    sleeves[k]["basis"] = new_b

        tax_this_month = 0.0
        if ts.month == 3:  # Indian FY-end
            rates = tax_rates_on(tax_rates, ts)
            equity_taxable = max(0.0, realized_equity_fy - rates.equity_ltcg_exempt_inr)
            equity_tax = equity_taxable * rates.equity_ltcg_rate
            debt_tax = max(0.0, realized_debt_fy) * rates.debt_ltcg_rate
            fd_tax = max(0.0, fd_interest_fy) * rates.fd_rate
            tax_due = equity_tax + debt_tax + fd_tax
            tax_this_month = _withdraw_proportional(tax_due, track_realized=False)
            total_tax_paid += tax_this_month
            realized_equity_fy = 0.0
            realized_debt_fy = 0.0
            fd_interest_fy = 0.0
        tax_rows.append(tax_this_month)

        per_asset_rows.append({k: s["value"] for k, s in sleeves.items()})
        total = sum(s["value"] for s in sleeves.values())
        total_rows.append(total)

        if total <= 1.0 and depletion_month is None:
            depletion_month = ts
            break

        cpi_annual = annual_rate_for(cpi_series, ts.year)
        monthly_factor = (1.0 + cpi_annual) ** (1.0 / 12.0)
        expense *= monthly_factor
        inflation_factor *= monthly_factor
        inflation_months += 1

    timeline_used = timeline[: len(total_rows)]
    monthly_corpus = pd.Series(total_rows, index=timeline_used, name="corpus")
    per_asset_corpus = pd.DataFrame(per_asset_rows, index=timeline_used).fillna(0.0)
    monthly_withdrawal = pd.Series(withdrawal_rows, index=timeline_used, name="withdrawal")
    monthly_tax = pd.Series(tax_rows, index=timeline_used, name="tax")
    monthly_return = pd.Series(return_rows, index=timeline_used, name="return")

    survived = depletion_month is None
    months_lasted = len(timeline_used) - 1
    peak_corpus = float(monthly_corpus.max())
    end_corpus = float(monthly_corpus.iloc[-1])

    running_peak = monthly_corpus.cummax()
    drawdowns = (monthly_corpus - running_peak) / running_peak.replace(0, np.nan)
    max_drawdown = float(drawdowns.min()) if not drawdowns.isna().all() else 0.0

    returns = monthly_return.iloc[1:]
    if len(returns) > 0:
        compounded = float((1.0 + returns).prod())
        years = len(returns) / 12.0
        if years > 0 and compounded > 0:
            annualized_return = compounded ** (1.0 / years) - 1.0
        else:
            annualized_return = 0.0
        annualized_volatility = float(returns.std() * np.sqrt(12))
        yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
        worst_calendar_year_return = float(yearly.min())
    else:
        annualized_return = 0.0
        annualized_volatility = 0.0
        worst_calendar_year_return = 0.0

    longest = 0
    cur = 0
    for v in drawdowns.fillna(0).values:
        if v < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    if inflation_months > 0:
        years_inf = inflation_months / 12.0
        average_inflation = inflation_factor ** (1.0 / years_inf) - 1.0
    else:
        average_inflation = 0.0

    inflation_index_vals: list[float] = [1.0]
    cum = 1.0
    for ts_i in timeline_used[1:]:
        cpi_i = annual_rate_for(cpi_series, ts_i.year)
        cum *= (1.0 + cpi_i) ** (1.0 / 12.0)
        inflation_index_vals.append(cum)
    inflation_index = pd.Series(inflation_index_vals, index=timeline_used, name="inflation_index")

    nav = (1.0 + returns).cumprod() if len(returns) > 0 else pd.Series([], dtype="float64")
    if len(nav) > 0:
        market_peak = nav.cummax()
        market_dd = (nav - market_peak) / market_peak.replace(0, np.nan)
        market_max_drawdown = float(market_dd.min()) if not market_dd.isna().all() else 0.0
        m_longest = 0
        m_cur = 0
        for v in market_dd.fillna(0).values:
            if v < 0:
                m_cur += 1
                m_longest = max(m_longest, m_cur)
            else:
                m_cur = 0
        longest_market_drawdown_months = int(m_longest)
    else:
        market_max_drawdown = 0.0
        longest_market_drawdown_months = 0

    cashflows = [-corpus]
    last_idx = len(monthly_withdrawal) - 1
    for i in range(1, last_idx):
        cashflows.append(float(monthly_withdrawal.iloc[i]))
    if last_idx >= 1:
        cashflows.append(float(monthly_withdrawal.iloc[last_idx]) + end_corpus)
    effective_annual_return = _money_weighted_irr_monthly(cashflows)

    return SimulationResult(
        monthly_corpus=monthly_corpus,
        per_asset_corpus=per_asset_corpus,
        monthly_withdrawal=monthly_withdrawal,
        monthly_tax=monthly_tax,
        monthly_return=monthly_return,
        depletion_month=depletion_month,
        survived=survived,
        months_lasted=months_lasted,
        peak_corpus=peak_corpus,
        end_corpus=end_corpus,
        max_drawdown=max_drawdown,
        annualized_return=float(annualized_return),
        annualized_volatility=annualized_volatility,
        worst_calendar_year_return=worst_calendar_year_return,
        longest_drawdown_months=int(longest),
        average_inflation=float(average_inflation),
        total_tax_paid=float(total_tax_paid),
        inflation_index=inflation_index,
        effective_annual_return=float(effective_annual_return),
        market_max_drawdown=float(market_max_drawdown),
        longest_market_drawdown_months=int(longest_market_drawdown_months),
    )


def _money_weighted_irr_monthly(cashflows: list[float]) -> float:
    """Bisect for the monthly IRR of a cashflow stream, then annualize.

    Cashflows are in chronological order, one per month, starting at t=0.
    Negative = paid in (initial investment); positive = received."""
    if not cashflows or all(cf == 0 for cf in cashflows):
        return 0.0

    def npv(r: float) -> float:
        return sum(cf / (1.0 + r) ** t for t, cf in enumerate(cashflows))

    lo, hi = -0.5, 1.0
    npv_lo, npv_hi = npv(lo), npv(hi)
    if npv_lo * npv_hi > 0:
        # No sign change in bracket; IRR is undefined or extreme. Fall back to 0.
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6 or (hi - lo) < 1e-10:
            break
        if npv_lo * npv_mid < 0:
            hi, npv_hi = mid, npv_mid
        else:
            lo, npv_lo = mid, npv_mid
    monthly = (lo + hi) / 2.0
    return (1.0 + monthly) ** 12 - 1.0
