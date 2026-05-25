"""Quick non-Streamlit smoke test: exercises the simulator end-to-end."""
from datetime import date

import pandas as pd

from data_loader import get_equity_panel, load_cpi, load_fd_rates, load_tax_rates
from simulator import simulate_swp

print("Loading static data...")
cpi = load_cpi()
fd_rates = load_fd_rates()
tax_rates = load_tax_rates()
print(f"  CPI:      {len(cpi)} years, {cpi.index.min()}–{cpi.index.max()}")
print(f"  FD rates: {len(fd_rates)} years, {fd_rates.index.min()}–{fd_rates.index.max()}")

print("\nFetching Nifty 50 monthly closes (2005-01 -> today)...")
panel = get_equity_panel(["NIFTY50"], date(2005, 1, 1), date.today())
print(f"  Panel shape: {panel.shape}, range: {panel.index.min().date()} -> {panel.index.max().date()}")

print("\nScenario 1 (smoke): ₹1cr, 2005-01-01, 100% Nifty 50, ₹40k/month, no rebalance")
r1 = simulate_swp(
    corpus=1e7,
    start_date=date(2005, 1, 1),
    allocation={"NIFTY50": 1.0},
    monthly_expense=40_000,
    cpi_series=cpi,
    price_panel=panel,
    fd_rate_series=fd_rates,
    tax_rates=tax_rates,
    rebalance="never",
)
print(f"  survived={r1.survived}, months_lasted={r1.months_lasted}, "
      f"peak=₹{r1.peak_corpus/1e7:.2f}cr, end=₹{r1.end_corpus/1e7:.2f}cr, "
      f"CAGR={r1.annualized_return*100:.2f}%, max_dd={r1.max_drawdown*100:.1f}%, "
      f"vol={r1.annualized_volatility*100:.1f}%")

print("\nScenario 2 (failure): ₹50L, 2008-01-01, 100% Nifty 50, ₹1L/month, annual rebalance")
panel2 = get_equity_panel(["NIFTY50"], date(2008, 1, 1), date.today())
r2 = simulate_swp(
    corpus=50e5,
    start_date=date(2008, 1, 1),
    allocation={"NIFTY50": 1.0},
    monthly_expense=1_00_000,
    cpi_series=cpi,
    price_panel=panel2,
    fd_rate_series=fd_rates,
    tax_rates=tax_rates,
    rebalance="annual",
)
print(f"  survived={r2.survived}, months_lasted={r2.months_lasted}, "
      f"depletion={r2.depletion_month.date() if r2.depletion_month else None}, "
      f"max_dd={r2.max_drawdown*100:.1f}%, "
      f"worst_year={r2.worst_calendar_year_return*100:.1f}%")

print("\nScenario 3 (mixed): ₹1cr, 2010-01-01, 50% Nifty/30% FD/20% Debt, ₹60k/month, annual rebalance")
panel3 = get_equity_panel(["NIFTY50"], date(2010, 1, 1), date.today())
r3 = simulate_swp(
    corpus=1e7,
    start_date=date(2010, 1, 1),
    allocation={"NIFTY50": 0.5, "FD": 0.3, "DEBT": 0.2},
    monthly_expense=60_000,
    cpi_series=cpi,
    price_panel=panel3,
    fd_rate_series=fd_rates,
    tax_rates=tax_rates,
    rebalance="annual",
)
print(f"  survived={r3.survived}, months_lasted={r3.months_lasted}, "
      f"peak=₹{r3.peak_corpus/1e7:.2f}cr, end=₹{r3.end_corpus/1e7:.2f}cr, "
      f"CAGR={r3.annualized_return*100:.2f}%, max_dd={r3.max_drawdown*100:.1f}%")

print("\nAll scenarios completed.")
