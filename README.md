# FIRE / SWP Simulator — India

Back-test a Systematic Withdrawal Plan (SWP) against historical Indian market data and find the sweet spot for early retirement (FIRE).

You provide a lump-sum corpus, a start date, an asset allocation, and your monthly expense. The app simulates month-by-month over real historical returns, inflates the expense with India CPI, and reports how long your money would have lasted — plus risk metrics.

## Asset universe (v1)

| Asset | Source |
|---|---|
| Nifty 50 (`^NSEI`) | yfinance |
| Sensex (`^BSESN`) | yfinance |
| Bank Nifty (`^NSEBANK`) | yfinance |
| Equity MF — Nifty 500 proxy (`^CRSLDX`) | yfinance |
| Debt Funds | Flat 7% annual (v1 simplification) |
| Fixed Deposit | Historical SBI 1-yr FD rate (CSV) |

Equity history limits:
- Nifty 50: from 1996
- Sensex: from ~1997 (Yahoo coverage)
- Bank Nifty: from 2000
- Nifty 500: from ~2007

## Quickstart

```powershell
cd E:\finance
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Project layout

```
finance/
├── app.py              # Streamlit UI
├── simulator.py        # Pure SWP simulation
├── data_loader.py      # yfinance + CSV loaders
├── static_data/
│   ├── cpi_india.csv   # Annual India CPI YoY %
│   ├── fd_rates.csv    # Annual SBI 1-yr FD %
│   └── README.md       # Data sourcing notes
├── requirements.txt
└── .gitignore
```

## How the simulation works

At each month-end:
1. Each sleeve grows by its monthly return (equity from yfinance; FD/debt from rates).
2. The monthly expense is withdrawn proportionally across sleeves.
3. The next month's expense is inflated by CPI / 12 (compounded).
4. If rebalance = annual, sleeves are re-split to target weights every December.
5. If the corpus hits zero, the depletion month is recorded.

Reported metrics: months lasted, peak/end corpus, max drawdown, annualized return & volatility, worst calendar-year return, longest drawdown duration.

## Out of scope (v1)

- SIP / accumulation phase
- Real estate, gold, international equity
- Per-scheme MF / debt fund selection (via AMFI NAV API)
- Monte Carlo / rolling-window FIRE success rate
- Tax modeling (LTCG, STCG, FD interest tax)

## Notes

- yfinance is rate-limited; the loader caches results for 24 hours.
- The CPI and FD CSVs ship with reasonable historical approximations. Replace with authoritative figures from World Bank / RBI for production use — see [static_data/README.md](static_data/README.md).
