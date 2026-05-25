# Static data

Two annual reference series the simulator can't get from yfinance.

| File | Series | Units | Authoritative source |
|---|---|---|---|
| `cpi_india.csv` | India CPI year-over-year change | percent | [World Bank — FP.CPI.TOTL.ZG (India)](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=IN) or RBI DBIE handbook |
| `fd_rates.csv` | SBI 1-year domestic FD rate | percent | [RBI DBIE — Interest Rates on Deposits with Scheduled Commercial Banks](https://dbie.rbi.org.in/) |

**Values currently in these files are reasonable historical approximations** sourced from publicly available references to make the app runnable out of the box. Replace them with authoritative figures for production use.

Format rules:
- One row per calendar year, sorted ascending.
- Header row required (`year,cpi_yoy_pct` and `year,sbi_1yr_fd_pct`).
- For years missing from the CSV, the simulator forward-fills the most recent known value (with a warning logged in the app).

When adding a new year, append a single row — no other changes needed.
