# 📈 AI Stock Engine — Portfolio Analyzer

A quantitative portfolio analysis platform that combines factor-based stock scoring, statistical portfolio optimization, stability analysis, and a composite rating system into a single interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Factor Scoring** | Multi-factor stock ranking (value, momentum, quality, low-vol) |
| **Portfolio Strategies** | Equal-weight, inverse-volatility, auto-diversified hybrid |
| **Bayesian Shrinkage** | James-Stein shrinkage for expected returns (reduces estimation error) |
| **Robust Covariance** | Ledoit-Wolf / diagonal shrinkage with positive-definiteness guarantee |
| **Backtest Engine** | Walk-forward backtest with transaction costs and vol-targeting |
| **Stability Analyzer** | Rolling composite stability score (0–100) across 5 components |
| **Composite Rating** | 5-component cross-sectional rating (Sharpe 30%, Drawdown 20%, Stability 20%, Regime 15%, Diversification 15%) |
| **Meta-Portfolio** | Optimal blending of user portfolios using mean-variance optimization |
| **Stop-Loss Scoring** | Dynamic risk assessment generating actionable stop thresholds (Hold/Trim/Stop) |
| **Model Governance** | Identifies overfitting (walk-forward gap) and conservative bias via 100-point penalty framework |
| **Error-Coefficient** | OLS regression-based predictive scaling adjusting for local prediction variance |
| **Automatic N-Selection** | Algorithmically determines the optimal number of stocks (N*) based on Sharpe vs diversification benefits |
| **DB Persistence** | Monthly allocations and performance snapshots stored in SQLite/PostgreSQL |
| **Audit Script** | Full system integrity check (31 checks, look-ahead bias detection, stress test) |

---

## 🗂️ Project Structure

```
portfolio_analyzer/
│
├── app/                        # Core application layer
│   ├── config.py               # DATABASE_URL and environment config
│   ├── database.py             # SQLAlchemy engine, session factory
│   ├── models.py               # ORM models (Stock, Price, Score, MonthlyAllocation, PortfolioMetrics, …)
│   └── main.py                 # FastAPI app entry point
│
├── services/                   # Business logic layer
│   ├── data_fetcher.py         # Fetches OHLCV prices from yfinance
│   ├── universe_loader.py      # Loads and filters the stock universe
│   ├── factor_engine.py        # Computes factor signals per stock
│   ├── scoring.py              # Combines factors into composite score
│   ├── recommendation.py       # Stock recommendation engine
│   ├── portfolio.py            # Equal-weight / inverse-vol portfolio builders
│   ├── user_portfolios.py      # User-defined portfolio management
│   ├── backtest.py             # Walk-forward backtest engine
│   ├── return_estimator.py     # Bayesian shrinkage expected returns
│   ├── covariance_estimator.py # Ledoit-Wolf robust covariance matrix
│   ├── auto_diversified_portfolio.py  # Auto-rebalancing hybrid portfolio
│   ├── portfolio_comparison.py # Backtest orchestration + meta-portfolio
│   ├── portfolio_rating.py     # Composite 5-component rating system
│   ├── model_governance.py     # Overfitting & Conservative Bias detection
│   ├── stability_analyzer.py   # Rolling stability score (0–100)
│   ├── allocation_persistence.py # DB write/read for allocations & metrics
│   └── research_validation.py  # Statistical validation utilities
│
├── dashboard/
│   └── run.py                  # Streamlit dashboard (main UI)
│
├── scripts/
│   ├── migrate_add_monthly_tables.py  # DB migration (monthly_allocations, portfolio_metrics)
│   ├── audit_system_integrity.py      # Full system audit + stress test
│   ├── check_db.py                    # Quick DB health check
│   └── save_chat_from_clipboard.py    # Chat history utility
│
├── tests/
│   ├── test_return_estimator.py       # 23 unit tests for Bayesian shrinkage
│   ├── test_portfolio_comparison.py   # Portfolio comparison tests
│   ├── test_research_validation.py    # Validation tests
│   └── run_tests.py                   # Test runner
│
├── data/
│   └── hybrid_rebalance_state.json    # Persisted monthly rebalance state
│
├── logs/
│   └── hybrid_rebalance.log           # Rebalance history log
│
├── reports/
│   └── audit_report_<timestamp>.md    # Generated audit reports
│
├── .env                        # Environment variables (DATABASE_URL, etc.)
├── .env.example                # Template for .env
├── .gitignore
├── requirements.txt
└── README.md                   # This file
```

---

## 🏗️ Architecture Overview

The `portfolio_analyzer` application is composed of several logical layers that interact to fetch data, compute signals, build portfolios, back‑test, and present results via a Streamlit dashboard.

```mermaid
graph TD
  A["Dashboard (run.py)"] --> B["Services Layer"]
  B --> C["Data Fetcher"]
  C --> D["Factor Engine"]
  D --> E["Return Estimator <br/> <i>Shrinkage + Error Coefficient</i>"]
  E --> F["Stop-Loss Scoring Layer"]
  F --> G["Optimizer <br/> <i>Sharpe - RiskPenalty</i>"]
  G --> H["Backtest"]
  H --> I["Stability Analyzer"]
  I --> J["Composite Rating"]
  J --> A
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd portfolio_analyzer
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL:
#   SQLite (default):   DATABASE_URL=sqlite:///./stocks.db
#   PostgreSQL:         DATABASE_URL=postgresql://user:pass@localhost/stockdb
```

### 3. Run database migrations

```bash
# Preview SQL (dry run)
python scripts/migrate_add_monthly_tables.py --dry-run

# Apply migrations
python scripts/migrate_add_monthly_tables.py
```

### 4. Launch the dashboard

```bash
python -m streamlit run dashboard/run.py
```

---

## 🧠 Core Concepts

### Automatic N-Selection Mechanism

The system determines the optimal breadth of the portfolio (N*) dynamically rather than hardcoding a fixed size.
1. Evaluates candidate N sizes (from 5 to 30) using a **Performance-Optimized Selection Loop**.
   - **Optimization**: Performs a single broad fetch and one greedy correlation pass for the maximum N (30), then efficiently slices and re-calculates all candidate metrics in memory.
2. Formulates a **Portfolio Benefit Curve** based on the following weighted scoring:
   - **40% Sharpe (Normalized)**: Risk-adjusted return potential.
   - **20% Diversification**: Measured via 1 - average pairwise correlation.
   - **20% Stability**: Rolling 180-day portfolio curve consistency.
   - **10% RRC (Risk Responsiveness)**: Aggregate stock-level adaptability scores.
   - **10% Governance**: Protection against overfitting (1 - Overfitting Risk).
   - *Residual Penalties (-5%)* applied for extreme concentration (>60% in top 3).
3. **Regime and Stability Awareness:** Automatically adjusts the minimum allowable N during high volatility regimes, and increases N if the mathematical peak exhibits low structural stability.

### Stop-Loss Scoring Engine

The dynamic Stop-Loss capability generates a targeted risk score (0-100) per equity without relying on arbitrary percentages.
1. Computes trailing dynamic ceilings (`high`, `moving average`, `recent volatility`).
2. Triggers progressive alerts (`Hold`, `Trim`, `Stop`) dynamically as the symbol drifts further below its local ceiling.
3. **Adaptive Universe Scaling**: Automatically switches from relative ranking to **Absolute Statistical Benchmarks** when evaluating fewer than 5 stocks (e.g., Stock-Specific mode). This prevents "1-of-1" ranking glitches where a single healthy stock could be flagged as high-risk.
4. Automatically penalizes Meta-Portfolio mean-variance optimization layers.
   - **Mechanism**: The optimizer objective function incorporates an additive penalty to the portfolio's Sharpe ratio based on convex amplification of the weighted average Risk Score. This convex curve geometrically punishes highest-risk clusters and prevents optimizers from hiding small weights in many risky stocks.
     $$Objective = max \left( Sharpe_{raw} - 0.2 \times \sum w_i \times \left(\frac{StopRiskScore_i}{100}\right)^2 \right)$$
5. Employs `error_bias` arrays via the expected-return calculation, widening triggers if a stock has inherently high prediction variance.
   - **Mechanism**: The raw mathematical divergence (`shrinkage_return - sample_return`) is rank-percentiled to a strictly bounded `[0, 100]` score. This is additively integrated into the `StopRiskScore` derivation layer algorithm, with the final composite risk score strictly hard-capped via `.clip(0, 100)` to prevent unbounded overflow penalization.

### Error-Coefficient Framework

Integrated into the Bayesian return estimators and factor pipelines, solving the "overconfident estimation" problem.
1. Employs Time-Series OLS against backward residuals enforcing structural temporal causality (no look-ahead).
2. Generates an `error_coefficient` vector ($\beta$ matrix) mapped globally across the market panel identifying structural sources of variance.
3. **Conceptual Distinction**: The `error_coefficient` (global regression betas derived entirely from market panel factors) is mathematically distinct from the `error_bias` array. The `error_bias` is simply the derived equity-specific scalar prediction constraint ($ShrunkReturn_{i} - CorrectedReturn_{i}$) calculated by feeding localized factor scores into the global coefficient betas.
4. Fully audited by `services/error_model_audit.py` to ensure prediction residuals match realized performance identically.

The system evaluates the structural health of generated models to protect against data mining and overfitting illusions.

1.  **Overfitting Detection**: Measures In-Sample vs. Out-of-Sample Drop-offs (Walk-Forward Gap), Weight-Sensitivity Instability, Excessive Turnover, and Extreme R² variances.
2.  **Conservative Bias Check**: Flags models artificially gaming the Stability Score by minimizing active risk entirely (e.g., tracking Beta < 0.3, Volatility < 8%, with stagnant CAGR).
3.  **Strategy Focus**: Governance checks are exclusively applied to the **Auto Diversified Hybrid** strategy. For benchmark strategies (Equal Weight, Inverse Vol), the console provides guidance to switch to the Hybrid strategy for deep-dive diagnostics.
4.  **Governance Health Score**: Evaluated objectively out of 100.
    - 🟢 **Balanced / Slightly Conservative**: Models with 1-2 minor conservative traits.
    - 🔴 **Overly Conservative**: Models failing 3+ major checks.
    Scores and classifications are visually rendered dynamically inside the "Research Mode" split dashboard layout.

### Composite Portfolio Rating (0–100)

The rating system evaluates portfolios across 5 cross-sectionally normalised dimensions:

| Component | Weight | Metric |
|---|---|---|
| Risk-adjusted return | **30%** | Normalised Sharpe ratio |
| Drawdown control | **20%** | Normalised `1 − |Max Drawdown|` |
| Stability Score | **20%** | Rolling composite stability (0–100) |
| Diversification quality | **15%** | `1 − avg_pairwise_correlation` |
| Regime consistency | **15%** | `1 − regime_sharpe_gap` |

**Grade thresholds:** 90–100 → A+ · 80–89 → A · 70–79 → B · 60–69 → C · <60 → D

Risk Responsiveness ensures portfolios are adaptive, not just statically low-volatility. RRC is a 0-100 score injected into multiple layers:
1. **Stock RRC**: Calculates Drawdown Recovery, Volatility Spike Reaction, Beta Compression, and Correlation Drift relative to the market. 
   - **Small-Universe Stability**: Uses absolute defensive benchmarks (e.g., 30-day recovery target) when viewing single stocks to avoid ranking glitches.
2. **Portfolio RRC**: Measures drawdown containment speed, volatility adaptation, and Sharpe stability across regimes. Penalizes non-adaptive meta-portfolios.

### Stability Score (0–100)

Five rolling components computed over a 125-day window:

| Component | Weight |
|---|---|
| Sharpe Consistency | 25% |
| Drawdown Stability | 25% |
| Turnover Penalty | 20% |
| Correlation Stability | 20% |
| Regime Robustness | 10% |

### Monthly Rebalance Discipline

The hybrid auto-portfolio enforces one rebalance per calendar month. The guard:
1. Reads `data/hybrid_rebalance_state.json`
2. Checks if `last_rebalance_date` falls in the current month
3. Skips if already rebalanced; `force=True` overrides

### ⚡ Performance & Scalability

Large-scale backtests (e.g., 10+ years of monthly rebalancing) have been optimized for high-speed execution:
1.  **Fast Candidate Evaluation**: Auto N-selection uses a vectorized, single-pass simulation to evaluate breadth, reducing Search-Space overhead by **90%**.
2.  **Superset Caching**: The internal return loader is superset-aware; it slices already-cached data for smaller requests (fewer stocks or shorter windows) instead of refetching from the DB.
3.  **Vectorized Rebalance Discovery**: Rebalance candidate dates are identified via vectorized binary searches against price/score date panels.
4.  **Database Indexing**: Optimized `ix_scores_date` and `ix_factors_date` indexes ensure O(1) temporal lookups during join operations.
    - **Impact**: Full Hybrid rebalance runtime reduced from **~60 minutes** to **<10 minutes**.

### No Look-Ahead Bias

All backtest functions enforce strict temporal causality:
- Scores: `score_date <= rebalance_date`
- Prices: `Price.date <= as_of_date`
- Vol-targeting uses lagged (t−1) rolling volatility

---

## 📊 Dashboard Sections

| Section | Content |
|---|---|
| **Section 1** | Composite Rating Table — ranked by score, trophy banner for #1 |
| **Section 2** | Equity Curve Overlay — top portfolio highlighted with bold line |
| **Section 3** | Correlation Heatmap — pairwise portfolio return correlations |
| **Section 4** | Rating Summary — compact rank/grade/stability view |
| **Hybrid Detail** | Stability breakdown, regime comparison, rolling Sharpe chart |
| **Meta Portfolio** | Blended weights, metrics, equity curve vs. best user portfolio |

---

## 🔧 Running the Audit

The audit script performs **31 checks** across static analysis, unit tests, and a stress test:

```bash
python scripts/audit_system_integrity.py

# With memory profiling:
pip install psutil
python scripts/audit_system_integrity.py --output-dir reports
```

**Checks include:**
- ✅ No look-ahead bias in backtest paths
- ✅ Bayesian shrinkage applied in all optimisers
- ✅ Robust covariance with PD guarantee
- ✅ Monthly discipline enforced
- ✅ Stability computed before rating
- ✅ Composite weight sum = 1.000
- ✅ Stress test: 100 stocks × 5 years × 7 portfolios in < 2s

---

## 🗄️ Database Schema

### `monthly_allocations`
Stores stock weights per rebalance — append-only, never overwrites history.

| Column | Type | Notes |
|---|---|---|
| `portfolio_type` | String | e.g. `auto_diversified_hybrid` |
| `portfolio_name` | String? | User label (null for system portfolios) |
| `rebalance_date` | Date | Month-end trading date |
| `stock_id` | FK → stocks | `ON DELETE SET NULL` |
| `weight` | Float | ∈ [0,1] |
| `forced` | Boolean | Monthly guard bypassed? |

**Unique constraint:** `(portfolio_type, portfolio_name, rebalance_date, stock_id)`

### `portfolio_metrics`
Performance + stability snapshot per rebalance — append-only.

Columns: `cagr`, `volatility`, `sharpe`, `max_drawdown`, `calmar`, `sortino`, `total_return`,
`stability_score`, `stability_grade`, `composite_rating`, `composite_grade`,
`n_stocks`, `expected_volatility`, `expected_sharpe`, `avg_pairwise_corr`

**Unique constraint:** `(portfolio_type, portfolio_name, rebalance_date)`

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
# or
python tests/run_tests.py
```

Current coverage: Over **150 passing tests**, featuring full suites for return estimators, auto-diversified portfolios, covariance estimations, error-coefficient models, stop-loss engines, Risk Responsiveness metrics, model governance integration, and automatic N-selection calibration.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive dashboard UI |
| `sqlalchemy` | ORM and database abstraction |
| `pandas` / `numpy` | Data manipulation |
| `scipy` | Portfolio optimisation (SLSQP) |
| `scikit-learn` | Ledoit-Wolf covariance shrinkage |
| `plotly` | Interactive charts |
| `yfinance` | Market data fetching |
| `python-dotenv` | Environment variable loading |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./stocks.db` | SQLAlchemy database URL |

---

## 📝 License

Internal research project. Not financial advice.
