"""Catalog of assistant prompt suggestions, runnable queries, and roadmap queries."""

ASSISTANT_PROMPT_SUGGESTIONS = [
    "Should I buy or hold TCS?",
    "Analyze INFY and explain its factor strengths.",
    "Show the top 10 stocks right now.",
    "Compare my portfolios and tell me which is strongest.",
    "Show the top 5 sectors and their best 3 stocks.",
    "Explain the hybrid portfolio and how it is built.",
]


ASSISTANT_RUNNABLE_QUERY_LIBRARY = {
    "Stock Decisions": [
        "Should I buy or hold TCS?",
        "Should I reduce Reliance?",
        "Suggest a stop-loss for ICICIBANK.",
        "Analyze INFY and explain its factor strengths.",
        "Analyze HDFCBANK fundamentals and factors.",
    ],
    "Market & Sectors": [
        "Show the top 10 stocks right now.",
        "Recommend 10 stocks to buy.",
        "Show the top 5 sectors and their best 3 stocks.",
        "List the best 3 sectors.",
        "Show the top 10 sectors.",
    ],
    "Portfolio & Strategy": [
        "Compare my portfolios and tell me which is strongest.",
        "Explain the hybrid portfolio and how it is built.",
        "How many stocks should the hybrid portfolio hold?",
        "Check for overfitting or bias in the model.",
        "Explain the latest calibration drift.",
        "Show the stability analysis.",
    ],
}


ASSISTANT_ROADMAP_QUERY_LIBRARY = {
    "Portfolio Health": [
        "What is my portfolio's total current value and gain/loss?",
        "Which holdings are contributing the most to my returns?",
        "Am I overexposed to any stock or sector?",
        "How diversified is my portfolio right now?",
        "How does my portfolio compare with Nifty 50?",
    ],
    "Rebalancing": [
        "Does my portfolio need rebalancing?",
        "What would be the ideal target allocation today?",
        "How far has my portfolio drifted from its target weights?",
        "How would equal-weight and inverse-volatility allocations differ?",
        "Which positions should I trim or add to rebalance efficiently?",
    ],
    "Backtesting": [
        "How would my portfolio have performed from 2024-07-01 to today?",
        "Compare Strategy A and Strategy B on CAGR, Sharpe, and drawdown.",
        "How would a custom portfolio of TCS, INFY, and HDFCBANK have performed?",
        "What if I had excluded my worst-performing stock?",
        "How sensitive are returns to changing Top N stocks?",
    ],
    "Tracker Queries": [
        "What is my invested amount vs current amount by stock?",
        "Which tracked position has the highest unrealized profit?",
        "Which tracked position has the highest unrealized loss?",
        "What is my average buy price for each holding?",
        "How concentrated is my tracker in the top 3 holdings?",
    ],
}


ASSISTANT_SUPPORTED_INTENTS = [
    {"Intent": "Stock Recommendation", "Status": "Supported now", "Functions": "get_stock_summary, get_stop_loss", "Use Cases": "Buy/hold/sell, stock summary, stop-loss", "Example": "Should I buy TCS?"},
    {"Intent": "Stock Research", "Status": "Supported now", "Functions": "get_stock_research", "Use Cases": "Factor analysis, deeper company review", "Example": "Analyze INFY fundamentals and factors."},
    {"Intent": "Market Overview", "Status": "Supported now", "Functions": "list_stocks", "Use Cases": "Top-ranked stocks, market opportunities", "Example": "Show the top 10 stocks."},
    {"Intent": "Sector Ranking", "Status": "Supported now", "Functions": "get_sector_rankings", "Use Cases": "Best sectors and leading stocks", "Example": "Show the top 5 sectors and their best 3 stocks."},
    {"Intent": "Portfolio Comparison", "Status": "Supported now", "Functions": "compare_portfolios", "Use Cases": "Compare saved portfolios or strategies", "Example": "Compare my portfolios and tell me which is strongest."},
    {"Intent": "Hybrid Portfolio", "Status": "Supported now", "Functions": "get_hybrid_portfolio", "Use Cases": "Hybrid construction and explanation", "Example": "Explain the hybrid portfolio."},
    {"Intent": "Automatic N", "Status": "Supported now", "Functions": "get_auto_n_analysis, get_hybrid_portfolio", "Use Cases": "Optimal stock count explanation", "Example": "How many stocks should the hybrid portfolio hold?"},
    {"Intent": "Governance Check", "Status": "Supported now", "Functions": "get_governance", "Use Cases": "Overfitting and model quality checks", "Example": "Check for overfitting or bias."},
    {"Intent": "Calibration Explanation", "Status": "Supported now", "Functions": "get_latest_calibration", "Use Cases": "Drift and coefficient review", "Example": "Explain the latest calibration drift."},
    {"Intent": "Stability Diagnostic", "Status": "Supported now", "Functions": "get_stability_analysis", "Use Cases": "Rolling stability and turnover", "Example": "Show the stability analysis."},
]


ASSISTANT_NEXT_INTENTS = [
    {"Intent": "Portfolio Health Summary", "Status": "Recommended next", "Functions": "get_portfolio_health, get_portfolio_exposure", "Use Cases": "Value, allocation, concentration, sector mix", "Example": "What is my portfolio's total value and sector exposure?"},
    {"Intent": "Portfolio Rebalancing", "Status": "Recommended next", "Functions": "get_rebalance_plan", "Use Cases": "Target weights, drift, trim/add suggestions", "Example": "Does my portfolio need rebalancing?"},
    {"Intent": "Portfolio Tracker Review", "Status": "Recommended next", "Functions": "get_tracker_summary", "Use Cases": "Invested vs current, unrealized P&L, average cost", "Example": "Which tracked position has the highest unrealized loss?"},
    {"Intent": "Custom Backtest Request", "Status": "Recommended next", "Functions": "run_custom_backtest", "Use Cases": "Date-range performance, what-if analysis", "Example": "How would my portfolio have performed since July 2024?"},
    {"Intent": "Portfolio Risk Review", "Status": "Recommended next", "Functions": "get_portfolio_risk_review", "Use Cases": "Volatility, beta, drawdown, correlation, overlap", "Example": "What is the biggest risk in my portfolio right now?"},
]
