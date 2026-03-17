---
name: Portfolio Analyzer Engine Optimization & Maintenance
description: Design patterns for managing heavy quantitative data, backtesting, and performance optimization in Streamlit/Cloud Run applications.
---

# Portfolio Analyzer Engine Optimization

This skill encapsulates the optimization patterns and maintenance workflows developed for the AI Stock Engine. It focuses on ensuring a responsive user experience despite heavy background computations (matrix ops, OLS regression, stop-loss analysis).

## Core Optimization Patterns

### 1. The "Immediate vs. Optional" UI Pattern
When a page section contains both "fast" data (e.g., simple ranking) and "slow" data (e.g., complex risk profiles):
- **Immediate**: Always render the fast data on page load.
- **Optional**: Gate the slow data behind a `st.button` or an expander.
- **Persistence**: Use `st.session_state` to keep the slow data visible once the user has opted to run it, avoiding redundant re-runs if they toggle mode or refresh the page subset.

### 2. Startup Guardrails
Heavy background maintenance tasks (like OLS Calibration) should NEVER block the application's root `main()` function or `st.set_page_config()`.
- **User-Triggered**: Move auto-running startup tasks to explicit UI controls (e.g., a "Run Maintenance" button).
- **TTL Caching**: Use `@st.cache_resource` with an intentional No-Op function if you want to explicitly disable a previous startup script without deleting the logic.

### 3. Database Performance for Quantitative Ranks
- **Composite Indexing**: For tables like `scores`, always index `(date, rank)`. Most quantitative queries filter by date and sort/slice by rank.
- **Bulk Loading (Anti-N+1)**: When loading structured objects (like User Portfolios with their related symbols), use SQLAlchemy `.options(joinedload(...))` to fetch the entire tree in one roundtrip.

### 4. Streamlit Performance
- **Heavy Backtests**: Ensure backtests use `@st.cache_data(ttl=...)` and respect the `st.session_state.use_multiprocessing` flag.
- **Rerun Management**: Use a custom `_safe_rerun()` helper that supports newer Streamlit versions while falling back gracefully.

## Deployment Checklist
1. **Memory Bounds**: Cloud Run requires at least 8GiB for high-depth backtests.
2. **PostgreSQL Sockets**: Child processes must call `engine.dispose(close=False)` to avoid inheriting and corrupting the primary connection pool.
3. **Environment Isolation**:
    - Local: Hardcode `user_id = 1` if auth is disabled.
    - Web: Use `require_login()` and map `st.session_state["user_id"]` from the OAuth object.
