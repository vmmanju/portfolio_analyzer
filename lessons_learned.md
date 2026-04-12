# Portfolio Analyzer: Lessons Learned

## 1. Architectural Lessons

### The "Silent Killer": Startup Auto-Operations
*   **The Problem**: A `Dockerfile` or `main.py` that runs "invisible" maintenance tasks (like `update_error_model_if_due()`) can triple the "Cold Start" time on Cloud Run.
*   **The Lesson**: Maintenance logic should be periodic (CRON) or user-triggered via a dedicated admin expander, never part of the critical rendering path.

### Environment-Specific Identity
*   **The Problem**: Portfolios were "missing" locally because the app used `user_id=1` while the database had entries for `m.manjunatha@gmail.com`.
*   **The Lesson**: Always expose the current `user_id` and `user_email` in a "Debug" section or Sidebar caption during development to make identity-mapping explicit.

## 2. Performance & Database Optimization

### Indices: Single vs. Composite
*   **The Problem**: Querying `scores` by `date` was fast, but sorting/slicing the top 10 stocks for that date was still slow under load.
*   **The Lesson**: A single index on `date` is insufficient for "Top N" queries. A composite index on `(date, rank)` allows the database to perform index-only scans for the most frequent quantitative operations.

### The N+1 Query Trap in ORMs
*   **The Problem**: Loading portfolios with their symbols took `1 + N` queries (1 for portfolios, N for each portfolio's stocks).
*   **The Lesson**: Even in simple scripts, the performance difference between 1 bulk query with `joinedload` and N small queries is the difference between an "instant" load and a 1-2 second lag.

## 3. UI/UX Design for Data Science

### "Perceived Speed" vs. "Real Speed"
*   **The Problem**: Users felt the site was "slow" because the whole page stayed blank while the heavy Stop-Loss analysis ran.
*   **The Lesson**: Decompose components. Rendering the Top 10 Symbols immediately gives the user something to read while they decide if they want to wait for the deeper "Risk Analysis."

### Session State as a "Short-Term Cache"
*   **The Problem**: Navigating back and forth re-triggered expensive backtests even when parameters hadn't changed.
*   **The Lesson**: Supplement `@st.cache_data` with manual `st.session_state` persistence for UI-heavy results. This avoids the overhead of hashing large function arguments on every script rerun.

### Stale Frontends and Cache Invalidation
*   **The Problem**: Users saw "old dates" (e.g. March 19) despite the database being updated to March 24. This was due to `st.cache_data` having a 1-hour TTL and `st.session_state` persisting the old date preference across page refreshes in the same session.
*   **The Lesson**: 
    1.  Use short TTLs (< 5 mins) for "metadata" functions that return the available date range.
    2.  Proactively check `st.session_state.date < db_latest_date` on every run and force an update to the latest date.
    3.  Trigger re-calibration of heavy stats (like Auto-N) whenever a newer `as_of_date` is selected.

## 4. Deployment & Infrastructure (GCP Cloud Run)

### Memory Quotas in Rollouts
*   **The Problem**: Deployments fail with `Quota violated` even if the project has enough memory.
*   **The Lesson**: Cloud Run rollouts double the usage (Old + New versions run at once). Ensure the region project quota permits sufficient headroom (e.g., peak use = 2x instance RAM).

### Multi-processing Disconnections
*   **The Problem**: Enabling Multiprocessing caused Neon PostgreSQL to close connections unexpectedly.
*   **The Lesson**: Child processes "inherit" DB socket handles but can't share them. `engine.dispose(close=False)` in the worker's first line is mandatory.

### Neon Database Size Limits
*   **The Problem**: Neon free-tier has a 500MB limit. 
*   **The Lesson**: Maintain a rolling window for historical data (e.g. 2 years) to keep the `prices` and `scores` tables within technical bounds.
