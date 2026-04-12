# Portfolio Analyzer — Lessons Learned Knowledge Base
# Format: each chunk is separated by --- and contains YAML-like frontmatter + lesson text.

---
id: ARCH-001
category: Architecture
title: Startup Auto-Operations Are a Cold Start Silent Killer
tags: [dockerfile, cold-start, cloud-run, performance, startup, maintenance]
---
**Problem**: A `Dockerfile` or `main.py` that runs "invisible" maintenance tasks (like `update_error_model_if_due()`) at startup can triple the Cold Start time on Cloud Run.
**Lesson**: Maintenance logic should be periodic (CRON) or user-triggered via a dedicated admin expander, never part of the critical rendering path.
**Fix Pattern**: Move auto-running startup tasks to explicit UI controls (e.g., a "Run Maintenance" button inside a Streamlit expander). Use `@st.cache_resource` with an intentional No-Op function if you want to explicitly disable a previous startup script without deleting the logic.

---
id: ARCH-002
category: Architecture
title: Environment-Specific Identity (Local vs. Web user_id Mismatch)
tags: [authentication, user_id, debug, local, environment, identity]
---
**Problem**: Portfolios were "missing" locally because the app used `user_id=1` while the production database had entries for `m.manjunatha@gmail.com`.
**Lesson**: Always expose the current `user_id` and `user_email` in a "Debug" section or Sidebar caption during development to make identity-mapping explicit.
**Fix Pattern**:
- Local: Hardcode `user_id = 1` if auth is disabled.
- Web: Use `require_login()` and map `st.session_state["user_id"]` from the OAuth object.

---
id: ARCH-003
category: Architecture
title: Two-Layer Intent Routing (LLM + Deterministic Override)
tags: [intent-router, llm, routing, assistant, fallback, regex]
---
**Problem**: Relying solely on an LLM for intent classification leads to misclassification under ambiguous phrasing. Relying solely on regex misses semantic nuance.
**Lesson**: Use a **two-layer router**: try LLM routing first (`llm_route_query()`), then apply deterministic regex overrides to fix known failure modes (e.g., LLM classifying "sectors" queries as "Market Overview").
**Fix Pattern**:
```python
if intent == "Market Overview" and re.search(r'\bsectors?\b', query, re.IGNORECASE):
    intent = "Sector Ranking"
```
This hybrid approach balances semantic understanding with reliability.

---
id: ARCH-004
category: Architecture
title: Bypassing LLM for Structured Formatting Intents
tags: [intent, assistant, response-generator, formatting, sector-ranking, reliability]
---
**Problem**: Having the LLM generate a markdown table for structured data (e.g., sector rankings) leads to inconsistent formatting and hallucinated values.
**Lesson**: For intents where the output is a precise table or structured report, **bypass the LLM entirely** in `assistant.py` and call a Python formatter (`response_generator.py`) directly. The LLM is only used for free-text explanation phases.
**Fix Pattern**: In `assistant.py`, check if `intent == "Sector Ranking"` and call `format_sector_ranking_response(data)` directly, skipping `llm_generate_explanation()`.

---
id: ARCH-005
category: Architecture
title: Adding a New Intent — Full Wiring Checklist
tags: [intent, new-feature, checklist, assistant, llm-engine, intent-router, function-registry, response-generator]
---
**Problem**: Adding a new intent requires changes across multiple files; missing one silently breaks the feature.
**Lesson**: Adding a new intent requires changes in exactly these 5 places:
1. `llm_engine.py` — add to `INTENTS_LIST` and to the system prompt guidelines
2. `intent_router.py` — add regex pattern to `INTENT_MAPPINGS` and a routing branch in `route_query()`
3. `function_registry.py` — implement the function that fetches data
4. `response_generator.py` — implement the markdown formatter
5. `assistant.py` — wire the intent to the function and formatter

---
id: PERF-001
category: Performance & Database
title: Composite Index for Top-N Quantitative Queries
tags: [database, index, composite-index, query, performance, scores, rank]
---
**Problem**: Querying `scores` by `date` was fast, but sorting/slicing the top 10 stocks for that date was still slow under load.
**Lesson**: A single index on `date` is insufficient for "Top N" queries. A composite index on `(date, rank)` allows the database to perform index-only scans for the most frequent quantitative operations.
**Fix Pattern**: `CREATE INDEX idx_scores_date_rank ON scores (date, rank);`

---
id: PERF-002
category: Performance & Database
title: The N+1 Query Trap in ORMs (joinedload)
tags: [sqlalchemy, orm, n+1, joinedload, portfolio, database, performance]
---
**Problem**: Loading portfolios with their symbols took `1 + N` queries (1 for portfolios, N for each portfolio's stocks).
**Lesson**: Even in simple scripts, the performance difference between 1 bulk query with `joinedload` and N small queries is the difference between an "instant" load and a 1-2 second lag.
**Fix Pattern**: Use `session.query(Portfolio).options(joinedload(Portfolio.symbols)).all()` to fetch the entire object tree in one roundtrip.

---
id: PERF-003
category: Performance & Database
title: Neon Free-Tier 500MB Storage Limit — Rolling Window
tags: [neon, database, storage, limit, rolling-window, prices, scores]
---
**Problem**: Neon free-tier has a 500MB limit. Unbounded historical data accumulation exceeds this.
**Lesson**: Maintain a rolling window for historical data (e.g., 2 years) to keep the `prices` and `scores` tables within technical bounds on free-tier databases.
**Fix Pattern**: All data fetch scripts default to `lookback_years=2`. When updating, first delete records older than the cutoff before inserting new ones.

---
id: PERF-004
category: Performance & Database
title: Sector Filtering via Case-Insensitive Partial Match
tags: [sector, filtering, database, sql, case-insensitive, ilike, partial-match]
---
**Problem**: Queries like "Technology" failed to match "Information Technology" in the database, returning no results.
**Lesson**: Use case-insensitive partial matching (SQL `ILIKE '%sector%'` or Pandas `str.contains(..., case=False)`) when filtering by sector name extracted from natural language.
**Fix Pattern in pandas**:
```python
mask = df["sector"].str.contains(sector_name, case=False, na=False)
```

---
id: UX-001
category: UI/UX
title: "Immediate vs. Optional" Pattern — Perceived vs. Real Speed
tags: [ui, ux, streamlit, session-state, button, performance, stop-loss, perceived-speed]
---
**Problem**: Users felt the site was "slow" because the whole page stayed blank while the heavy Stop-Loss analysis ran.
**Lesson**: Decompose page sections. Render "fast" data (e.g., Top 10 Symbols) immediately on load. Gate "slow" data (e.g., Risk Analysis) behind a `st.button`. Use `st.session_state` to persist the slow data so re-navigating doesn't re-trigger it.
**Fix Pattern**:
```python
st.dataframe(fast_data)  # Always shown immediately
if st.button("Run Risk Analysis") or st.session_state.get("risk_ran"):
    st.session_state["risk_ran"] = True
    st.dataframe(slow_data)
```

---
id: UX-002
category: UI/UX
title: Session State as a Short-Term Cache for Expensive Backtests
tags: [session-state, cache, backtest, streamlit, performance, rerun]
---
**Problem**: Navigating back and forth re-triggered expensive backtests even when parameters hadn't changed.
**Lesson**: Supplement `@st.cache_data` with manual `st.session_state` persistence for UI-heavy results. This avoids the overhead of hashing large function arguments on every script rerun.
**Fix Pattern**: Store results in `st.session_state["backtest_result"]` after the first run. Check for its presence before re-running. Clear it explicitly only when the user changes inputs.

---
id: UX-003
category: UI/UX
title: Stale Frontend Date — Cache TTL and Session State Forcing
tags: [cache, ttl, session-state, date, stale, streamlit, auto-n, cache-invalidation]
---
**Problem**: Users saw "old dates" (e.g., March 19) despite the database being updated to March 24. This was due to `st.cache_data` having a 1-hour TTL and `st.session_state` persisting the old date preference.
**Lesson**:
1. Use short TTLs (< 5 mins / 300s) for "metadata" functions that return the available date range.
2. Proactively check `st.session_state.date < db_latest_date` on every run and force-update to the latest date.
3. Re-trigger heavy calibrations (like Auto-N) whenever a newer `as_of_date` is selected.
**Fix Pattern**:
```python
db_latest = get_latest_date()  # TTL=60s
if st.session_state.get("end_date", date.min) < db_latest:
    st.session_state["end_date"] = db_latest
```

---
id: DEPLOY-001
category: Deployment & GCP
title: Memory Quota in Cloud Run Rollouts (Peak = 2x Instance RAM)
tags: [cloud-run, memory, quota, deployment, rollout, gcp]
---
**Problem**: Deployments fail with `Quota violated` even if the project has enough memory.
**Lesson**: Cloud Run rollouts spin up the new revision simultaneously with the old one. Peak usage during rollout = 2x instance RAM × max-instances. Set `--memory 8Gi --max-instances 1` so the rollout safely peaks at 16GiB, well within the `us-central1` 40GiB quota.
**Fix Pattern**: `gcloud run deploy ... --memory 8Gi --max-instances 1`

---
id: DEPLOY-002
category: Deployment & GCP
title: Multiprocessing + PostgreSQL Disconnections (engine.dispose)
tags: [multiprocessing, postgresql, neon, connection-pool, engine, dispose, cloud-run, fork]
---
**Problem**: Enabling Python `ProcessPoolExecutor` caused Neon PostgreSQL to close connections unexpectedly.
**Lesson**: Linux uses `fork` to spawn child processes. Inherited SQLAlchemy `engine` connection pools map to physical network sockets. Multiple children writing through the same socket overwhelms the DB router. Call `engine.dispose(close=False)` as the first line in every worker function.
**Fix Pattern**:
```python
def parallel_worker_task(args):
    from app.database import engine
    engine.dispose(close=False)  # Drop inherited socket, create fresh pool
    # ... DB queries ...
```

---
id: DEPLOY-003
category: Deployment & GCP
title: gcloud Source Upload UnicodeDecodeError on Windows
tags: [gcloud, windows, unicode, upload, dockerignore, gcloudignore, deploy]
---
**Problem**: The `gcloud` CLI on Windows crashes with `UnicodeDecodeError: invalid start byte 0xff` when it tries to parse binary files (`.docx`, `.xlsx`, `.pyc`) during source upload.
**Lesson**: Disable the gcloudignore feature entirely and rely on `.dockerignore` to exclude unnecessary files.
**Fix Pattern**:
```bash
gcloud config set gcloudignore/enabled false
```
Ensure `.dockerignore` excludes `__pycache__/`, `.agents/`, `.pytest_cache/`, `.git/`, `*.docx`, `*.xlsx`.

---
id: DEPLOY-004
category: Deployment & GCP
title: Custom Domain Mapping Only Works in us-central1
tags: [custom-domain, cloud-run, region, gcp, dns, ssl]
---
**Problem**: Custom domain mapping is not natively available in `asia-south1`; it requires a load balancer (extra cost and complexity).
**Lesson**: Deploy to `us-central1` for native Cloud Run custom domain mapping. Add A/AAAA records for the root and `CNAME ghs.googlehosted.com.` for `www`. SSL provisioning takes 30–60 minutes.

---
id: DEPLOY-005
category: Deployment & GCP
title: Secrets Management — Mount secrets.toml via GCP Secret Manager
tags: [secrets, gcp, secret-manager, streamlit, secrets-toml, docker, deploy]
---
**Problem**: Baking `secrets.toml` into the Docker image exposes credentials.
**Lesson**: Use GCP Secret Manager and mount the secret as a volume at the path Streamlit expects: `/app/.streamlit/secrets.toml`.
**Fix Pattern**:
```bash
gcloud secrets create streamlit-secrets --data-file=.streamlit/secrets.toml
gcloud run deploy ... --set-secrets="/app/.streamlit/secrets.toml=streamlit-secrets:latest"
```

---
id: DEPLOY-006
category: Deployment & GCP
title: OAuth Redirect URIs Must Match Exactly
tags: [oauth, google-login, oidc, redirect-uri, streamlit, secrets]
---
**Problem**: Google OIDC login fails after switching to a custom domain because the `redirect_uri` in `secrets.toml` still pointed to the old Cloud Run URL.
**Lesson**: When moving to a custom domain, update THREE places simultaneously:
1. `secrets.toml` → `redirect_uri = "https://www.jooju.in/oauth2callback"`
2. Google Cloud Console → Authorized Redirect URIs (exact match)
3. Google Cloud Console → Authorized JavaScript Origins (`https://jooju.in`, `https://www.jooju.in`)

---
id: ASSIST-001
category: AI Assistant
title: LLM Intent Routing Prompt — Ordering and Override Rules
tags: [llm, intent, routing, prompt, sector, market-overview, guideline]
---
**Problem**: The LLM misclassified "top sectors and stocks" as "Market Overview" because "Market Overview" was listed first and matched the keyword "stocks".
**Lesson**: In the LLM routing system prompt, **order matters**. Put more specific intents (like "Sector Ranking") after generic ones but include an explicit override rule:
> "This intent ALWAYS overrides 'Market Overview' if the word 'sector' is the primary grouping or focus."
The deterministic override in `intent_router.py` also serves as a safety net.

---
id: ASSIST-002
category: AI Assistant
title: Extracting Numeric Parameters from Queries via Regex
tags: [regex, intent-router, top-n, stocks-per-sector, parameter-extraction, sector-ranking]
---
**Problem**: The intent router detected "Sector Ranking" but didn't extract how many sectors or how many stocks per sector the user wanted.
**Lesson**: Extract numeric parameters from the raw query string using ordered regex patterns. Try the more specific pattern first (e.g., "top N sectors") before falling back to a generic number match.
**Fix Pattern**:
```python
n_match = re.search(r'(?:top|best)\s+(\d+)\s+sectors?', query.lower())
if not n_match:
    n_match = re.search(r'(?:top|best)\s+(\d+)', query.lower())
top_n = int(n_match.group(1)) if n_match else 10

s_match = re.search(r'(\d+)\s+stocks?', query.lower())
stocks_per_sector = int(s_match.group(1)) if s_match else 0
```

---
id: ASSIST-003
category: AI Assistant
title: Anti-Hallucination — Ground LLM with Structured JSON Only
tags: [llm, hallucination, json, prompt, explanation, system-prompt, anti-hallucination]
---
**Problem**: The LLM generated US stock symbols when asked about Indian sector stocks because it had no grounding data and invented results.
**Lesson**: The `llm_generate_explanation()` function must always pass structured JSON from the backend as its sole data source. The system prompt must explicitly forbid inventing metrics:
> "DO NOT invent metrics. Use ONLY the data provided in the JSON."
The RAG retrieval (this system) is a complement — it provides *procedural* context, not data. Data always comes from the engine.

---
id: ASSIST-004
category: AI Assistant
title: Sector Ranking — Enriching Response with Top Stocks per Sector
tags: [sector-ranking, function-registry, stocks-per-sector, response, enrichment]
---
**Problem**: `get_sector_rankings` returned aggregated sector metrics but no stock-level data, so the LLM hallucinated stock names when the user asked "top sectors and 2 stocks from each."
**Lesson**: When `stocks_per_sector > 0`, the function registry must cross-reference the `ranked_stocks` DataFrame and attach matching stocks to each sector's data object.
**Fix Pattern**:
```python
if stocks_per_sector > 0 and not ranked_df.empty:
    s_df = ranked_df[ranked_df["sector"] == sector_name].head(stocks_per_sector)
    sector_info["top_stocks"] = [{"symbol": r["symbol"], "rank": r["rank"]} for _, r in s_df.iterrows()]
```

---
id: ASSIST-005
category: AI Assistant
title: LLM Response Validation — Fallback on Bad JSON
tags: [llm, json, validation, fallback, error-handling, intent-router]
---
**Problem**: The Ollama LLM occasionally returns malformed JSON, causing an unhandled exception that crashes the assistant.
**Lesson**: Always wrap LLM JSON parsing in a try/except and return safe defaults. Validate that the returned `intent` value is in the known `INTENTS_LIST`; if not, force `intent = ""` to trigger the regex fallback.
**Fix Pattern**:
```python
try:
    data = json.loads(response)
    intent = data.get("intent", "")
    if intent not in INTENTS_LIST:
        intent = ""
except Exception:
    return "", "", "", [], ""
```

---
id: ASSIST-006
category: AI Assistant
title: Deterministic Regex as Safety Net for LLM Misclassification
tags: [regex, fallback, sector, intent-router, deterministic, safety-net]
---
**Problem**: Even after prompt tuning, the LLM occasionally misclassifies sector queries as "Market Overview" under certain phrasings.
**Lesson**: Add a deterministic post-processing step after LLM routing. If the LLM returns "Market Overview" but the query contains the word "sector", override to "Sector Ranking". This catch costs nothing and prevents the most common class of misclassification.
**Fix Pattern**:
```python
if intent == "Market Overview" and re.search(r'\bsectors?\b', query, re.IGNORECASE):
    intent = "Sector Ranking"
```

---
id: ASSIST-007
category: AI Assistant
title: Rank Semantics in LLM Explanation Prompts
tags: [rank, llm, explanation, prompt, semantics, lower-is-better]
---
**Problem**: The LLM described stocks with smaller rank numbers (e.g., Rank 1) as "worse" because numerically 1 < 200.
**Lesson**: Explicitly state rank semantics in the system prompt for explanation generation:
> "A mathematically smaller rank (e.g., 1) is a BETTER ('higher') structural rank than a larger number (e.g., 200)."

---
id: MAINT-001
category: Maintenance Workflow
title: Standard DB Sync + Verification Workflow
tags: [maintenance, database, scripts, update, verify, audit, production]
---
**Lesson**: The standard workflow to keep the production database current is:
1. Run `scripts/update_db_all_nse.py` targeting the production `DATABASE_URL` (NSE 500 universe).
2. Verify with `psql`: check `max(date)` in both `prices` and `scores` tables and confirm record counts.
3. Run `scripts/audit_system_integrity.py` to verify no look-ahead bias and bayesian shrinkage is applied.
4. If TTL/session logic changed, redeploy the dashboard so new containers pick up the changes.
5. Force-recompute Auto-N calibration via the "Force Recompute (N)" button in the web UI if the date has advanced.

---
id: MAINT-002
category: Maintenance Workflow
title: Full Deployment Checklist (Cloud Run)
tags: [deployment, checklist, cloud-run, memory, oauth, secrets, gcp]
---
**Lesson**: Before deploying to Cloud Run, verify:
1. `--memory 8Gi --max-instances 1` (peak rollout = 16GiB, within 40GiB quota)
2. `engine.dispose(close=False)` is in all multiprocessing worker functions
3. `secrets.toml` updated in GCP Secret Manager (`gcloud secrets versions add ...`)
4. OAuth `redirect_uri` in `secrets.toml` matches the current domain exactly
5. `gcloud config set gcloudignore/enabled false` if on Windows to avoid Unicode crash
6. `.dockerignore` excludes `__pycache__/`, `.agents/`, `.git/`, `*.docx`, `*.pyc`
