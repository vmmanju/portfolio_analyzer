"""
Phase 9: Interactive Research Interface (Streamlit).

Research-grade UI: Portfolio, Stock Recommendation, Research modes.
Calls existing backend services. No backend logic rewritten.

Parameter flow:
  Sidebar -> session_state (start_date, end_date, strategy, regime_mode, risk_profile, top_n, mode).
  Portfolio: backtest.run_backtest(start_date, end_date, strategy, top_n, target_vol from regime_mode).
  Stock: recommendation.get_stock_signal(symbol), generate_recommendation(signal, RISK_MAP[risk_profile]).
  Research: research_validation.run_walk_forward, run_weight_sensitivity, run_regime_analysis (cached by strategy, top_n).

Caching:
  _cached_backtest: keyed by (start_date, end_date, strategy, regime_mode, top_n), ttl=300s.
  _cached_walk_forward, _cached_weight_sensitivity, _cached_regime_analysis: keyed by (strategy, top_n), ttl=600s.
"""

import sys
from pathlib import Path

# Ensure project root is first so "app" is the backend package, not this file
_root = Path(__file__).resolve().parent.parent
_root_str = str(_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
else:
    # Ensure it's first
    sys.path.remove(_root_str)
    sys.path.insert(0, _root_str)

from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time
from sqlalchemy import select

from app.database import get_db_context
from app.models import Factor, Score, Stock


# --- Backend imports (no logic here) ---
from services.backtest import run_backtest, STRATEGY_EQUAL_WEIGHT, STRATEGY_INVERSE_VOL
from services.portfolio import (
    construct_equal_weight_portfolio,
    construct_inverse_vol_portfolio,
    get_latest_scoring_date,
    load_ranked_stocks,
)
from services.recommendation import get_stock_signal, generate_recommendation
from services.research_validation import (
    run_regime_analysis,
    run_weight_sensitivity,
    run_walk_forward,
)
from services.research_validation import generate_validation_report
from services.auto_diversified_portfolio import STRATEGY_AUTO_HYBRID
import os
from services.universe_loader import load_nse_top_100, sync_universe_with_db
from services import portfolio_comparison as pc
from services import user_portfolios as up_store
from services import portfolio_tracker as pt_store
from services import sector_analytics as sa
from app.auth import require_login
from app.config import settings



def _safe_rerun() -> None:
    """Safely request a Streamlit rerun across Streamlit versions.

    Tries `st.experimental_rerun()` first. If unavailable, falls back to
    toggling query params which triggers a rerun, and finally toggles a
    session_state key and calls `st.stop()` as a last resort.
    """
    try:
        # Preferred API when available
        st.experimental_rerun()
        return
    except Exception:
        pass

    try:
        params = st.experimental_get_query_params() or {}
        params["_rerun"] = [str(int(time.time()))]
        st.experimental_set_query_params(**params)
        return
    except Exception:
        pass

    # Final fallback: toggle a session_state key and stop execution
    st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
    try:
        st.stop()
    except Exception:
        return


# --- Default date range from DB ---
@st.cache_data(ttl=60)
def _get_score_date_range() -> tuple[date, date]:
    """Earliest and latest score dates."""
    with get_db_context() as db:
        min_row = db.execute(select(Score.date).order_by(Score.date).limit(1)).scalars().first()
        max_row = db.execute(select(Score.date).order_by(Score.date.desc()).limit(1)).scalars().first()
    if min_row is None or max_row is None:
        today = date.today()
        return today - timedelta(days=365), today
    return min_row, max_row


# --- Cached backtest (memoized by params) ---
@st.cache_data(ttl=300)
def _cached_backtest(
    start_date: date,
    end_date: date,
    strategy: str,
    regime_mode: str,
    top_n: int,
    selected_symbols: list[str] | None = None,
    custom_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_vol = 0.15 if regime_mode == "volatility_targeting" else None
    return run_backtest(
        strategy=strategy,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
        target_vol=target_vol,
        selected_symbols=selected_symbols,
        custom_weights=custom_weights,
    )


@st.cache_data(ttl=600)
def _cached_available_symbols() -> list[str]:
    """Return sorted list of symbols from DB; if empty, attempt NSE Top 100 fetch+sync."""
    with get_db_context() as db:
        rows = db.execute(select(Stock.symbol)).scalars().all()
    if rows:
        return sorted(rows)

    # attempt to fetch and sync from NSE
    try:
        lst = load_nse_top_100()
        if lst:
            sync_universe_with_db(lst)
            with get_db_context() as db:
                rows = db.execute(select(Stock.symbol)).scalars().all()
            return sorted(rows) if rows else []
    except Exception:
        return []
    return []


# --- Current allocation (latest score date + selected strategy/top_n) ---
@st.cache_data(ttl=300)
def _get_current_allocation(strategy: str, top_n: int, selected_symbols: list[str] | None = None) -> pd.DataFrame:
    latest = get_latest_scoring_date()
    if latest is None:
        return pd.DataFrame(columns=["symbol", "weight", "composite_score", "volatility_score"])
    if strategy == STRATEGY_EQUAL_WEIGHT:
        alloc = construct_equal_weight_portfolio(latest, top_n=top_n, selected_symbols=selected_symbols)
    else:
        alloc = construct_inverse_vol_portfolio(latest, top_n=top_n, selected_symbols=selected_symbols)
    if alloc.empty:
        return pd.DataFrame(columns=["symbol", "weight", "composite_score", "volatility_score"])
    ranked = load_ranked_stocks(latest, selected_symbols=selected_symbols)
    merged = alloc.merge(
        ranked[["stock_id", "symbol", "composite_score", "volatility_score"]],
        on="stock_id",
        how="left",
    )
    return merged[["symbol", "weight", "composite_score", "volatility_score"]].sort_values(
        "weight", ascending=False
    )


# --- Risk profile UI -> backend ---
RISK_MAP = {"low": "conservative", "medium": "moderate", "high": "aggressive"}


def _init_session_state() -> None:
    default_start, default_end = _get_score_date_range()
    # Default start date preference: 2nd July 2024, clamped to available DB range
    preferred = date(2024, 7, 2)
    if "start_date" not in st.session_state:
        start_choice = preferred
        if start_choice < default_start:
            start_choice = default_start
        if start_choice > default_end:
            start_choice = default_end
        st.session_state.start_date = start_choice
    if "end_date" not in st.session_state or st.session_state.end_date < default_end:
        st.session_state.end_date = default_end
    if "strategy" not in st.session_state:
        st.session_state.strategy = STRATEGY_EQUAL_WEIGHT
    if "regime_mode" not in st.session_state:
        st.session_state.regime_mode = "static"
    if "risk_profile" not in st.session_state:
        st.session_state.risk_profile = "medium"
    if "top_n" not in st.session_state:
        st.session_state.top_n = 5
    if "mode" not in st.session_state:
        st.session_state.mode = "Portfolio Mode"
    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = None
    # user_portfolios: used by Portfolio Comparison
    if "user_portfolios" not in st.session_state:
        try:
            _uid = st.session_state.get("user_id")
            loaded = up_store.load_user_portfolios(user_id=_uid)
            st.session_state.user_portfolios = loaded or []
        except Exception:
            st.session_state.user_portfolios = []
    # pm_portfolios: user-defined portfolios managed inside Portfolio Mode
    if "pm_portfolios" not in st.session_state:
        try:
            _uid = st.session_state.get("user_id")
            loaded = up_store.load_user_portfolios(user_id=_uid)
            st.session_state.pm_portfolios = loaded or []
        except Exception:
            st.session_state.pm_portfolios = []
    # Track which allocations have been refreshed today (keyed by portfolio name)
    if "pm_last_refresh" not in st.session_state:
        st.session_state.pm_last_refresh = {}  # name -> date string
    if "pm_allocations" not in st.session_state:
        st.session_state.pm_allocations = {}   # name -> allocation result dict
    if "pm_backtest_results" not in st.session_state:
        st.session_state.pm_backtest_results = {}  # name -> {curve, summary}
    if "pm_tracker_rows" not in st.session_state:
        try:
            _uid = st.session_state.get("user_id")
            st.session_state.pm_tracker_rows = pt_store.load_tracked_positions(user_id=_uid)
        except Exception:
            st.session_state.pm_tracker_rows = []
    # Hybrid auto portfolio settings
    if "include_hybrid_portfolio" not in st.session_state:
        st.session_state.include_hybrid_portfolio = False
    if "hybrid_top_n" not in st.session_state:
        st.session_state.hybrid_top_n = "auto"
    if "hybrid_auto_n" not in st.session_state:
        st.session_state.hybrid_auto_n = True
    if "hybrid_max_corr" not in st.session_state:
        st.session_state.hybrid_max_corr = 0.80
    if "hybrid_min_score" not in st.session_state:
        st.session_state.hybrid_min_score = 0.0
    if "use_multiprocessing" not in st.session_state:
        st.session_state.use_multiprocessing = False


def _sidebar() -> None:
    st.sidebar.header("Global parameters")
    
    st.session_state.mode = st.sidebar.radio(
        "Mode",
        ["Home", "Portfolio Mode", "Portfolio Comparison", "Assistant"],
        index=0,
    )
    st.sidebar.divider()

    default_start, default_end = _get_score_date_range()

    st.session_state.start_date = st.sidebar.date_input(
        "Start date",
        value=st.session_state.get("start_date", default_start),
        min_value=default_start,
        max_value=default_end,
    )
    st.session_state.end_date = st.sidebar.date_input(
        "End date",
        value=st.session_state.get("end_date", default_end),
        min_value=st.session_state.start_date,
        max_value=default_end,
    )

    st.sidebar.divider()
    st.session_state.strategy = st.sidebar.selectbox(
        "Strategy",
        [STRATEGY_EQUAL_WEIGHT, STRATEGY_INVERSE_VOL, STRATEGY_AUTO_HYBRID],
        index=0,
    )
    st.session_state.regime_mode = st.sidebar.selectbox(
        "Regime strategy",
        ["static", "volatility_targeting", "regime_adaptive"],
        index=0,
        help="regime_adaptive: placeholder for future",
    )
    
    st.sidebar.divider()
    st.session_state.risk_profile = st.sidebar.selectbox(
        "Risk profile",
        ["low", "medium", "high"],
        index=1,
    )
    st.session_state.top_n = st.sidebar.slider("Top N", min_value=5, max_value=50, value=st.session_state.get("top_n", 5), step=1)
    # selected_stocks kept as empty in session_state (no longer shown in sidebar)
    st.session_state.selected_stocks = None

    # Load available symbols silently (used by Portfolio Comparison Lab editors)
    available = _cached_available_symbols()

    # --- Portfolio Comparison UI (sidebar) ---
    st.sidebar.divider()
    st.sidebar.header("Portfolio Comparison Lab")
    st.sidebar.markdown(
        "Expand a portfolio below to pick stocks and settings. "
        "When ready, switch to **Portfolio Comparison** mode and press **Run comparison**."
    )

    # Add portfolio button
    c_add, c_spacer = st.sidebar.columns([3, 1])
    if c_add.button("➕ Add Portfolio"):
        cur = st.session_state.get("user_portfolios", [])
        if len(cur) >= 5:
            st.sidebar.warning("Maximum 5 portfolios reached.")
        else:
            cur.append({
                "name": f"Portfolio {len(cur) + 1}",
                "symbols": [],
                "strategy": st.session_state.get("strategy", STRATEGY_EQUAL_WEIGHT),
                "regime_mode": st.session_state.get("regime_mode", "static"),
                "top_n": st.session_state.get("top_n", 5),
            })
            st.session_state.user_portfolios = cur

    # Per-portfolio editors (max 5)
    up_list = st.session_state.get("user_portfolios", [])
    for i, p in enumerate(list(up_list)):
        title = p.get("name", f"Portfolio {i+1}")
        with st.sidebar.expander(title, expanded=False):
            name = st.text_input("Name", value=p.get("name", f"Portfolio {i+1}"), key=f"pf_name_{i}")
            syms = st.multiselect(
                "Stocks",
                options=available,
                default=[s for s in p.get("symbols", []) if s in available],
                key=f"pf_symbols_{i}",
            )
            strategy_sel = st.selectbox(
                "Strategy",
                [STRATEGY_EQUAL_WEIGHT, STRATEGY_INVERSE_VOL],
                index=0 if p.get("strategy", STRATEGY_EQUAL_WEIGHT) == STRATEGY_EQUAL_WEIGHT else 1,
                key=f"pf_strategy_{i}",
            )
            regime_sel = st.selectbox(
                "Regime mode",
                ["static", "volatility_targeting", "regime_adaptive"],
                index=["static", "volatility_targeting", "regime_adaptive"].index(p.get("regime_mode", "static")),
                key=f"pf_regime_{i}",
            )
            topn_sel = st.slider("Top N", min_value=5, max_value=50, value=p.get("top_n", 5), key=f"pf_topn_{i}")
            if st.button("Clear stocks", key=f"pf_clear_{i}"):
                st.session_state[f"pf_symbols_{i}"] = []
                st.session_state["_pc_run"] = False
            remove_key = f"pf_remove_{i}"
            if st.button("Remove Portfolio", key=remove_key):
                name_to_remove = st.session_state.get(f"pf_name_{i}", p.get("name"))
                try:
                    up_store.delete_user_portfolio_by_name(name_to_remove, user_id=st.session_state.get("user_id"))
                except Exception:
                    pass
                lst = st.session_state.get("user_portfolios", [])
                if i < len(lst):
                    lst.pop(i)
                    st.session_state.user_portfolios = lst
                    _safe_rerun()

    # Rebuild session_state.user_portfolios from widget values
    new_list = []
    for i in range(len(st.session_state.get("user_portfolios", []))):
        new_list.append(
            {
                "name": st.session_state.get(f"pf_name_{i}", f"Portfolio {i+1}"),
                "symbols": st.session_state.get(f"pf_symbols_{i}", []),
                "strategy": st.session_state.get(f"pf_strategy_{i}", STRATEGY_EQUAL_WEIGHT),
                "regime_mode": st.session_state.get(f"pf_regime_{i}", "static"),
                "top_n": st.session_state.get(f"pf_topn_{i}", st.session_state.get("top_n", 5)),
            }
        )

    # Check if portfolios changed (or if dates / global hybrid settings changed)
    _current_global_state = {
        "start": st.session_state.start_date,
        "end": st.session_state.end_date,
        "hybrid_on": st.session_state.get("include_hybrid_portfolio", False),
        "hybrid_top_n": st.session_state.get("hybrid_top_n", 15),
    }

    if (
        new_list != st.session_state.get("user_portfolios", [])
        or _current_global_state != st.session_state.get("_pc_global_state", {})
    ):
        st.session_state["_pc_run"] = False
        st.session_state["_pc_global_state"] = _current_global_state
        
    st.session_state.user_portfolios = new_list

    # Save / Load controls for persistence
    c_save, c_load = st.sidebar.columns([1, 1])
    if c_save.button("Save to DB"):
        try:
            saved = up_store.save_portfolios(new_list, user_id=st.session_state.get("user_id"))
            st.sidebar.success(f"Saved {len(saved)} portfolios to DB.")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")
    if c_load.button("Load from DB"):
        try:
            loaded = up_store.load_user_portfolios(user_id=st.session_state.get("user_id"))
            st.session_state.user_portfolios = loaded or []
            _safe_rerun()
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")

    # ── Auto Hybrid Portfolio ──────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**🤖 Auto Diversified Hybrid**")
    st.sidebar.checkbox(
        "Include in comparison",
        value=st.session_state.get("include_hybrid_portfolio", False),
        key="include_hybrid_portfolio",
        help="Adds a correlation-filtered, inverse-vol + score-tilted monthly portfolio to the comparison",
    )
    if st.session_state.get("include_hybrid_portfolio"):
        with st.sidebar.expander("⚙️ Hybrid settings", expanded=False):
            st.checkbox("Automatic N Selection", value=st.session_state.get("hybrid_auto_n", True), key="hybrid_auto_n")
            if not st.session_state.get("hybrid_auto_n", True):
                st.slider("Top N stocks", 10, 20, 
                          st.session_state.get("hybrid_top_n", 15) if isinstance(st.session_state.get("hybrid_top_n"), int) else 15, 
                          1, key="hybrid_top_n")
            else:
                st.session_state.hybrid_top_n = "auto"
            st.slider("Max pairwise corr", 0.50, 0.95, st.session_state.get("hybrid_max_corr", 0.80), 0.05, key="hybrid_max_corr")
            st.number_input("Min composite score", 0.0, 2.0, st.session_state.get("hybrid_min_score", 0.0), 0.05, key="hybrid_min_score")

    # --- Multiprocessing (footer) ---
    st.sidebar.divider()
    st.session_state.use_multiprocessing = st.sidebar.checkbox(
        "Enable Multiprocessing",
        value=st.session_state.use_multiprocessing,
        help="Use multiprocessing to speed up intensive backtests across multiple cores."
    )


def render_home_mode() -> None:
    """Home Mode: Introduction to the AI Stock Engine and its features."""
    st.title("🏡 Introduction to AI Stock Engine")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the **AI Stock Engine**, a comprehensive platform for building, analyzing, and backtesting 
    quantitative equity portfolios on the Indian Stock Market (NSE).

    Use the sidebar on the left to navigate between different tools and configure global parameters that affect your analysis.
    
    ### 🧭 Modes Overview
    
    - **Home**: This introductory page.
    - **Portfolio Mode**: Build and manage your own custom portfolio. You can select specific stocks, adjust their weights, test Stop-Loss thresholds, and analyze their historical performance.
    - **Portfolio Comparison**: The interactive Lab environment. Pit multiple user-created portfolios or system strategies against each other to compare their metrics natively.
    - **Assistant**: Natural language AI agent ready to analyze your existing strategies, give automated market breakdowns, and answer complex financial queries dynamically using LLMs.

    ---

    ### ⚙️ Global Parameters Guide
    
    The settings in the sidebar globally dictate how the data engine builds and evaluates portfolios:

    - **Start/End Date**: The historical window for backtesting and analysis.
    - **Strategy**: The default capital allocation method used when building portfolios natively.
    - **Regime Strategy**: Defines market adaptability:
        - *Static*: Weights remain fixed through the period.
        - *Volatility Targeting*: Dynamically scales portfolio leverage/cash to target a specific volatility level (e.g., 15%).
        - *Regime Adaptive*: Advanced machine learning model that dynamically adjusts factor exposures based on the current market environment (bull, bear, volatile).
    - **Risk Profile**: Affects optimization constraints logic (Low = conservative bounds, High = aggressive growth chasing).
    - **Top N**: The maximum number of top-ranked stocks the engine will select from the universe.
    
    **To get started, expand the sidebar on the left and select 'Portfolio Mode' or 'Assistant'!**
    """)


def render_portfolio_mode() -> None:
    """Portfolio Mode: manage user-defined portfolio with custom weights and Stop-Loss diagnostics."""
    from datetime import date as _date

    st.subheader("📁 Portfolio Mode — My Custom Portfolio")
    st.caption(
        "Define your own portfolio with specific stocks and weights. "
        "Run full performance analysis and view Stop-Loss Risk diagnostics."
    )

    available = _cached_available_symbols()
    today_str = str(_date.today())
    user_id = st.session_state.get("user_id")

    pm_portfolios: list = st.session_state.get("pm_portfolios", [])
    if not pm_portfolios:
        try:
            loaded = up_store.load_user_portfolios(user_id=st.session_state.get("user_id"))
            if loaded:
                pm_portfolios = [loaded[0]]
        except Exception:
            pass

    if not pm_portfolios:
        pm_portfolios = [{
            "name": "My Custom Portfolio",
            "symbols": [],
            "strategy": "custom",
            "weights": {},
            "regime_mode": "static",
            "top_n": 50,
        }]

    # Only one portfolio is managed here
    p = pm_portfolios[0]
    pname = p.get("name", "My Custom Portfolio")
    
    pm_last_refresh: dict = st.session_state.get("pm_last_refresh", {})
    pm_allocations: dict = st.session_state.get("pm_allocations", {})
    pm_backtest_results: dict = st.session_state.get("pm_backtest_results", {})

    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")

    st.markdown("##### Portfolio Tracker")
    st.caption(
        "Track your invested amount against the latest value in the database. "
        "Quantity is used to calculate the current amount for each stock."
    )

    tracker_seed = st.session_state.get("pm_tracker_rows", [])
    tracker_df = pd.DataFrame(tracker_seed)
    if tracker_df.empty:
        tracker_df = pd.DataFrame(columns=pt_store.TRACKER_COLUMNS)
    for _col in pt_store.TRACKER_COLUMNS:
        if _col not in tracker_df.columns:
            tracker_df[_col] = None
    tracker_df = tracker_df[pt_store.TRACKER_COLUMNS]

    with st.expander("📌 Tracked Holdings", expanded=True):
        edited_tracker_df = st.data_editor(
            tracker_df,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="pm_tracker_editor",
            column_config={
                "symbol": st.column_config.SelectboxColumn(
                    "Stock",
                    options=available,
                    required=False,
                    help="Pick a stock available in the analyzer universe.",
                ),
                "invested_amount": st.column_config.NumberColumn(
                    "Invested Amount",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                    help="Total capital invested in this stock.",
                ),
                "quantity": st.column_config.NumberColumn(
                    "Quantity",
                    min_value=0.0001,
                    step=1.0,
                    format="%.4f",
                    help="Number of shares or units held.",
                ),
            },
        )
        tracker_rows = pt_store.normalize_tracker_positions(edited_tracker_df)
        st.session_state.pm_tracker_rows = tracker_rows

        tc1, tc2 = st.columns(2)
        if tc1.button("💾 Save Tracker"):
            try:
                saved_positions = pt_store.save_tracked_positions(tracker_rows, user_id=user_id)
                st.session_state.pm_tracker_rows = saved_positions
                st.success(f"Saved {len(saved_positions)} tracked position(s).")
            except Exception as exc:
                st.error(f"Tracker save failed: {exc}")
        if tc2.button("🔄 Reload Tracker"):
            try:
                st.session_state.pm_tracker_rows = pt_store.load_tracked_positions(user_id=user_id)
                _safe_rerun()
            except Exception as exc:
                st.error(f"Tracker reload failed: {exc}")

    tracker_snapshot = pt_store.build_tracker_snapshot(tracker_rows, user_id=user_id)
    tracker_positions_df = tracker_snapshot.get("positions_df", pd.DataFrame())
    tracker_summary = tracker_snapshot.get("summary", {})
    missing_tracker_prices = tracker_snapshot.get("missing_prices", [])

    if not tracker_positions_df.empty:
        tm1, tm2, tm3, tm4 = st.columns(4)
        total_pnl = tracker_summary.get("total_pnl", 0.0)
        pnl_pct = tracker_summary.get("total_pnl_pct")
        pnl_delta = f"{pnl_pct:.2%}" if pnl_pct is not None else None
        tm1.metric("Total Invested", f"₹{tracker_summary.get('total_invested', 0.0):,.2f}")
        tm2.metric("Current Value", f"₹{tracker_summary.get('total_current', 0.0):,.2f}")
        tm3.metric("Profit / Loss", f"₹{total_pnl:,.2f}", delta=pnl_delta)
        tm4.metric(
            "Priced Positions",
            f"{tracker_summary.get('priced_positions', 0)}/{tracker_summary.get('total_positions', 0)}",
        )
        latest_price_date = tracker_summary.get("latest_price_date")
        if latest_price_date:
            st.caption(f"Latest price snapshot: `{latest_price_date}`")
        if missing_tracker_prices:
            st.warning("Latest prices are missing for: " + ", ".join(sorted(missing_tracker_prices)))

        tracker_fig = go.Figure()
        tracker_fig.add_trace(
            go.Bar(
                name="Invested Amount",
                x=tracker_positions_df["Symbol"],
                y=tracker_positions_df["Invested Amount"],
                marker_color="#1f77b4",
            )
        )
        tracker_fig.add_trace(
            go.Bar(
                name="Current Amount",
                x=tracker_positions_df["Symbol"],
                y=tracker_positions_df["Current Amount"].fillna(0.0),
                marker_color="#2ca02c",
            )
        )
        tracker_fig.update_layout(
            barmode="group",
            height=380,
            margin=dict(t=30, b=20),
            yaxis_title="Amount (INR)",
            legend_title_text="",
        )
        st.plotly_chart(tracker_fig, use_container_width=True)

        st.dataframe(
            tracker_positions_df.style.format(
                {
                    "Invested Amount": "₹{:,.2f}",
                    "Quantity": "{:,.4f}",
                    "Average Cost": "₹{:,.2f}",
                    "Current Price": "₹{:,.2f}",
                    "Current Amount": "₹{:,.2f}",
                    "P&L": "₹{:,.2f}",
                    "P&L %": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Add tracked stocks above to compare invested amount with current value.")

    st.divider()

    with st.expander(f"⚙️ {pname} Configuration", expanded=True):
        new_name = st.text_input("Portfolio Name", value=pname, help="Name used to store in DB.")
        
        chosen_symbols = st.multiselect(
            "Stocks in this portfolio",
            options=available,
            default=[s for s in (p.get("symbols") or []) if s in available],
            help="Choose the stocks for this portfolio.",
        )
        
        strategy_sel = st.selectbox(
            "Weighting strategy",
            ["custom", STRATEGY_EQUAL_WEIGHT, STRATEGY_INVERSE_VOL],
            index=0 if p.get("strategy", "custom") == "custom" else (1 if p.get("strategy") == STRATEGY_EQUAL_WEIGHT else 2),
            help="Select how stocks should be weighted."
        )

        new_weights = p.get("weights", {})
        if strategy_sel == "custom" and chosen_symbols:
            st.write("Customize Weights (will be re-normalized to sum to 1.0)")
            default_w = 1.0 / len(chosen_symbols)
            # Create a dataframe for data editor
            custom_data = []
            for s in chosen_symbols:
                w = new_weights.get(s, default_w)
                custom_data.append({"symbol": s, "weight": float(w)})
                
            custom_df = pd.DataFrame(custom_data)
            edited_df = st.data_editor(custom_df, hide_index=True, use_container_width=True)
            new_weights = {row["symbol"]: float(row["weight"]) for _, row in edited_df.iterrows()}

        pm_portfolios[0] = {
            "name": new_name,
            "symbols": chosen_symbols,
            "strategy": strategy_sel,
            "regime_mode": "static",
            "top_n": 50,  # Ensure all chosen stocks can be processed
            "weights": new_weights,
        }
        st.session_state.pm_portfolios = pm_portfolios

        c_save, c_refresh = st.columns(2)
        if c_save.button("💾 Save to DB"):
            try:
                pid = up_store.save_user_portfolio(pm_portfolios[0], user_id=st.session_state.get("user_id"))
                st.success(f"✅ Portfolio saved (id={pid}).")
            except Exception as e:
                st.error(f"Save failed: {e}")

        if c_refresh.button("🔄 Refresh Today's Allocation"):
            if not chosen_symbols:
                st.warning("Add at least one stock before refreshing allocation.")
            else:
                with st.spinner("Computing allocation…"):
                    result = up_store.refresh_portfolio_allocation(pm_portfolios[0])
                if result.get("error"):
                    st.error(f"Refresh failed: {result['error']}")
                else:
                    pm_last_refresh[new_name] = today_str
                    pm_allocations[new_name] = result
                    st.session_state.pm_last_refresh = pm_last_refresh
                    st.session_state.pm_allocations = pm_allocations
                    st.success(f"✅ Allocation refreshed for **{new_name}**.")

    st.divider()

    alloc_result = pm_allocations.get(new_name)
    if alloc_result and alloc_result.get("allocation_df") is not None:
        adf = alloc_result["allocation_df"]
        if not adf.empty:
            st.markdown(f"**Current allocation & Stop-Loss** as of `{alloc_result['date']}`")
            
            # Enrich ADF with Stop-Loss metrics using Batch Analysis
            enriched_adf = adf.copy()
            if "stock_id" in enriched_adf.columns:
                stock_ids = enriched_adf["stock_id"].astype(int).tolist()
                try:
                    sls_profiles = _cached_analyze_universe_stop_loss(alloc_result['date'], stock_ids)
                    
                    risk_levels = []
                    triggers = []
                    for sid in stock_ids:
                        prof = sls_profiles.get(sid, {})
                        risk_levels.append(prof.get("risk_level", "Unknown"))
                        r_trig = prof.get('trigger_threshold')
                        triggers.append(f"{r_trig:.2f}" if pd.notna(r_trig) and r_trig else "N/A")
                    
                    enriched_adf["Risk Level"] = risk_levels
                    enriched_adf["Suggested Stop Trigger"] = triggers
                except Exception as e:
                    st.error(f"Error computing Stop-Loss metrics: {e}")
                
                # Hide stock_id
                enriched_adf = enriched_adf.drop(columns=["stock_id"])
            
            # Clean up column names for display
            rename_map = {
                "symbol": "Stock",
                "weight": "Weight",
                "composite_score": "Composite Score",
                "volatility_score": "Volatility Score"
            }
            enriched_adf = enriched_adf.rename(columns=rename_map)

            def risk_color(val):
                if val == "Critical": return "background-color: #b71c1c; color: white"
                elif val == "High": return "background-color: #e65100; color: white"
                elif val == "Moderate": return "background-color: #f57f17; color: black"
                elif val == "Low": return "background-color: #1b5e20; color: white"
                return ""

            format_dict = {}
            if "Weight" in enriched_adf.columns: format_dict["Weight"] = "{:.2%}"
            if "Composite Score" in enriched_adf.columns: format_dict["Composite Score"] = "{:.2f}"
            if "Volatility Score" in enriched_adf.columns: format_dict["Volatility Score"] = "{:.2f}"

            style_subset = ["Risk Level"] if "Risk Level" in enriched_adf.columns else []
            styled_df = (
                enriched_adf.style
                .map(risk_color, subset=style_subset)
                .format(format_dict)
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No allocation data available. Click 'Refresh Today's Allocation'.")
    else:
        st.info("Allocation not yet computed for today. Click **🔄 Refresh Today** above.")

    st.divider()

    st.markdown("##### 📊 Performance Analysis")
    st.caption(f"Backtest period: **{start_date}** → **{end_date}**  | Strategy: `{strategy_sel}`")

    if st.button("▶ Run Analysis"):
        if not chosen_symbols:
            st.warning("Add at least one stock before running analysis.")
        elif start_date is None or end_date is None:
            st.warning("Set a date range in the sidebar first.")
        else:
            with st.spinner("Running backtest…"):
                try:
                    curve_bt, summary_bt = _cached_backtest(
                        start_date, end_date, strategy_sel, 
                        "static", 50,
                        selected_symbols=tuple(chosen_symbols),
                        custom_weights=new_weights if strategy_sel == "custom" else None
                    )
                    pm_backtest_results[new_name] = {
                        "curve": curve_bt,
                        "summary": summary_bt,
                    }
                    st.session_state.pm_backtest_results = pm_backtest_results
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

    bt_result = pm_backtest_results.get(new_name)
    if bt_result:
        summary_bt = bt_result.get("summary") or {}
        curve_bt = bt_result.get("curve")

        if summary_bt:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("CAGR",         f"{summary_bt.get('CAGR', 0):.2%}")
            m2.metric("Volatility",   f"{summary_bt.get('Volatility', 0):.2%}")
            m3.metric("Sharpe",       f"{summary_bt.get('Sharpe', 0):.2f}")
            m4.metric("Max Drawdown", f"{summary_bt.get('Max Drawdown', 0):.2%}")
            m5.metric("Total Return", f"{summary_bt.get('Total Return', 0):.2%}")
            m6.metric("Win Rate",     f"{summary_bt.get('Win Rate', 0):.1%}")

        if curve_bt is not None and not curve_bt.empty and "date" in curve_bt.columns:
            fig_bt = make_subplots(
                rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                vertical_spacing=0.06, subplot_titles=("Cumulative Return", "Drawdown"),
            )
            curve_bt_ = curve_bt.copy()
            curve_bt_["date"] = pd.to_datetime(curve_bt_["date"])
            fig_bt.add_trace(
                go.Scatter(
                    x=curve_bt_["date"], y=curve_bt_["cumulative_return"],
                    name="Cumulative Return", line=dict(color="#4CAF50", width=2),
                    fill="tozeroy", fillcolor="rgba(76,175,80,0.08)",
                ), row=1, col=1,
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=curve_bt_["date"], y=curve_bt_["drawdown"],
                    name="Drawdown", line=dict(color="#F44336", width=1.5),
                    fill="tozeroy", fillcolor="rgba(244,67,54,0.10)",
                ), row=2, col=1,
            )
            fig_bt.update_yaxes(title_text="Return", tickformat=".1%", row=1, col=1)
            fig_bt.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
            fig_bt.update_xaxes(title_text="Date", row=2, col=1)
            fig_bt.update_layout(height=480, margin=dict(t=40, b=20, l=0, r=0), legend=dict(orientation="h", y=1.04), hovermode="x unified")
            st.plotly_chart(fig_bt, use_container_width=True)


def _recommendation_color(rec: str) -> str:
    if rec in ("Strong Buy", "Buy"):
        return "green"
    if rec == "Hold":
        return "#CCCC00"
    if rec == "Reduce":
        return "orange"
    if rec == "Avoid":
        return "red"
    return "gray"



# ─────────────────────────────────────────────────────────────────────────────
# Error Calibration UI helpers  (Model Error Calibration section)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _cached_error_calibration() -> dict:
    """Fetch the latest model calibration record from DB, cached for 10 minutes.

    Returns the same dict shape as get_current_coefficients():
        calibration_source : 'cached' | 'none'
        window_start / window_end : date | None
        intercept, coefficients, r_squared, n_observations, blend_weight

    Raises no exceptions — returns a 'none'-source dict on any error.
    """
    try:
        from services.error_model import get_current_coefficients
        return get_current_coefficients()
    except Exception as exc:
        return {
            "calibration_source": "none",
            "reason": f"Could not load calibration: {exc}",
            "intercept": float("nan"),
            "coefficients": {f: float("nan") for f in ["momentum", "quality", "value", "volatility"]},
            "r_squared": float("nan"),
            "n_observations": 0,
            "window_start": None,
            "window_end": None,
            "blend_weight": float("nan"),
        }


def _interp_text(factor: str, beta: float) -> str:
    """Return a one-line human interpretation of a factor coefficient sign."""
    if abs(beta) < 1e-6:
        return f"**{factor.capitalize()}** coefficient ≈ 0 — no systematic bias detected."
    direction = "positive" if beta > 0 else "negative"
    interpretations: dict[str, tuple[str, str]] = {
        "momentum":   (
            "Model *underestimates* high-momentum stocks — actual returns exceed prediction.",
            "Model *overestimates* high-momentum stocks — actual returns trail prediction.",
        ),
        "quality":    (
            "Model *underestimates* high-quality stocks.",
            "Model *overestimates* high-quality stocks.",
        ),
        "value":      (
            "Model *underestimates* high-value stocks.",
            "Model *overestimates* high-value stocks.",
        ),
        "volatility": (
            "Model *underestimates* high-volatility stocks — riskier than priced.",
            "Model *overestimates* high-volatility stocks — less risky than priced.",
        ),
    }
    pos_text, neg_text = interpretations.get(factor, ("Positive bias detected.", "Negative bias detected."))
    return pos_text if beta > 0 else neg_text


def render_error_calibration_section() -> None:
    """Render the '🔬 Model Error Calibration' expander inside Stock Mode.

    Displays:
    - Calibration window, intercept, R², n_observations, blend_weight
    - Bar chart of coefficient magnitudes (green=positive, red=negative)
    - One-line interpretation per factor
    - Next scheduled update date (last_end + 6 months)
    - Warning if calibration is older than 6 months

    Design rules:
    - Reads from DB via _cached_error_calibration() — NO recomputation.
    - All rendering is defensive: gracefully handles missing / NaN values.
    """
    import math
    from datetime import date as _date

    _INTERVAL_MONTHS = 6
    FACTOR_NICE = {"momentum": "Momentum", "quality": "Quality",
                   "value": "Value", "volatility": "Volatility"}

    with st.expander("🔬 Model Error Calibration", expanded=False):

        calib = _cached_error_calibration()
        source  = calib.get("calibration_source", "none")

        if source == "none":
            st.warning(
                "No calibration record found in the database yet. "
                "Click **▶ Run Calibration** to compute error coefficients over the "
                "last 5 years. This may take a minute."
            )
            if st.button("▶ Run Calibration Now", key="calib_run_btn"):
                with st.spinner("Running 5-year error model calibration…"):
                    try:
                        from services.error_model import update_error_model_if_due
                        result = update_error_model_if_due(
                            current_date=_date.today(),
                            force=True,
                        )
                        status = result.get("updated", False)
                        if status:
                            st.success(
                                f"✅ Calibration complete!  "
                                f"N={result.get('n_observations', '?')}  "
                                f"R²={result.get('r_squared', float('nan')):.4f}"
                            )
                            # Bust the cache so the panel reloads fresh data
                            _cached_error_calibration.clear()
                        else:
                            reason = result.get("reason", "unknown")
                            st.error(
                                f"Calibration did not complete: {reason}. "
                                "Check that you have at least 30 months of price and "
                                "score data in the database."
                            )
                    except Exception as _ce:
                        st.error(f"Calibration failed: {_ce}")
            return

        # ── Metadata row ──────────────────────────────────────────────────────
        win_start = calib.get("window_start")
        win_end   = calib.get("window_end")
        r2        = calib.get("r_squared", float("nan"))
        n_obs     = calib.get("n_observations", 0)
        intercept = calib.get("intercept", float("nan"))
        blend_w   = calib.get("blend_weight", float("nan"))
        coefs     = calib.get("coefficients") or {}

        # ── Staleness warning ─────────────────────────────────────────────────
        today = _date.today()
        is_stale = False
        next_update: Optional[_date] = None
        if win_end:
            # Approximate: add 6 calendar months
            ny = win_end.year + (win_end.month + _INTERVAL_MONTHS - 1) // 12
            nm = (win_end.month + _INTERVAL_MONTHS - 1) % 12 + 1
            try:
                next_update = _date(ny, nm, min(win_end.day, 28))
            except ValueError:
                next_update = _date(ny, nm, 28)
            months_elapsed = (today.year - win_end.year) * 12 + (today.month - win_end.month)
            is_stale = months_elapsed >= _INTERVAL_MONTHS

        if is_stale:
            st.warning(
                f"⚠️ Calibration is **{months_elapsed} months** old (last updated {win_end}). "
                f"Next update was due on **{next_update}**. Consider triggering a manual recalibration."
            )
        else:
            st.success(
                f"✅ Calibration is current.  "
                + (f"Next scheduled update: **{next_update}**" if next_update else "")
            )

        # ── Key metadata metrics ───────────────────────────────────────────────
        st.markdown("##### Calibration Window & Diagnostics")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Window Start",     str(win_start) if win_start else "—")
        mc2.metric("Window End",       str(win_end)   if win_end   else "—")
        mc3.metric("R²",               f"{r2:.4f}"    if not (r2 is None or (isinstance(r2, float) and math.isnan(r2))) else "—")
        mc4.metric("Observations (n)", str(n_obs)      if n_obs    else "—")
        mc5.metric("Blend weight (w)", f"{blend_w:.2f}" if not (blend_w is None or (isinstance(blend_w, float) and math.isnan(blend_w))) else "—")

        # ── Intercept ─────────────────────────────────────────────────────────
        intercept_str = f"{intercept:+.6f}" if not (intercept is None or (isinstance(intercept, float) and math.isnan(intercept))) else "—"
        icol = "🟢" if (intercept is not None and not (isinstance(intercept, float) and math.isnan(intercept)) and intercept >= 0) else "🔴"
        st.markdown(f"**Intercept (β₀):** `{intercept_str}` {icol}")

        st.divider()

        # ── Coefficient bar chart ─────────────────────────────────────────────
        st.markdown("##### Factor Coefficients β")
        labels, values, colours = [], [], []
        for fkey, fname in FACTOR_NICE.items():
            v = coefs.get(fkey)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = 0.0
            labels.append(fname)
            values.append(float(v))
            colours.append("#4CAF50" if v >= 0 else "#F44336")

        fig_bar = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colours,
                text=[f"{v:+.6f}" for v in values],
                textposition="outside",
                hovertemplate="%{x}<br>β = %{y:.6f}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            title="Coefficient Magnitudes (β per factor)",
            yaxis_title="Coefficient value",
            xaxis_title="Factor",
            yaxis_zeroline=True,
            yaxis_zerolinecolor="#888",
            height=350,
            margin=dict(t=40, b=20, l=0, r=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Per-factor interpretation ──────────────────────────────────────────
        st.markdown("##### Interpretation")
        for fkey, fname in FACTOR_NICE.items():
            v = coefs.get(fkey)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = 0.0
            # Badge
            pct = abs(float(v))
            if float(v) > 1e-6:
                badge  = '<span style="background:#1b5e20;color:#a5d6a7;padding:2px 8px;border-radius:4px;font-size:12px">▲ Positive bias</span>'
            elif float(v) < -1e-6:
                badge  = '<span style="background:#b71c1c;color:#ef9a9a;padding:2px 8px;border-radius:4px;font-size:12px">▼ Negative bias</span>'
            else:
                badge  = '<span style="background:#37474f;color:#cfd8dc;padding:2px 8px;border-radius:4px;font-size:12px">≈ Neutral</span>'

            interp = _interp_text(fkey, float(v))
            st.markdown(
                f"**{fname}** &nbsp; `β = {float(v):+.6f}` &nbsp; {badge}  \n"
                f"→ _{interp}_",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(
            f"Calibration source: `{source}` &nbsp;|&nbsp; "
            f"Blend weight w={blend_w:.2f} means posterior = w×new + (1−w)×prior. &nbsp;|&nbsp; "
            f"Last DB write: {win_end}"
        )


def render_stock_mode() -> None:
    st.subheader("Stock recommendation mode")
    risk_ui = st.session_state.risk_profile
    risk_backend = RISK_MAP.get(risk_ui, "moderate")

    symbol = st.text_input("Stock symbol", value="", placeholder="e.g. RELIANCE.NS").strip()
    if not symbol:
        st.caption("Enter a symbol and press Enter.")
        _render_stock_mode_footer()
        return

    signal = get_stock_signal(symbol)
    if signal is None:
        st.error(f"No data for symbol: {symbol}")
        _render_stock_mode_footer()
        return

    rec_dict = generate_recommendation(signal, risk_profile=risk_backend)

    # Combined database session to minimize roundtrips
    stock_id = None
    factor_vals = {}
    df_trend = pd.DataFrame()

    with get_db_context() as db:
        # 1. Resolve stock_id
        stock_id = db.execute(select(Stock.id).where(Stock.symbol == symbol)).scalars().first()
        
        if stock_id:
            # 2. Factor breakdown (latest)
            latest_score_date = db.execute(
                select(Score.date).where(Score.stock_id == stock_id).order_by(Score.date.desc()).limit(1)
            ).scalars().first()
            
            if latest_score_date:
                frow = db.execute(
                    select(
                        Factor.momentum_score,
                        Factor.quality_score,
                        Factor.value_score,
                        Factor.volatility_score,
                    ).where(Factor.stock_id == stock_id, Factor.date == latest_score_date)
                ).first()
                if frow:
                    factor_vals = {
                        "Momentum": frow[0] or 0,
                        "Quality": frow[1] or 0,
                        "Value": frow[2] or 0,
                        "Volatility": frow[3] or 0,
                    }
            
            # 3. Score trend
            start = st.session_state.start_date
            end = st.session_state.end_date
            trend_rows = db.execute(
                select(Score.date, Score.composite_score)
                .where(
                    Score.stock_id == stock_id,
                    Score.date >= start,
                    Score.date <= end,
                )
                .order_by(Score.date)
            ).all()
            if trend_rows:
                df_trend = pd.DataFrame(trend_rows, columns=["date", "composite_score"])

    # Recommendation card
    rec_text = rec_dict.get("recommendation", "Hold")
    color = _recommendation_color(rec_text)
    st.markdown(f'### Recommendation: <span style="color: {color}; font-weight: bold;">{rec_text}</span>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conviction", rec_dict.get("conviction", ""))
    col2.metric("Rank", signal.get("rank", "—"))
    
    # Show underlying blended final score
    f_score = rec_dict.get("final_stock_score", signal.get("composite_score", 0))
    col3.metric("Final Score", f"{f_score:.3f}")
    
    # Show Stop Risk Score
    sls_score = rec_dict.get("stop_loss_score", 50.0)
    col4.metric("Stop Risk", f"{sls_score:.1f}/100")
    
    alloc_range = rec_dict.get("suggested_allocation_range", (0, 0))
    st.caption(f"Suggested allocation range: {alloc_range[0]:.1%} – {alloc_range[1]:.1%}")

    # Factor breakdown
    with st.expander("Factor breakdown"):
        if not factor_vals:
            factor_vals = {
                "Momentum": signal.get("momentum_score") or 0,
                "Quality": 0,
                "Value": 0,
                "Volatility": signal.get("volatility_score") or 0,
            }
        fig = go.Figure(
            data=[go.Bar(x=list(factor_vals.keys()), y=list(factor_vals.values()))]
        )
        fig.update_layout(title="Factor scores (z-score)", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    # Risk Responsiveness
    # ... (Gauge Chart remains same) ...
    with st.expander("Risk Responsiveness"):
        rrc = signal.get("rrc_score", 50.0)
        fig_rrc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rrc,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "RRC Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
            }
        ))
        fig_rrc.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_rrc, use_container_width=True)
        
        st.markdown("**Stress-Period Behavior Summary**")
        if rrc > 60:
            st.success("This stock adapts well during volatility spikes. It shows positive correlation drift and speedy recovery traits.")
        elif rrc < 40:
            st.error("This stock shows delayed or weak response to rising risk. High drawdown risk during systematic stress.")
        else:
            st.info("This stock has a neutral responsiveness to market volatility spikes.")

    # Score trend
    with st.expander("Score trend"):
        if not df_trend.empty:
            fig = go.Figure(
                data=[go.Scatter(x=df_trend["date"], y=df_trend["composite_score"], mode="lines+markers")]
            )
            fig.update_layout(title="Composite score over time", xaxis_title="Date", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical score trend available for selected date range.")

    # Risk context
    with st.expander("Risk context"):
        st.write("**Regime:**", signal.get("regime", "—"))
        st.write("**Risk commentary:**", rec_dict.get("risk_commentary", ""))
        st.write("**Explanation:**", rec_dict.get("explanation", ""))

    # ── Stop-Loss Diagnostics ────────────────────────────────────────────────
    if stock_id:
        st.divider()
        st.markdown("#### Stop-Loss Risk Diagnostics")
        latest_date = get_latest_scoring_date()
        if latest_date:
            sls_profile = _cached_analyze_stock_stop_loss(latest_date, stock_id)
            if sls_profile:
                sl_score = sls_profile.get("stop_loss_score", 0.0)
                sl_risk = sls_profile.get("risk_level", "Unknown")
                
                # Color coding
                if sl_score >= 80:
                    sl_color = "red"
                    st.error(f"🚨 CRITICAL DOWNSIDE RISK (SLS = {sl_score:.1f}). Immediate Exit Subroutine Triggered.")
                elif sl_score >= 70:
                    sl_color = "orange"
                    if sl_score > 75:
                        st.warning(f"⚠️ High Stop-Loss Risk detected (SLS = {sl_score:.1f}). Consider reducing exposure.")
                elif sl_score >= 40:
                    sl_color = "gold"
                else:
                    sl_color = "green"
                    
                col_g1, col_g2, col_g3 = st.columns([1, 1, 1.5])
                
                with col_g1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sl_score,
                        title={"text": "SLS Score", "font": {"size": 14}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                            "bar": {"color": sl_color},
                            "steps": [
                                {"range": [0, 40], "color": "rgba(0, 255, 0, 0.1)"},
                                {"range": [40, 70], "color": "rgba(255, 215, 0, 0.1)"},
                                {"range": [70, 80], "color": "rgba(255, 165, 0, 0.1)"},
                                {"range": [80, 100], "color": "rgba(255, 0, 0, 0.1)"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=180, margin=dict(t=40, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_g2:
                    comps = sls_profile.get("components", {})
                    st.metric("Risk Level", sl_risk)
                    st.metric("Current Drawdown", f"{sls_profile.get('raw_drawdown', 0):.2%}")
                    st.metric("Suggested Stop Trigger", f"₹ {sls_profile.get('trigger_threshold', 0):.2f}")
                    st.metric("Regime Multiplier", "High" if comps.get('regime_stress_score', 0) > 0 else "Low")

                with col_g3:
                    if comps:
                        fig_bar = go.Figure(go.Bar(
                            x=list(comps.values()),
                            y=["Drawdown", "Volatility Z", "Momentum", "Error Bias", "Regime"],
                            orientation='h',
                            marker_color=sl_color
                        ))
                        fig_bar.update_layout(title="Component Breakdown (0-100 scale)", height=180, margin=dict(t=30, b=20, l=10, r=20), xaxis=dict(range=[0, 100]))
                        st.plotly_chart(fig_bar, use_container_width=True)

    _render_stock_mode_footer()


def _render_stock_mode_footer() -> None:
    st.divider()
    st.subheader("🏆 Top 10 Ranked Stocks")
    st.caption("Cross-sectional rank across all factors. Based on latest scoring date.")

    latest_date = get_latest_scoring_date()
    if not latest_date:
        st.info("No scoring date found. Run the data update script first.")
        return

    # Load basic ranking immediately (this is fast)
    top_stocks = load_ranked_stocks(latest_date)
    if top_stocks.empty:
        st.info("No ranking data available for the latest date.")
        return

    top_10 = top_stocks.nsmallest(10, 'rank')[
        ['stock_id', 'rank', 'symbol', 'composite_score', 'volatility_score']
    ].copy()

    # Check if user wants to see Risk Metrics (the slow part)
    show_risk = st.button("🔍 Analyze Risk Metrics (Stop-Loss)", key="btn_enrich_top10", 
                          help="Calculates suggested stop-loss triggers and risk levels for these 10 stocks. Takes ~3-5s.")
    
    if show_risk or st.session_state.get("top10_risk_loaded"):
        st.session_state["top10_risk_loaded"] = True
        with st.spinner("Analyzing risk profiles..."):
            stock_ids = top_10["stock_id"].astype(int).tolist()
            try:
                sls_profiles = _cached_analyze_universe_stop_loss(latest_date, stock_ids)
                risk_levels = []
                triggers = []
                for sid in stock_ids:
                    prof = sls_profiles.get(sid, {})
                    risk_levels.append(prof.get("risk_level", "Unknown"))
                    r_trig = prof.get('trigger_threshold')
                    triggers.append(f"{r_trig:.2f}" if pd.notna(r_trig) and r_trig else "N/A")
                top_10["Risk Level"] = risk_levels
                top_10["Suggested Stop Trigger"] = triggers
            except Exception as e:
                st.error(f"Error computing Stop-Loss metrics: {e}")

    # Format for display
    display_df = top_10.drop(columns=["stock_id"]).copy()
    display_df.rename(columns={
        "rank": "Rank",
        "symbol": "Symbol",
        "composite_score": "Composite Score",
        "volatility_score": "Volatility Score"
    }, inplace=True)

    def risk_color(val):
        if val == "Critical": return "background-color: #b71c1c; color: white"
        elif val == "High": return "background-color: #e65100; color: white"
        elif val == "Moderate": return "background-color: #f57f17; color: black"
        elif val == "Low": return "background-color: #1b5e20; color: white"
        return ""

    st.dataframe(
        display_df.style
        .map(risk_color, subset=["Risk Level"] if "Risk Level" in display_df.columns else [])
        .format({"Composite Score": "{:.3f}", "Volatility Score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"As of latest scoring date: {latest_date}")

    # Model Error Calibration section
    st.divider()
    render_error_calibration_section()


def _log_execution_time(task_name: str, start_time: float) -> None:
    import logging, time
    elapsed = time.time() - start_time
    # Use standard logging but also write to a dedicated file just in case
    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task: {task_name} | Multiprocessing: {st.session_state.get('use_multiprocessing', True)} | Time: {elapsed:.2f}s"
    print(msg)
    try:
        with open("analysis_times.log", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


@st.cache_data(ttl=600)
def _cached_walk_forward(strategy: str, top_n: int, start_date: date, end_date: date, selected_symbols: list[str] | None = None, use_multiprocessing: bool = True) -> pd.DataFrame:
    t0 = time.time()
    res = run_walk_forward(
        strategy=strategy, top_n=top_n, use_rolling=False, start_date=start_date, end_date=end_date, selected_symbols=selected_symbols, use_multiprocessing=use_multiprocessing
    )
    _log_execution_time("Walk-Forward Validation", t0)
    return res


@st.cache_data(ttl=600)
def _cached_weight_sensitivity(strategy: str, top_n: int, selected_symbols: list[str] | None = None, use_multiprocessing: bool = True) -> pd.DataFrame:
    t0 = time.time()
    res = run_weight_sensitivity(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols, use_multiprocessing=use_multiprocessing)
    _log_execution_time("Weight Sensitivity Analysis", t0)
    return res


@st.cache_data(ttl=600)
def _cached_regime_analysis(strategy: str, top_n: int, selected_symbols: list[str] | None = None) -> dict:
    return run_regime_analysis(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols)


@st.cache_data(ttl=600)
def _cached_analyze_stock_stop_loss(as_of_date: date, stock_id: int) -> dict:
    from services.stop_loss_engine import analyze_stock_stop_loss
    return analyze_stock_stop_loss(as_of_date, stock_id)


@st.cache_data(ttl=600)
def _cached_analyze_universe_stop_loss(as_of_date: date, stock_ids: list[int]) -> dict[int, dict]:
    from services.stop_loss_engine import analyze_universe_stop_loss
    return analyze_universe_stop_loss(as_of_date, list(stock_ids))


@st.cache_data(ttl=600)
def _cached_lightweight_analysis(strategy: str, top_n: int, start_date: date, end_date: date, selected_symbols: list[str] | None, regime_mode: str, use_multiprocessing: bool) -> dict:
    import services.portfolio_comparison as pc
    p = pc.UserPortfolio(
        name="Current Model",
        symbols=selected_symbols or [],
        strategy=strategy,
        regime_mode=regime_mode,
        top_n=top_n
    )
    t0 = time.time()
    res = pc.backtest_user_portfolios([p], start_date, end_date, use_multiprocessing=use_multiprocessing)
    _log_execution_time("Lightweight Metrics Analysis", t0)
    
    if res and "Current Model" in res:
        ratings = pc.compute_full_ratings(res)
        res["Current Model"]["ratings"] = ratings["ratings"].get("Current Model", {})
    return res

def render_research_mode() -> None:
    st.subheader("Institutional Model Governance & Research Console")
    strategy = st.session_state.strategy
    top_n    = st.session_state.top_n
    start    = st.session_state.start_date
    end      = st.session_state.end_date
    selected = st.session_state.get("selected_stocks") or None
    regime_mode = st.session_state.get("regime_mode", "static")

    _param_key = (strategy, top_n, str(start), str(end), str(sorted(selected or [])))

    # Gate lightweight analysis behind a button — avoids auto-running a full backtest on every page load
    _lw_key = "rm_lw_result"
    _lw_pkey = "rm_lw_param_key"
    if st.session_state.get(_lw_pkey) != _param_key:
        st.session_state.pop(_lw_key, None)

    if st.session_state.get(_lw_key) is None:
        st.info("Click **▶ Run Governance Analysis** to compile lightweight portfolio metrics for the current settings.")
        if not st.button("▶ Run Governance Analysis", key="btn_run_governance"):
            # Render placeholder sections so user can still trigger heavy analysis
            portfolio_results = {}
            base_res = None
        else:
            with st.spinner("Compiling Lightweight Metrics..."):
                base_res = _cached_lightweight_analysis(
                    strategy, top_n, start, end, selected, regime_mode, st.session_state.use_multiprocessing
                )
            st.session_state[_lw_key] = base_res
            st.session_state[_lw_pkey] = _param_key
    else:
        base_res = st.session_state[_lw_key]
        
    portfolio_results = {}
    if base_res and "Current Model" in base_res:
        cm = base_res["Current Model"]
        portfolio_results["metrics"] = cm.get("metrics", {})
        portfolio_results["ratings"] = cm.get("ratings", {})
        portfolio_results["avg_pairwise_corr"] = cm.get("avg_pairwise_corr", 0.5)
        portfolio_results["rrc_score"] = cm.get("rrc_score", 50.0)
        portfolio_results["warnings"] = cm.get("warnings", [])

    if base_res is None:
        # User hasn't run governance yet — return early; button is already shown above
        return
        
    _wf_key   = "rm_wf_result"
    _wf_pkey  = "rm_wf_param_key"
    if st.session_state.get(_wf_pkey) != _param_key:
        st.session_state.pop(_wf_key, None)
    wf_data = st.session_state.get(_wf_key)
    if wf_data is not None:
        portfolio_results["walk_forward"] = wf_data

    _ws_key  = "rm_ws_result"
    _ws_pkey = "rm_ws_param_key"
    if st.session_state.get(_ws_pkey) != _param_key:
        st.session_state.pop(_ws_key, None)
    ws_data = st.session_state.get(_ws_key)
    if ws_data is not None:
        portfolio_results["sensitivity"] = ws_data
        
    _rg_key  = "rm_rg_result"
    _rg_pkey = "rm_rg_param_key"
    if st.session_state.get(_rg_pkey) != _param_key:
        st.session_state.pop(_rg_key, None)
    rg_data = st.session_state.get(_rg_key)
    if rg_data is not None:
        portfolio_results["regime"] = rg_data
        
    # Model Governance Scoring runs ONLY for Auto-diversified Hybrid
    from services.model_governance import (
        run_overfitting_diagnostics,
        run_conservative_bias_check,
        compute_model_governance_score
    )
    
    # Ensure strategy comparison is robust to whitespace/case if needed, but constants are best
    # Robust strategy comparison
    is_hybrid = (str(strategy).lower() == STRATEGY_AUTO_HYBRID.lower())
    
    if is_hybrid:
        overfit = run_overfitting_diagnostics(portfolio_results)
        conservative = run_conservative_bias_check(portfolio_results)
        stability = portfolio_results.get("ratings", {}).get("stability_score", 50.0)
        gov_score_val = compute_model_governance_score(overfit, conservative, stability)
        gov_score_text = f"{gov_score_val:.1f} / 100"
    else:
        overfit = {"overfitting_risk": "N/A", "flags": [], "metrics": {}}
        conservative = {"conservative_bias": False, "classification": "N/A", "flags": []}
        gov_score_text = "N/A (Check Hybrid Strategy)"

    st.markdown(f"### Governance Health Score: **{gov_score_text}**")
    
    left_panel, right_panel = st.columns([1,1])
    
    with left_panel:
        st.markdown("#### Research Tests")
        
        # 1. Overfitting
        o_status = overfit['overfitting_risk'] if is_hybrid else "na"
        o_color = "🟢" if o_status == "Low" else "🟡" if o_status == "Moderate" else "🔴" if o_status == "High" else "⚪"
        with st.expander(f"{o_color} Overfitting Diagnostics"):
            st.write(f"**Risk Level:** {o_status}")
            if is_hybrid and overfit.get("flags"):
                for f in overfit["flags"]: st.write(f"- {f}")
            elif is_hybrid:
                st.write("- No critical overfitting flags detected.")
            else:
                st.info("Switch Global Strategy to 'Auto Diversified Hybrid' to enable governance scoring.")
                
        # 2. Conservative Bias
        c_status = conservative['classification'] if is_hybrid else "na"
        # 🟢 for Balanced or Slightly, 🔴 ONLY for Overly Conservative
        c_color = "🔴" if (conservative.get("conservative_bias") and is_hybrid) else "🟢"
        if not is_hybrid: c_color = "⚪"
        
        with st.expander(f"{c_color} Conservative Bias Check"):
            st.write(f"**Classification:** {c_status}")
            if is_hybrid:
                if conservative.get("flags"):
                    for f in conservative["flags"]: st.write(f"- {f}")
                else:
                    st.write("- Model is balanced.")
            else:
                st.info("Switch Global Strategy to 'Auto Diversified Hybrid' to enable governance scoring.")
                
        # 3. Walk-Forward
        with st.expander("Walk-Forward Gap"):
            if wf_data is not None:
                gap = (wf_data["train_Sharpe"] - wf_data["test_Sharpe"]).mean()
                wf_color = "🔴" if gap > 1.0 else "🟢"
                st.write(f"{wf_color} **Mean Gap:** {gap:.2f}")
                st.dataframe(wf_data, use_container_width=True)
            else:
                st.info("Heavy computation. Click below to run advanced analysis.")

        # 4. Sensitivity Stability
        with st.expander("Sensitivity Stability"):
            if ws_data is not None:
                sharpes = ws_data["Sharpe"].values
                chg = (sharpes.max() - sharpes.min()) / abs(sharpes.max()) if sharpes.max() > 0 else 0
                s_color = "🔴" if chg > 0.25 else "🟢"
                st.write(f"{s_color} **Sharpe Variation:** {chg*100:.1f}%")
                fig = go.Figure(data=[go.Scatter(x=ws_data["momentum_weight"], y=ws_data["Sharpe"], mode="lines+markers")])
                fig.update_layout(xaxis_title="Momentum weight", yaxis_title="Sharpe", height=200, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Heavy computation. Click below to run advanced analysis.")
                
        # 5. Regime Robustness
        with st.expander("Regime Robustness"):
            if rg_data is not None:
                h_cagr = rg_data.get("cagr_high_vol", 0)
                l_cagr = rg_data.get("cagr_low_vol", 0)
                st.write(f"**High Vol CAGR:** {h_cagr:.2%}")
                st.write(f"**Low Vol CAGR:** {l_cagr:.2%}")
            else:
                st.info("Heavy computation. Click below to run advanced analysis.")

        st.markdown("##### Advanced Analysis Controls")
        if st.button("▶ Run Heavy Analysis (Walk-Forward, Sensitivity, Regime)", key="btn_heavy"):
            with st.spinner("Executing Heavy Diagnostics..."):
                wf = _cached_walk_forward(strategy, top_n, start, end, selected_symbols=selected, use_multiprocessing=st.session_state.use_multiprocessing)
                st.session_state[_wf_key] = wf; st.session_state[_wf_pkey] = _param_key
                
                ws = _cached_weight_sensitivity(strategy, top_n, selected_symbols=selected, use_multiprocessing=st.session_state.use_multiprocessing)
                st.session_state[_ws_key] = ws; st.session_state[_ws_pkey] = _param_key
                
                rg = _cached_regime_analysis(strategy, top_n, selected_symbols=selected)
                st.session_state[_rg_key] = rg; st.session_state[_rg_pkey] = _param_key
                st.rerun()

    with right_panel:
        st.markdown("#### Factor Analytics & Interpretation")
        m = portfolio_results.get("metrics", {})
        r = portfolio_results.get("ratings", {})
        
        d = {
            "Metric": [
                "Sharpe", "Volatility", "Max Drawdown", "Stability Score", 
                "Composite Rating", "Beta", "Turnover", "Stop-Loss Score", "RRC"
            ],
            "Value": [
                f"{m.get('Sharpe', 0.0):.2f}" if m else "na",
                f"{m.get('Volatility', 0.0):.2%}" if m else "na",
                f"{m.get('MaxDrawdown', 0.0):.2%}" if m else "na",
                f"{r.get('stability_score', 0.0):.1f}/100" if r else "na",
                f"{r.get('composite_score', 0.0):.1f}/100" if r else "na",
                f"{m.get('Beta', 0.0):.2f}" if m else "na",
                f"{m.get('monthly_turnover', 0.0):.1%}" if m else "na",
                f"{r.get('stop_loss_score', 0.0):.1f}/100" if r else "na",
                f"{r.get('components', {}).get('rrc_component', 50.0):.1f}/100" if r else "na"
            ]
        }
        st.table(pd.DataFrame(d))
        
        st.markdown("##### Interpretation Ranges")
        st.caption("Sharpe: < 0.5 → Weak | 0.5–1 → Acceptable | 1–2 → Strong | > 2 → Exceptional")
        st.caption("Volatility: < 8% → Defensive | 8–15% → Moderate | 15–25% → Aggressive")
        st.caption("Stability Score: < 40 → Unstable | 40–70 → Moderate | > 70 → Strong")
        st.caption("Composite Rating: < 60 → Weak | 60–75 → Acceptable | 75–90 → Strong | > 90 → Excellent")
        
        st.markdown("##### Contextual Explanations")
        st.info("- Sharpe > 2 with low stability may indicate overfitting.\n- Low volatility + low CAGR suggests conservative bias.\n- High RRC improves resilience during regime shifts.")

    # ── Automatic N Selection ──────────────────────────────────────────────
    st.divider()
    col_n1, col_n2 = st.columns([3, 1])
    with col_n1:
        st.markdown("### Automatic N Selection")
    with col_n2:
        force_n = st.button("Force Recompute (N)", key="force_n")
        
    with st.spinner("Loading optimal portfolio breadth (N)..."):
        from services.auto_n_selector import update_optimal_n_if_due
        n_res = update_optimal_n_if_due(
            as_of_date=end,
            min_score=st.session_state.get("hybrid_min_score", 0.0),
            max_corr=st.session_state.get("hybrid_max_corr", 0.80),
            regime_mode=st.session_state.get("regime_mode", "volatility_targeting"),
            risk_profile=st.session_state.get("risk_profile", "medium"),
            force=force_n,
            user_id=st.session_state.get("user_id")
        )
        
    n_star = n_res.get("optimal_n", 15)
    df_eval = n_res.get("evaluation_table", pd.DataFrame())
    diag = n_res.get("diagnostics", {})
    
    col_nl, col_nr = st.columns([1, 1.5])
    with col_nl:
        st.markdown(f"**Selected Optimal N:** `{n_star}`")
        if n_res.get('calibration_source') == 'cached':
            st.caption(f"Last updated: {n_res.get('last_run_date', 'Unknown')} (Cached)")
        else:
            st.caption(f"Status: Computed Fresh")
            
        st.markdown("#### Diagnostics")
        st.write(f"- Concentration Flag: {'🔴 Yes' if diag.get('concentration_flag') else '🟢 No'}")
        st.write(f"- Overfit Risk Flag: {'🔴 Yes' if diag.get('overfit_flag') else '🟢 No'}")
        st.write(f"- Regime Adjustment Applied: {'Yes' if diag.get('regime_adjustment_applied') else 'No'}")
        st.write(f"- Regime Note: *{diag.get('regime_note', 'Normal')}*")
        if diag.get("stability_guardrail_triggered"):
            st.warning("⚠️ **Stability Guardrail Triggered:** Increased N above mathematical peak to ensure robustness.")
    
    with col_nr:
        st.markdown("#### Benefit vs N Curve")
        if not df_eval.empty and "n" in df_eval.columns:
            _fig_n = go.Figure()
            _fig_n.add_trace(go.Scatter(x=df_eval["n"], y=df_eval.get("PortfolioBenefit", []), mode="lines+markers", name="Net Benefit", line=dict(width=3)))
            _fig_n.add_trace(go.Scatter(x=df_eval["n"], y=df_eval.get("sharpe_normalized", []), mode="lines+markers", line=dict(dash='dot', width=1), name="Norm Sharpe"))
            _fig_n.add_trace(go.Scatter(x=df_eval["n"], y=df_eval.get("diversification_score", []), mode="lines+markers", line=dict(dash='dot', width=1), name="Diversification"))
            
            # Highlight N*
            _fig_n.add_vline(x=n_star, line_width=2, line_dash="dash", line_color="green", annotation_text=f"  N*={n_star}")
            _fig_n.update_layout(xaxis_title="Number of Stocks (N)", yaxis_title="Score / Benefit", height=300, margin=dict(l=0,r=0,t=30,b=0), hovermode="x unified")
            st.plotly_chart(_fig_n, use_container_width=True)
            
            st.caption("**Interpretation:** Selects maximum benefit. If curve is very steep, it's highly sensitive to concentration. If flat, shows stable diversification.")
        else:
            st.warning("No valid evaluation data to display.")

    # ── Model Error Calibration ──────────────────────────────────────────────
    st.divider()
    render_error_calibration_section()


def render_portfolio_comparison() -> None:
    st.subheader("Portfolio Comparison Lab")
    user_defs = st.session_state.get("user_portfolios", [])
    if not user_defs:
        st.info("Create up to 5 portfolios in the sidebar to begin comparison.")
        st.caption("Expand a portfolio in the sidebar, pick stocks and settings, then come back here and click 'Run comparison'.")
        return

    # Require an explicit run (or Auto-run) to avoid starting analysis before user is ready
    pc_run = st.session_state.get("_pc_run", False)
    pc_auto = st.session_state.get("_pc_auto", False)
    if not pc_run and not pc_auto:
        st.info("Ready to run comparison. Add stocks to portfolios and click 'Run comparison'.")
        c_run, c_auto = st.columns([2, 1])
        if c_run.button("Run comparison"):
            st.session_state["_pc_run"] = True
            pc_run = True
        # Auto-run checkbox: Streamlit manages the _pc_auto key — do NOT write to it manually
        pc_auto = c_auto.checkbox("Auto-run on changes", value=False, key="_pc_auto")
        if not pc_run and not pc_auto:
            return

    # Build dataclass list
    ups = []
    for p in user_defs:
        ups.append(
            pc.UserPortfolio(
                name=p.get("name", "Unnamed"),
                symbols=p.get("symbols", []),
                strategy=p.get("strategy", STRATEGY_EQUAL_WEIGHT),
                regime_mode=p.get("regime_mode", "static"),
                top_n=p.get("top_n", 5),
            )
        )

    # Inject Auto Hybrid if toggled on
    _hybrid_name = "Auto Diversified Hybrid"
    if st.session_state.get("include_hybrid_portfolio"):
        from services.auto_diversified_portfolio import STRATEGY_AUTO_HYBRID
        ups.append(pc.UserPortfolio(
            name=_hybrid_name,
            symbols=[],
            strategy=STRATEGY_AUTO_HYBRID,
            regime_mode="volatility_targeting",
            top_n=st.session_state.get("hybrid_top_n", 15),
        ))

    start = st.session_state.start_date
    end = st.session_state.end_date

    with st.spinner("Backtesting portfolios…"):
        t0 = time.time()
        results = pc.backtest_user_portfolios(ups, start, end, use_multiprocessing=st.session_state.use_multiprocessing)
        _log_execution_time("Backtest User Portfolios", t0)

    if not results:
        st.warning("No backtest results. Check your portfolio definitions and date range.")
        return

    # ── Composite Rating (new unified system) ────────────────────────────────
    from services.portfolio_rating import grade_colour as _gc
    full_r = pc.compute_full_ratings(results, meta_result=None)
    ratings   = full_r["ratings"]
    top_name  = full_r["top_name"]
    rated_df  = full_r["rated_df"]

    # Section 1 — Composite Rating Table (sorted by Composite Score)
    st.markdown("**Section 1 – Composite Rating Table**")

    # Trophy banner for the #1 portfolio
    if top_name:
        top_r = ratings.get(top_name, {})
        top_sc = top_r.get("composite_score", 0)
        top_gr = top_r.get("grade", "")
        badge_col = _gc(top_gr)
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#1e293b,#0f172a);border-radius:10px;"
            f"padding:14px 20px;margin-bottom:12px;border-left:5px solid {badge_col};'>"
            f"<span style='font-size:22px'>🏆</span>"
            f" <b style='color:{badge_col};font-size:18px'>{top_name}</b>"
            f"<span style='color:#94a3b8;font-size:14px'> ranked #1 with composite score "
            f"<b style='color:white'>{top_sc:.1f}/100 ({top_gr})</b></span></div>",
            unsafe_allow_html=True,
        )

    rows = []
    for name, v in results.items():
        m    = v.get("metrics", {})
        warn = ", ".join(v.get("warnings", [])) if v.get("warnings") else ""
        r    = ratings.get(name, {})
        sc   = r.get("composite_score", 0.0)
        gr   = r.get("grade", "—")
        rnk  = r.get("rank", "—")
        stab = r.get("stability_score", v.get("stability", {}).get("stability_score", 0.0))
        sgr  = r.get("stability_grade", "—")
        # find top_n from ups, properly handling 'auto' for hybrid portfolio
        _mapped_u = next((u for u in ups if u.name == name), None)
        _top_n_val = _mapped_u.top_n if _mapped_u else "N/A"
        
        # If it was returned as 'auto', but we have the actual N stored in metrics (via auto_diversified_portfolio)
        if _top_n_val == "auto" or (name == _hybrid_name and str(_top_n_val).lower() == "auto"):
            # Check the metrics dict where we tucked optimal_n from the backtest
            _metrics_n = v.get("metrics", {}).get("optimal_n")
            if _metrics_n is not None:
                _top_n_val = f"{_metrics_n} (Auto)"
        rows.append({
            "Rank 🏅":           rnk,
            "Portfolio":         name,
            "N Used":            str(_top_n_val),
            "CAGR":              f"{m.get('CAGR', 0):.2%}",
            "Sharpe":            f"{m.get('Sharpe', 0):.3f}",
            "Max DD":            f"{m.get('Max Drawdown', 0):.2%}",
            "Calmar":            f"{m.get('Calmar', 0):.3f}",
            "Stability 📊":     f"{stab:.1f} ({sgr})",
            "RRC 📈":            f"{r.get('components', {}).get('rrc_component', 50):.1f}/100",
            "Composite ⭐":     f"{sc:.1f}/100",
            "Grade":             gr,
            "Warnings":          warn,
        })

    df_comp = pd.DataFrame(rows).sort_values("Rank 🏅")
    st.dataframe(df_comp, use_container_width=True)

    # Component breakdown expander
    with st.expander("🔍 Composite Rating Component Breakdown", expanded=False):
        comp_rows = []
        for name, r in ratings.items():
            comps = r.get("components", {})
            comp_rows.append({
                "Portfolio":    name,
                "Grade":        r.get("grade", "—"),
                "Composite":    f"{r.get('composite_score', 0):.1f}",
                "Sharpe (30%)": f"{comps.get('sharpe_component', 0):.1f}",
                "Drawdown (20%)": f"{comps.get('drawdown_component', 0):.1f}",
                "Stability (15%)": f"{comps.get('stability_component', 0):.1f}",
                "Regime (10%)": f"{comps.get('regime_component', 0):.1f}",
                "Diversif (15%)": f"{comps.get('diversif_component', 0):.1f}",
                "RRC (10%)":    f"{comps.get('rrc_component', 0):.1f}",
            })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        # Radar chart for all portfolios
        categories = ["Sharpe", "Drawdown", "Stability", "Regime", "Diversif", "RRC"]
        fig_radar = go.Figure()
        for name, r in ratings.items():
            comps = r.get("components", {})
            vals = [
                comps.get("sharpe_component", 0),
                comps.get("drawdown_component", 0),
                comps.get("stability_component", 0),
                comps.get("regime_component", 0),
                comps.get("diversif_component", 0),
            ]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                opacity=0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=420,
            margin=dict(t=20, b=20),
            title="5-Component Rating Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Section 2 — Equity curve overlay
    st.markdown("**Section 2 – Equity Curve Overlay**")
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for name, v in results.items():
        eq = v.get("equity_curve")
        if eq is None or eq.empty:
            continue
        try:
            eq_dt = eq.copy()
            eq_dt["date"] = pd.to_datetime(eq_dt["date"])
            is_top = (name == top_name)
            fig.add_trace(go.Scatter(
                x=eq_dt["date"], y=eq_dt["cumulative_return"],
                name=name + (" 🏆" if is_top else ""),
                line=dict(width=3 if is_top else 1.5),
            ))
        except Exception:
            continue
    st.plotly_chart(fig, use_container_width=True)


    # Section 3 — Correlation heatmap
    st.markdown("**Section 3 – Correlation Heatmap**")
    corr, cov = pc.compute_portfolio_correlation(results)
    if corr is None:
        st.caption("Insufficient return series to compute correlations.")
    else:
        hm = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1))
        hm.update_layout(title="Daily return correlation")
        st.plotly_chart(hm, use_container_width=True)

    # Section 4 — Rating summary
    st.markdown("**Section 4 – Rating Summary**")
    # Show the columns that actually exist in the composite-rated df_comp
    _summary_cols = [c for c in ["Rank 🏅", "Portfolio", "Composite ⭐", "Grade", "Stability 📊", "Warnings"]
                     if c in df_comp.columns]
    st.dataframe(df_comp[_summary_cols], use_container_width=True)

    # Meta-portfolio construction
    st.markdown("**Meta Portfolio (Blended)**")
    with st.spinner("Constructing meta-portfolio…"):
        meta = pc.construct_meta_portfolio(results, portfolios=ups, start_date=start, end_date=end, user_ratings=ratings)

    weights = meta.get("weights", {})
    if weights:
        wdf = pd.DataFrame([{"Portfolio": k, "Weight": v} for k, v in weights.items()])
        st.table(wdf)
        mm = meta.get("meta_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meta CAGR", f"{mm.get('CAGR', 0):.2%}")
        c2.metric("Meta Sharpe", f"{mm.get('Sharpe', 0):.2f}")
        c3.metric("Meta Max Drawdown", f"{mm.get('Max Drawdown', 0):.2%}")
        c4.metric("Stop Risk Score", f"{meta.get('meta_stop_loss_score', 0):.1f}/100")

        # Meta equity curve
        meq = meta.get("meta_equity_curve")
        if meq is not None and not meq.empty:
            figm = go.Figure()
            try:
                meq_dt = meq.copy()
                meq_dt["date"] = pd.to_datetime(meq_dt["date"])
                figm.add_trace(go.Scatter(x=meq_dt["date"], y=meq_dt["cumulative_return"], name="Meta cumulative"))
                st.plotly_chart(figm, use_container_width=True)
            except Exception:
                st.caption("Could not render meta equity curve")

        # Winner info: compare meta vs best user-defined portfolio
        winner = meta.get("winner", {})
        meta_score = meta.get("meta_stock_rating") or meta.get("meta_portfolio_rating") or 0.0
        best_name = winner.get("best_user_name")
        best_score = winner.get("best_user_score", 0.0)
        if winner.get("winner_type") == "meta":
            try:
                delta = float(meta_score) - float(best_score)
            except Exception:
                delta = 0.0
            st.success(f"Blended (meta) portfolio outperforms best user portfolio '{best_name}' by {delta:.3f} rating points.")
        else:
            st.info(f"Best user-defined portfolio is '{best_name}' with rating {best_score:.3f}. The blended portfolio does not outperform it.")

        # If stock-level allocation is available, show it
        stock_weights = meta.get("stock_weights")
        if stock_weights:
            try:
                sw_df = pd.DataFrame([{"Symbol": k, "Weight": v} for k, v in stock_weights.items()])
                st.markdown("**Meta Stock-level Allocation**")
                st.table(sw_df.sort_values("Weight", ascending=False).head(50))
            except Exception:
                pass
    else:
        st.caption("Meta-portfolio could not be constructed (insufficient data).")

    # ── Section 6: Auto Hybrid Detail ───────────────────────────────────
    if st.session_state.get("include_hybrid_portfolio") and _hybrid_name in results:
        st.divider()
        st.markdown("**🤖 Auto Diversified Hybrid — Portfolio Detail**")
        from services.auto_diversified_portfolio import (
            build_diversified_hybrid_portfolio,
            get_monthly_rebalance_dates,
            rebalance_only_if_new_month,
            _load_rebalance_state,
        )
        from services.backtest import get_latest_score_date_on_or_before

        _h_top_n = st.session_state.get("hybrid_top_n", 15)
        _h_max_corr = st.session_state.get("hybrid_max_corr", 0.80)
        _h_min_score = st.session_state.get("hybrid_min_score", 0.0)

        # ── Scheduler status ────────────────────────────────────────────
        _rs = _load_rebalance_state()
        _last_reb = _rs.get("last_rebalance_date", "Never")
        _history: list = _rs.get("history", [])

        _sc1, _sc2 = st.columns([3, 1])
        _sc1.info(f"📅 Last rebalance: **{_last_reb}**  |  Runs automatically when month changes on page load.")

        # ── Rebalance status (no longer auto-triggers — use the buttons below) ─
        from datetime import date as _date_cls
        _today = _date_cls.today()
        _needs_rebalance = (
            _last_reb == "Never"
            or (_last_reb != "Never" and (
                _date_cls.fromisoformat(_last_reb).year != _today.year
                or _date_cls.fromisoformat(_last_reb).month != _today.month
            ))
        )
        if _needs_rebalance:
            st.warning(
                f"⚠️ A new month has started since the last rebalance ({_last_reb}). "
                "Click **🔄 Trigger Rebalance** below to run the monthly rebalance."
            )

        # Manual trigger buttons
        _bc1, _bc2, _ = st.columns([2, 2, 4])
        if _bc1.button("🔄 Trigger Rebalance", key="hybrid_manual_reb",
                       help="Rebalances if month has changed; no-op if already done this month"):
            with st.spinner("Running monthly rebalance check…"):
                _man_result = rebalance_only_if_new_month(
                    top_n=_h_top_n, max_corr=_h_max_corr, min_score=_h_min_score,
                    user_id=st.session_state.get("user_id"),
                )
            if _man_result["status"] == "rebalanced":
                st.success(f"✅ {_man_result['message']}")
                _last_reb = str(_man_result["rebalanced_on"])
            elif _man_result["status"] == "skipped":
                st.info(f"ℹ️ {_man_result['message']}")
            else:
                st.error(f"❌ {_man_result['message']}")
            _rs = _load_rebalance_state()
            _history = _rs.get("history", [])

        if _bc2.button("🔁 Force Rebalance", key="hybrid_force_reb",
                       help="Force a fresh rebalance even if one already ran this month"):
            with st.spinner("Forcing hybrid rebalance…"):
                _force_result = rebalance_only_if_new_month(
                    top_n=_h_top_n, max_corr=_h_max_corr, min_score=_h_min_score,
                    force=True,
                    user_id=st.session_state.get("user_id"),
                )
            if _force_result["status"] == "rebalanced":
                st.success(f"✅ {_force_result['message']}")
                _last_reb = str(_force_result["rebalanced_on"])
            else:
                st.error(f"❌ {_force_result['message']}")
            _rs = _load_rebalance_state()
            _history = _rs.get("history", [])

        # Rebalance history table
        if _history:
            with st.expander("📋 Rebalance history (last 36 months)", expanded=False):
                _hist_rows = [
                    {
                        "Date": h.get("date"), "Build date": h.get("build_date"),
                        "Stocks": h.get("n_stocks"), "Exp Vol": f"{h.get('expected_volatility', 0):.1%}",
                        "Exp Sharpe": f"{h.get('expected_sharpe', 0):.2f}",
                        "Forced": "✓" if h.get("forced") else "",
                        "Warnings": ", ".join(h.get("warnings", [])) or "—",
                    }
                    for h in reversed(_history)
                ]
                st.dataframe(pd.DataFrame(_hist_rows), use_container_width=True)

        _h_top_n = st.session_state.get("hybrid_top_n", 15)
        _h_max_corr = st.session_state.get("hybrid_max_corr", 0.80)
        _h_min_score = st.session_state.get("hybrid_min_score", 0.0)

        with st.spinner("Loading hybrid snapshot…"):
            try:
                _latest_score_d = get_latest_score_date_on_or_before(end)
                _snapshot = build_diversified_hybrid_portfolio(
                    as_of_date=_latest_score_d or end,
                    top_n=_h_top_n, max_corr=_h_max_corr, min_score=_h_min_score,
                )
            except Exception as _exc:
                _snapshot = None
                st.error(f"Hybrid snapshot failed: {_exc}")

        if _snapshot and _snapshot.get("selected_stocks"):
            _h_warn = _snapshot.get("warnings", [])
            if _h_warn:
                for _w in _h_warn:
                    st.warning(f"🚨 {_w}")

            _hc1, _hc2, _hc3 = st.columns(3)
            _hc1.metric("Selected stocks", len(_snapshot["selected_stocks"]))
            _hc2.metric("Expected Vol (ann.)", f"{_snapshot.get('expected_volatility', 0):.1%}")
            _hc3.metric("Expected Sharpe (est.)", f"{_snapshot.get('expected_sharpe', 0):.2f}")
            st.caption(f"Avg pairwise correlation: **{_snapshot.get('avg_pairwise_corr', 0):.2f}**  —  Snapshot date: `{_snapshot.get('rebalance_date')}`")

            # Selected stocks table
            with st.expander("📊 Selected stocks & weights", expanded=True):
                _sw = _snapshot.get("weights", {})
                _syms = {int(sid): sym for sid, sym in _snapshot["selected_stocks"]}
                _rows = [
                    {"Symbol": _syms.get(int(sid), str(sid)), "Weight": f"{w:.3%}"}
                    for sid, w in sorted(_sw.items(), key=lambda x: x[1], reverse=True)
                ]
                st.dataframe(pd.DataFrame(_rows), use_container_width=True)

            # Weight distribution bar chart
            with st.expander("📊 Weight distribution", expanded=False):
                _bar_syms = [_syms.get(int(sid), str(sid)) for sid in sorted(_sw, key=lambda x: _sw[x], reverse=True)]
                _bar_vals = [_sw[sid] for sid in sorted(_sw, key=lambda x: _sw[x], reverse=True)]
                _fig_bar = go.Figure(go.Bar(x=_bar_syms, y=_bar_vals, marker_color="#4CAF50"))
                _fig_bar.update_layout(yaxis_tickformat=".1%", xaxis_title="Symbol", yaxis_title="Weight", height=320, margin=dict(t=20, b=10))
                st.plotly_chart(_fig_bar, use_container_width=True)

            # Correlation heatmap of selected stocks
            with st.expander("🔥 Correlation heatmap (selected stocks)", expanded=False):
                _h_eq = results[_hybrid_name].get("equity_curve")
                if _h_eq is not None and not _h_eq.empty:
                    # Rebuild return matrix from equity curve for the selected stocks
                    _h_sid_list = [sid for sid, _ in _snapshot["selected_stocks"]]
                    from services.auto_diversified_portfolio import _load_returns_up_to
                    _h_ret = _load_returns_up_to(_h_sid_list, _latest_score_d or end)
                    if not _h_ret.empty and len(_h_ret.columns) >= 2:
                        _h_corr = _h_ret.rename(columns={int(sid): sym for sid, sym in _snapshot["selected_stocks"]}).corr()
                        _fig_hm = go.Figure(go.Heatmap(
                            z=_h_corr.values,
                            x=list(_h_corr.columns),
                            y=list(_h_corr.index),
                            colorscale="RdBu", zmin=-1, zmax=1,
                        ))
                        _fig_hm.update_layout(title="Pairwise return correlation", height=420, margin=dict(t=40, b=10))
                        st.plotly_chart(_fig_hm, use_container_width=True)
                    else:
                        st.caption("Insufficient return data for correlation heatmap.")
                else:
                    st.caption("No equity data available.")

            # Stability breakdown for the hybrid portfolio
            _h_stab = results[_hybrid_name].get("stability", {})
            if _h_stab and _h_stab.get("stability_score", 0) > 0:
                with st.expander("📈 Stability Score Breakdown", expanded=True):
                    from services.stability_analyzer import grade_to_colour
                    _ss = _h_stab["stability_score"]
                    _sg = _h_stab.get("grade", "N/A")
                    _suf = _h_stab.get("data_sufficiency", "")

                    _stab_c1, _stab_c2, _stab_c3 = st.columns(3)
                    _stab_c1.metric("Stability Score", f"{_ss:.1f}/100")
                    _stab_c2.metric("Stability Grade", _sg)
                    _stab_c3.metric("Data", _suf.capitalize() if _suf else "—")

                    # Component bars
                    st.markdown("**Component breakdown**")
                    for _comp_name, _comp_val in _h_stab.get("components", {}).items():
                        _bar_col = "#22c55e" if _comp_val >= 70 else "#f59e0b" if _comp_val >= 40 else "#ef4444"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
                            f"<span style='width:230px;font-size:13px'>{_comp_name}</span>"
                            f"<div style='flex:1;background:#e5e7eb;border-radius:4px;height:14px'>"
                            f"<div style='width:{_comp_val:.1f}%;background:{_bar_col};border-radius:4px;height:14px'></div></div>"
                            f"<span style='width:45px;text-align:right;font-size:13px'><b>{_comp_val:.1f}</b></span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Rolling Sharpe chart
                    _rm = _h_stab.get("rolling_metrics", pd.DataFrame())
                    if not _rm.empty and "rolling_sharpe" in _rm.columns:
                        _rs_data = _rm["rolling_sharpe"].dropna()
                        if len(_rs_data) > 0:
                            st.markdown("**Rolling 12-month Sharpe**")
                            _fig_rs = go.Figure()
                            _fig_rs.add_trace(go.Scatter(
                                x=_rs_data.index, y=_rs_data.values,
                                mode="lines", name="Rolling Sharpe",
                                line=dict(color="#6366f1", width=2),
                                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                            ))
                            _fig_rs.add_hline(y=0, line_dash="dot", line_color="gray")
                            _fig_rs.update_layout(height=240, margin=dict(t=10, b=10), showlegend=False)
                            st.plotly_chart(_fig_rs, use_container_width=True)

                    # Regime comparison
                    _summ = _h_stab.get("summary", {})
                    if _summ.get("regime_sharpe_high") is not None:
                        st.markdown("**Regime Sharpe comparison**")
                        _rg_c1, _rg_c2, _rg_c3 = st.columns(3)
                        _rg_c1.metric("High-vol Sharpe", f"{_summ.get('regime_sharpe_high', 0):.3f}")
                        _rg_c2.metric("Low-vol Sharpe", f"{_summ.get('regime_sharpe_low', 0):.3f}")
                        _rg_c3.metric("Gap", f"{_summ.get('regime_sharpe_gap', 0):.3f}",
                                      delta_color="inverse")
        else:
            st.info("Hybrid portfolio could not be built — check that scores and price data exist for the selected date range.")

    # ── Sector Analysis ──────────────────────────────────────────────────────
    with st.expander("📊 Sector Analysis", expanded=False):
        try:
            sector_map = sa.get_sector_map()
            if not sector_map:
                st.warning("No sector data available. Run `populate_sectors()` from `data_fetcher.py` first.")
            else:
                # Sector relative performance table
                st.markdown("### Sector Relative Performance")
                _sd_range = _get_score_date_range()
                if _sd_range and _sd_range[0] and _sd_range[1]:
                    _scoring_dt = _sd_range[1]
                    _sector_perf = sa.compute_sector_relative_performance(
                        start_date=_sd_range[0],
                        end_date=_sd_range[1],
                        scoring_date=_scoring_dt,
                    )
                    if not _sector_perf.empty:
                        _disp = _sector_perf.copy()
                        _disp["CAGR"] = _disp["cagr"].apply(lambda x: f"{x * 100:.1f}%")
                        _disp["Sharpe"] = _disp["sharpe"].apply(lambda x: f"{x:.2f}")
                        _disp["Max DD"] = _disp["max_drawdown"].apply(lambda x: f"{x * 100:.1f}%")
                        _disp["Vol"] = _disp["volatility"].apply(lambda x: f"{x * 100:.1f}%")
                        _disp["Avg Score"] = _disp["avg_score"].apply(lambda x: f"{x:.3f}")
                        _disp = _disp.rename(columns={"sector": "Sector", "n_stocks": "# Stocks"})
                        st.dataframe(
                            _disp[["Sector", "# Stocks", "CAGR", "Sharpe", "Max DD", "Vol", "Avg Score"]],
                            use_container_width=True, hide_index=True,
                        )

                        # Bar chart of CAGR by sector
                        _fig_sec = go.Figure()
                        _colors = ["#22c55e" if v >= 0 else "#ef4444" for v in _sector_perf["cagr"]]
                        _fig_sec.add_trace(go.Bar(
                            x=_sector_perf["sector"],
                            y=_sector_perf["cagr"] * 100,
                            marker_color=_colors,
                            text=[f"{v:.1f}%" for v in _sector_perf["cagr"] * 100],
                            textposition="outside",
                        ))
                        _fig_sec.update_layout(
                            title="Sector CAGR (%)",
                            yaxis_title="CAGR %",
                            height=350, margin=dict(t=40, b=40),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_sec, use_container_width=True)
                    else:
                        st.caption("Not enough price data for sector performance.")
                else:
                    st.caption("Score date range unavailable.")

                # Portfolio sector exposure (for hybrid portfolio if available)
                st.markdown("### Portfolio Sector Exposure")
                _hybrid_weights = None
                if "_latest_hybrid_result" in st.session_state:
                    _hybrid_weights = st.session_state["_latest_hybrid_result"].get("weights", {})
                elif "_pc_results" in st.session_state:
                    for _pname, _pdata in st.session_state["_pc_results"].items():
                        if isinstance(_pdata, dict) and "weights" in _pdata:
                            _hybrid_weights = _pdata["weights"]
                            break

                if _hybrid_weights:
                    _exp_df = sa.compute_portfolio_sector_exposure(_hybrid_weights)
                    if not _exp_df.empty:
                        _fig_exp = go.Figure()
                        _fig_exp.add_trace(go.Bar(
                            y=_exp_df["sector"],
                            x=_exp_df["pct"],
                            orientation="h",
                            marker_color="#6366f1",
                            text=[f"{p:.1f}%" for p in _exp_df["pct"]],
                            textposition="outside",
                        ))
                        _fig_exp.update_layout(
                            title="Hybrid Portfolio — Sector Allocation",
                            xaxis_title="Allocation %",
                            height=max(250, len(_exp_df) * 35 + 80),
                            margin=dict(l=150, t=40, b=30),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_exp, use_container_width=True)
                    else:
                        st.caption("No sector data for portfolio stocks.")
                else:
                    st.caption("Run the Hybrid portfolio to see sector exposure.")
        except Exception as _sec_err:
            st.error(f"Sector analysis error: {_sec_err}")



@st.cache_resource
def _startup_auto_calibrate() -> None:
    """Intentionally a no-op — calibration is now user-triggered via the
    '🔬 Model Error Calibration' expander to avoid blocking startup."""
    pass


def main() -> None:
    # ── Page Config MUST be first ──
    st.set_page_config(page_title="AI Stock Engine", layout="wide")

    # ── Auth gate ──
    # Check if we are running the public Web Analyzer or Local Analyzer
    if settings.ENVIRONMENT == "web":
        try:
            auth_sec = st.secrets.get("auth")
            _has_auth = auth_sec is not None and "cookie_secret" in auth_sec
        except Exception:
            _has_auth = False

        if _has_auth:
            user_info = require_login()
            st.session_state["user_id"] = user_info.id
            st.session_state["user_email"] = user_info.email
            st.session_state["user_name"] = user_info.name
        else:
            st.error("Authentication missing for Web Mode.")
            st.stop()
    else:
        # Dev / Local mode — bypass auth, use default admin user_id=1
        if "user_id" not in st.session_state or st.session_state["user_id"] is None:
            st.session_state["user_id"] = 1
            st.session_state["user_email"] = "admin@localhost"
            st.session_state["user_name"] = "Local Admin"
        
        st.sidebar.success("💻 Running Local Analyzer (Full Dataset)")

    _startup_auto_calibrate()
    _init_session_state()
    _sidebar()

    # ── User info in sidebar ──
    if st.session_state.get("user_email"):
        with st.sidebar:
            st.divider()
            st.markdown(f"👤 **{st.session_state.get('user_name', '')}**")
            st.caption(st.session_state["user_email"])
            if st.button("🚪 Logout", key="logout_btn"):
                st.logout()

    st.title("AI Stock Engine – Research Interface (V2)")
    st.caption("Portfolio | Stock recommendation | Research validation | Assistant Enabled")

    if st.session_state.mode == "Home":
        render_home_mode()
    elif st.session_state.mode == "Portfolio Mode":
        render_portfolio_mode()
    elif st.session_state.mode == "Stock Mode":
        render_stock_mode()
    elif st.session_state.mode == "Research Mode":
        render_research_mode()
    elif st.session_state.mode == "Portfolio Comparison":
        render_portfolio_comparison()
    elif st.session_state.mode == "Assistant":
        render_assistant_mode()
    else:
        # Fallback for any unhandled mode, or initial state before selection
        render_home_mode()

def render_assistant_mode() -> None:
    st.header("🤖 AI Assistant (V3 - Live Data Patch)")
    st.markdown("Ask natural language questions about the engine, your portfolios, or stock recommendations.")

    # Status/Diagnostic for LLM
    from app.assistant.llm_engine import OLLAMA_URL, MODEL_NAME
    with st.expander("🌐 Engine Connectivity Status"):
        st.code(f"Ollama URL: {OLLAMA_URL}")
        import urllib.request
        import json
        try:
            # 1. Check reachability
            tags_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
            with urllib.request.urlopen(tags_url, timeout=2) as r:
                data = json.loads(r.read().decode())
                st.success("✅ Ollama is Reachable")
                
                # 2. Check model existence
                models = [m.get("name") for m in data.get("models", [])]
                if any(MODEL_NAME in m for m in models):
                    st.success(f"🧠 Model '{MODEL_NAME}' is Ready")
                else:
                    st.warning(f"⏳ Model '{MODEL_NAME}' is NOT found. It will be pulled on the first query (takes ~1-2 mins).")
        except Exception as e:
            st.error(f"❌ Ollama is Unreachable: {e}")

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []

    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "details" in msg and msg["details"]:
                with st.expander("🛠️ Diagnostics & Factors Used"):
                    st.json(msg["details"])

    if prompt := st.chat_input("Ask me about TCS, hybrid portfolios, or overfitting..."):
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing engine data..."):
                from app.assistant.schemas import AssistantRequest
                from app.assistant.assistant import process_query
                
                req = AssistantRequest(user_query=prompt)
                res = process_query(req)
                
                symbol_debug = res.raw_outputs.get("get_stock_summary", {}).get("symbol", "N/A")
                if symbol_debug == "N/A":
                    symbol_debug = res.raw_outputs.get("get_stop_loss", {}).get("symbol", "N/A")
                
                response_text = f"**Intent:** {res.intent} | **Symbol Detected:** {symbol_debug}\n\n**Analysis:**\n{res.explanation}"
                st.markdown(response_text)
                
                # Check for Chart Data (Price Movement)
                pm_data = res.raw_outputs.get("get_price_movement", {})
                if pm_data and "history" in pm_data:
                    df_chart = pd.DataFrame(pm_data["history"])
                    df_chart["date"] = pd.to_datetime(df_chart["date"])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_chart["date"], 
                        y=df_chart["close"],
                        mode='lines',
                        name=pm_data.get('symbol', 'Price'),
                        line=dict(color='#00d4ff', width=2)
                    ))
                    fig.update_layout(
                        title=f"Price Movement: {pm_data.get('symbol')}",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                details = {
                    "functions_called": res.functions_called,
                    "raw_outputs": res.raw_outputs
                }
                with st.expander("🛠️ Diagnostics & Factors Used"):
                    st.json(details)
                
        st.session_state.assistant_messages.append({
            "role": "assistant", 
            "content": response_text, 
            "details": details
        })


if __name__ == "__main__":
    main()
