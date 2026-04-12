from typing import Dict, Any

def get_stock_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    from services.recommendation import get_stock_signal, generate_recommendation
    try:
        symbol = params.get("symbol")
        signal = get_stock_signal(symbol)
        if not signal:
            return {"error": f"Stock '{symbol}' not found or data missing."}
        
        # Unify scaling to 0-100
        raw_score = signal.get("composite_score", 0.0)
        scaled_score = round(raw_score * 100, 2)
        
        # Factor & Fundamental Fallback
        from app.database import get_db_context
        from app.models import Stock, Factor, Fundamental
        from sqlalchemy import select, desc
        from services.portfolio import get_latest_scoring_date
        from datetime import date as _date
        
        factors = {}
        fundamentals = {}
        with get_db_context() as db:
            try:
                latest_date = get_latest_scoring_date()
            except Exception:
                latest_date = _date.today()
            stock = db.execute(select(Stock).where(Stock.symbol == symbol)).scalars().first()
            if stock:
                # 1. Factors
                f = db.execute(select(Factor).where(Factor.stock_id == stock.id, Factor.date == latest_date)).scalars().first()
                if not f or (f.quality_score == 0 and f.value_score == 0 and f.growth_score == 0):
                    f = db.execute(select(Factor).where(Factor.stock_id == stock.id, Factor.quality_score > 0).order_by(Factor.date.desc())).scalars().first()
                if f:
                    factors = {
                        "quality": round(f.quality_score * 100, 2),
                        "value": round(f.value_score * 100, 2),
                        "growth": round(f.growth_score * 100, 2)
                    }
                
                # 2. Fundamentals
                fund = db.execute(select(Fundamental).where(Fundamental.stock_id == stock.id).order_by(desc(Fundamental.quarter))).scalars().first()
                if fund:
                    fundamentals = {
                        "eps": fund.eps,
                        "roe": round(fund.roe * 100, 2) if fund.roe else None,
                        "debt_equity": fund.debt_equity,
                        "period": fund.quarter,
                        "source": "database"
                    }
                else:
                    # Fallback to Live Fetch
                    from services.financial_utils import fetch_live_fundamentals
                    try:
                        fundamentals = fetch_live_fundamentals(symbol)
                    except Exception:
                        fundamentals = {"source": "unavailable"}

        rec = generate_recommendation(signal, risk_profile="moderate")
        return {
            "symbol": signal["symbol"],
            "composite_score": scaled_score,
            "rank": signal["rank"],
            "momentum": round(signal.get("momentum_score", 0.0) * 100, 2),
            "volatility": round(signal.get("volatility_score", 0.0) * 100, 2),
            "factors": factors,
            "fundamentals": fundamentals,
            "recommendation": rec["recommendation"],
            "suggested_allocation": rec["suggested_allocation_range"],
            "current_regime": signal["regime"]
        }
    except Exception as e:
        return {"error": str(e)}

def get_stock_research(params: Dict[str, Any]) -> Dict[str, Any]:
    from app.database import get_db_context
    from services.recommendation import get_stock_signal
    from app.models import Factor
    from sqlalchemy import select
    from datetime import date as _date
    try:
        symbol = params.get("symbol")
        signal = get_stock_signal(symbol)
        if not signal:
            return {"error": f"Stock '{symbol}' not found."}
            
        # Get factor & fundamental details
        with get_db_context() as db:
            from services.portfolio import get_latest_scoring_date
            try:
                latest_date = get_latest_scoring_date()
            except Exception:
                latest_date = _date.today()
            from app.models import Stock, Factor, Fundamental
            from sqlalchemy import desc
            stock = db.execute(select(Stock).where(Stock.symbol == symbol)).scalars().first()
            factors = {}
            fundamentals = {}
            data_date = str(latest_date)
            
            if stock:
                # 1. Try latest factors
                f = db.execute(select(Factor).where(Factor.stock_id == stock.id, Factor.date == latest_date)).scalars().first()
                if not f or (f.quality_score == 0 and f.value_score == 0 and f.growth_score == 0):
                    f = db.execute(select(Factor).where(Factor.stock_id == stock.id, Factor.quality_score > 0).order_by(Factor.date.desc())).scalars().first()
                    if f: data_date = str(f.date)
                
                if f:
                    factors = {
                        "quality": round(f.quality_score * 100, 2), 
                        "value": round(f.value_score * 100, 2), 
                        "growth": round(f.growth_score * 100, 2)
                    }
                
                # 2. Get latest fundamentals
                fund = db.execute(select(Fundamental).where(Fundamental.stock_id == stock.id).order_by(desc(Fundamental.quarter))).scalars().first()
                if fund:
                    fundamentals = {
                        "eps": fund.eps,
                        "roe": round(fund.roe * 100, 2) if fund.roe else None,
                        "debt_equity": fund.debt_equity,
                        "period": fund.quarter,
                        "source": "database"
                    }
                else:
                    # Fallback to Live Fetch
                    from services.financial_utils import fetch_live_fundamentals
                    try:
                        fundamentals = fetch_live_fundamentals(symbol)
                    except Exception:
                        fundamentals = {"source": "unavailable"}

        return {
            "symbol": symbol,
            "composite_score": round(signal.get("composite_score", 0.0) * 100, 2),
            "factors": factors,
            "fundamentals": fundamentals,
            "trend": signal.get("trend_direction", "Stable"),
            "analysis_date": data_date
        }
    except Exception as e:
        return {"error": str(e)}

def get_stop_loss(params: Dict[str, Any]) -> Dict[str, Any]:
    from services.stop_loss_engine import analyze_stock_stop_loss
    from services.portfolio import get_latest_scoring_date
    try:
        symbol = params.get("symbol")
        latest_date = get_latest_scoring_date()
        # Lookup stock_id
        from app.database import get_db_context
        from app.models import Stock
        with get_db_context() as db:
            stock = db.execute(select(Stock).where(Stock.symbol == symbol)).scalars().first()
            if not stock:
                return {"error": f"Stock {symbol} not found."}
            
            # This service function returns the required dict
            profile = analyze_stock_stop_loss(latest_date, stock.id)
            return profile
    except Exception as e:
        return {"error": str(e)}

def get_hybrid_portfolio(params: Dict[str, Any] = None) -> Dict[str, Any]:
    params = params or {}
    from services.auto_diversified_portfolio import construct_auto_hybrid_portfolio
    from services.portfolio import get_latest_scoring_date
    try:
        latest = get_latest_scoring_date()
        risk = params.get("risk_profile", "moderate")
        top_n = params.get("top_n", "auto")
        
        # Call the core service logic
        result = construct_auto_hybrid_portfolio(
            date=latest,
            risk_profile=risk,
            top_n=top_n
        )
        # Simplify for LLM
        return {
            "date": str(latest),
            "stock_count": len(result.get("symbols", [])),
            "expected_sharpe": result.get("metrics", {}).get("sharpe", 0),
            "symbols": result.get("symbols", [])[:10], # Top 10 for context
            "weights": result.get("weights", {})
        }
    except Exception as e:
        return {"error": str(e)}

def get_meta_portfolio(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from services.portfolio_comparison import get_system_meta_portfolio
    try:
        # Get the standard system meta portfolio
        result = get_system_meta_portfolio()
        return result
    except Exception as e:
        return {"error": str(e)}

def compare_portfolios(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    payload = payload or {}
    from services.portfolio_comparison import backtest_user_portfolios, UserPortfolio, compute_full_ratings
    from services.user_portfolios import load_user_portfolios
    import datetime
    try:
        # Try load for user 1 first, then fallback to global (user_id=None) if empty
        raw_portfolios = load_user_portfolios(user_id=1)
        if not raw_portfolios:
            raw_portfolios = load_user_portfolios(user_id=None)
            
        if not raw_portfolios:
            return {
                "error": "No saved portfolios found to compare.",
                "context": "You haven't saved any portfolios yet. Please use the 'Portfolio Comparison' tab to create and save at least two portfolios first."
            }
            
        # Convert to dataclasses expected by backtester
        ups = []
        for p in raw_portfolios:
            ups.append(UserPortfolio(
                name=p["name"],
                symbols=p["symbols"],
                strategy=p["strategy"],
                regime_mode=p.get("regime_mode", "volatility_targeting"),
                top_n=p["top_n"]
            ))

        start = payload.get("start_date", "2024-01-01")
        end = payload.get("end_date", datetime.date.today().isoformat())
        s_date = datetime.date.fromisoformat(start)
        e_date = datetime.date.fromisoformat(end)
        
        # Run comparison logic
        results = backtest_user_portfolios(ups, s_date, e_date, use_multiprocessing=False)
        full_r = compute_full_ratings(results, meta_result=None)
        ratings = full_r["ratings"]
        
        # Summarize for LLM to avoid context overflow
        summary = {}
        for name, v in results.items():
            m = v.get("metrics", {})
            r = ratings.get(name, {})
            summary[name] = {
                "cagr": round(m.get("CAGR", 0.0), 4),
                "sharpe": round(m.get("Sharpe", 0.0), 3),
                "max_dd": round(m.get("Max Drawdown", 0.0), 4),
                "composite_score": round(r.get("composite_score", 0.0), 1),
                "grade": r.get("grade", "N/A")
            }
            
        return {
            "comparison_summary": summary,
            "portfolio_count": len(ups),
            "period": f"{start} to {end}",
            "top_performer": full_r.get("top_name")
        }
    except Exception as e:
        return {"error": str(e)}

def get_governance(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from services.model_governance import run_overfitting_diagnostics
    try:
        # We pass a simple mock since this is for general status
        mock_results = {"metrics": {"Sharpe": 1.2, "monthly_turnover": 0.15}}
        status = run_overfitting_diagnostics(mock_results)
        return status
    except Exception as e:
        return {"error": str(e)}

def get_auto_n_analysis(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from services.research_validation import run_walk_forward
    from services.portfolio import get_latest_scoring_date
    try:
        latest = get_latest_scoring_date()
        # Direct service call
        result = run_walk_forward(strategy="equal_weight", top_n_list=[5, 10, 15, 20])
        return {"auto_n_results": result}
    except Exception as e:
        return {"error": str(e)}
        
def get_stability_analysis(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from services.research_validation import run_weight_sensitivity
    try:
        result = run_weight_sensitivity(strategy="equal_weight", top_n=10)
        return {"stability_metrics": result}
    except Exception as e:
        return {"error": str(e)}

def list_stocks(params: Dict[str, Any] = None) -> Dict[str, Any]:
    params = params or {}
    from services.portfolio import load_ranked_stocks, get_latest_scoring_date
    try:
        top_n = int(params.get("top_n", 10))
        sector = params.get("sector")
        latest = get_latest_scoring_date()
        df = load_ranked_stocks(latest, sector=sector)
        df_top = df.head(top_n)
        
        stocks = []
        for _, row in df_top.iterrows():
            stocks.append({
                "symbol": row["symbol"],
                "composite_score": round(row["composite_score"] * 100, 2),
                "volatility_score": round(float(row["volatility_score"]) * 100, 2) if "volatility_score" in row and row["volatility_score"] is not None else "N/A",
                "rank": row["rank"]
            })
        date_str = latest.isoformat() if latest else "N/A"
        return {"stocks": stocks, "count": len(stocks), "date": date_str}
    except Exception as e:
        return {"error": str(e)}


def get_price_movement(params: Dict[str, Any]) -> Dict[str, Any]:
    from services.data_fetcher import fetch_price_data
    from datetime import date, timedelta
    try:
        symbol = params.get("symbol")
        days = int(params.get("days", 365))
        start_date = date.today() - timedelta(days=days)
        df = fetch_price_data(symbol, start_date=start_date)
        if df.empty:
            return {"error": f"No price data found for {symbol}"}
        
        # We return the data for the chart, but also summary stats for the LLM
        history = [{"date": str(d), "close": float(c)} for d, c in zip(df["date"], df["close"])]
        
        start_price = float(df["close"].iloc[0])
        end_price = float(df["close"].iloc[-1])
        pct_change = ((end_price / start_price) - 1.0) * 100
        
        return {
            "symbol": symbol,
            "period_days": days,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "pct_change": round(pct_change, 2),
            "high": round(float(df["close"].max()), 2),
            "low": round(float(df["close"].min()), 2),
            "history": history # Full data for plotting
        }
    except Exception as e:
        return {"error": str(e)}

def get_portfolio_review(params: Dict[str, Any]) -> Dict[str, Any]:
    from services.user_portfolios import load_user_portfolios
    from services.recommendation import get_stock_signal
    try:
        query_name = params.get("portfolio_name", "").strip().lower()
        # Fallback search: user 1 -> global
        portfolios = load_user_portfolios(user_id=1)
        if not portfolios:
            portfolios = load_user_portfolios(user_id=None)
        
        # 1. Try exact match
        target = None
        for p in portfolios:
            if p["name"].lower() == query_name:
                target = p
                break
        
        # 2. Try partial match if no exact match (e.g. "Meta" matches "Meta Portfolio")
        if not target and query_name:
            for p in portfolios:
                p_name_low = p["name"].lower()
                if query_name in p_name_low or p_name_low in query_name:
                    target = p
                    break
        
        if not target:
            available = [p["name"] for p in portfolios]
            return {
                "error": f"Portfolio '{params.get('portfolio_name')}' not found.",
                "context": "Available portfolios in the database are listed below. Please ask for one of these specifically.",
                "available_portfolios": available
            }
        
        results = []
        for symbol in target["symbols"]:
            signal = get_stock_signal(symbol)
            if signal:
                results.append({
                    "symbol": symbol,
                    "score": round(signal.get("composite_score", 0) * 100, 2),
                    "recommendation": signal.get("recommendation", "Hold"),
                    "rank": signal.get("rank", "N/A"),
                    "risk": signal.get("risk_level", "Moderate")
                })
        
        return {
            "portfolio_name": target["name"],
            "strategy": target["strategy"],
            "stock_count": len(target["symbols"]),
            "stock_reviews": results
        }
    except Exception as e:
        return {"error": str(e)}

def get_multi_stock_review(params: Dict[str, Any]) -> Dict[str, Any]:
    from services.recommendation import get_stock_signal
    try:
        symbols = params.get("symbols", [])
        if not symbols:
            return {"error": "No symbols provided for review."}
            
        results = []
        for symbol in symbols:
            signal = get_stock_signal(symbol)
            if signal:
                results.append({
                    "symbol": symbol,
                    "score": round(signal.get("composite_score", 0) * 100, 2),
                    "recommendation": signal.get("recommendation", "Hold"),
                    "rank": signal.get("rank", "N/A"),
                    "risk": signal.get("risk_level", "Moderate")
                })
        
        return {
            "title": "Custom Stock Review",
            "stock_reviews": results
        }
    except Exception as e:
        return {"error": str(e)}

def get_sector_rankings(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from services.sector_analytics import compute_sector_relative_performance
    from services.portfolio import get_latest_scoring_date, load_ranked_stocks
    from datetime import date, timedelta
    import pandas as pd
    try:
        top_n = int(params.get("top_n", 10)) if params else 10
        stocks_per_sector = int(params.get("stocks_per_sector", 0)) if params else 0
        latest = get_latest_scoring_date()
        if not latest:
            latest = date.today()
            
        # Default lookback of 180 days for sector performance
        start_date = latest - timedelta(days=180)
        
        df = compute_sector_relative_performance(
            start_date=start_date,
            end_date=latest,
            scoring_date=latest
        )
        
        if df.empty:
            return {"error": "No sector data available for the requested period."}
            
        # If requested, fetch top stocks per sector
        ranked_df = pd.DataFrame()
        if stocks_per_sector > 0:
            ranked_df = load_ranked_stocks(latest)
            
        # Filter to top_n and convert to list of dicts
        df_top = df.head(top_n)
        rankings = []
        for i, row in df_top.iterrows():
            sector_name = row["sector"]
            sector_info = {
                "rank": i + 1,
                "sector": sector_name,
                "n_stocks": int(row["n_stocks"]),
                "cagr": round(float(row["cagr"]) * 100, 2),
                "sharpe": round(float(row["sharpe"]), 2),
                "max_drawdown": round(float(row["max_drawdown"]) * 100, 2),
                "avg_score": round(float(row["avg_score"]) * 100, 2)
            }
            
            if stocks_per_sector > 0 and not ranked_df.empty:
                s_df = ranked_df[ranked_df["sector"] == sector_name].head(stocks_per_sector)
                top_stocks = []
                for _, s_row in s_df.iterrows():
                    top_stocks.append({
                        "symbol": s_row["symbol"],
                        "rank": s_row["rank"],
                        "score": round(s_row["composite_score"] * 100, 2)
                    })
                sector_info["top_stocks"] = top_stocks
                
            rankings.append(sector_info)
            
        return {
            "rankings": rankings,
            "period": f"{start_date} to {latest}",
            "count": len(rankings)
        }
    except Exception as e:
        return {"error": str(e)}

def get_latest_calibration(params: Dict[str, Any] = None) -> Dict[str, Any]:
    from app.api.routers.calibration import get_latest_calibration as get_cal
    try:
        resp = get_cal()
        return resp.model_dump()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

REGISTRY = {
    "get_stock_summary": get_stock_summary,
    "get_stock_research": get_stock_research,
    "get_stop_loss": get_stop_loss,
    "get_hybrid_portfolio": get_hybrid_portfolio,
    "get_meta_portfolio": get_meta_portfolio,
    "compare_portfolios": compare_portfolios,
    "get_governance": get_governance,
    "get_auto_n_analysis": get_auto_n_analysis,
    "get_stability_analysis": get_stability_analysis,
    "get_latest_calibration": get_latest_calibration,
    "list_stocks": list_stocks,
    "get_price_movement": get_price_movement,
    "get_portfolio_review": get_portfolio_review,
    "get_multi_stock_review": get_multi_stock_review,
    "get_sector_rankings": get_sector_rankings
}
