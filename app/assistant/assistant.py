import json
from typing import Dict, Any

from app.assistant.intent_router import IntentRouter
from app.assistant.function_registry import REGISTRY
from app.assistant.schemas import AssistantRequest, AssistantResponse
from app.assistant.prompts import (
    PORTFOLIO_EXPLANATION_TEMPLATE,
    GENERIC_EXPLANATION_TEMPLATE
)
from app.assistant.response_generator import (
    generate_stock_explanation,
    generate_portfolio_comparison,
    generate_risk_explanation,
     generate_auto_n_explanation,
    generate_governance_explanation,
    generate_market_overview_explanation,
    generate_portfolio_review_explanation,
    generate_sector_ranking_explanation
)
from app.assistant.llm_engine import llm_generate_explanation

def format_explanation(intent: str, outputs: Dict[str, Any], user_query: str = "") -> str:
    """
    Generate an explanation from the structured outputs.
    Fast-path Python generators are used for structured intents.
    LLM explanation is only used for analytical/narrative intents.
    """
    try:
        # --- FAST-PATH: structured intents with dedicated Python generators ---
        # These return immediately without any LLM call, so there is no blocking.

        if intent == "Market Overview":
            return generate_market_overview_explanation(outputs.get("list_stocks", {}))

        if intent == "Sector Ranking":
            return generate_sector_ranking_explanation(outputs.get("get_sector_rankings", {}))

        if intent == "Portfolio Review":
            return generate_portfolio_review_explanation(outputs.get("get_portfolio_review", {}))

        if intent == "Multi-Stock Review":
            return generate_portfolio_review_explanation(outputs.get("get_multi_stock_review", {}))

        if intent in ["Governance Check", "Calibration Explanation", "Stability Diagnostic"]:
            gov_data = outputs.get("get_governance", {})
            return generate_governance_explanation(gov_data)

        if intent == "Risk Diagnostics":
            stop_data = outputs.get("get_stop_loss", {})
            symbol = stop_data.get("symbol", "Target Stock")
            return generate_risk_explanation(stop_data, symbol)

        if intent == "Automatic N Explanation":
            auto_data = outputs.get("get_auto_n_analysis", {})
            return generate_auto_n_explanation(auto_data)

        if intent == "Price Movement Analysis":
            data = outputs.get("get_price_movement", {})
            # Try LLM for richer narrative (history already stripped in pruning below)
            # but fall back to structured template if LLM is unavailable
            pass  # fall through to LLM path below

        # --- LLM PATH: analytical intents that benefit from narrative generation ---
        llm_input_data = outputs.copy()

        # Prune large history arrays before sending to LLM
        if "get_price_movement" in llm_input_data:
            llm_input_data["get_price_movement"] = {
                k: v for k, v in llm_input_data["get_price_movement"].items()
                if k != "history"
            }

        # Prune huge portfolio comparison objects
        if intent == "Portfolio Comparison" and "compare_portfolios" in llm_input_data:
            comp_data = llm_input_data["compare_portfolios"]
            llm_input_data["compare_portfolios"] = {
                "comparison_summary": comp_data.get("comparison_summary", {}),
                "top_performer": comp_data.get("top_performer"),
                "period": comp_data.get("period")
            }

        print(f"[DEBUG] Requesting LLM explanation for intent: {intent}")
        llm_response = llm_generate_explanation(intent, user_query, llm_input_data)
        if llm_response:
            print(f"[DEBUG] LLM explanation OK (length: {len(llm_response)})")
            return llm_response

        print("[DEBUG] LLM unavailable – using Python fallback.")

        # --- PYTHON FALLBACK for LLM-path intents ---
        if intent == "Price Movement Analysis":
            data = outputs.get("get_price_movement", {})
            return (
                f"**Price Movement for {data.get('symbol', 'N/A')}**\n"
                f"- Start price: {data.get('start_price', 'N/A')}\n"
                f"- End price:   {data.get('end_price', 'N/A')}\n"
                f"- Change:      {data.get('pct_change', 'N/A')}%"
            )

        if intent in ["Stock Recommendation", "Stock Research"]:
            stock_data = outputs.get("get_stock_summary") or outputs.get("get_stock_research", {})
            stop_data = outputs.get("get_stop_loss", {})
            return generate_stock_explanation(stock_data, stop_data)

        if intent in ["Hybrid Portfolio Request", "Portfolio Comparison", "Meta Portfolio Request"]:
            port_data = (
                outputs.get("get_hybrid_portfolio")
                or outputs.get("compare_portfolios")
                or outputs.get("get_meta_portfolio", {})
            )
            if not port_data or "error" in port_data:
                port_data = {}

            if "comparison_summary" in port_data:
                summary = port_data["comparison_summary"]
                top_name = port_data.get("top_performer")
                if top_name and top_name in summary:
                    m = summary[top_name]
                    return PORTFOLIO_EXPLANATION_TEMPLATE.format(
                        portfolio_type=f"{intent} (Top: {top_name})",
                        cagr=f"{round(m.get('cagr', 0) * 100, 2)}%",
                        sharpe=m.get("sharpe", "N/A"),
                        stability=m.get("stability", "N/A"),
                        composite_rating=m.get("composite_score", "N/A"),
                        governance_score=port_data.get("governance_score", "N/A"),
                    ).strip()

            metrics = port_data.get("expected_metrics", {})
            return PORTFOLIO_EXPLANATION_TEMPLATE.format(
                portfolio_type=intent,
                cagr=metrics.get("CAGR", port_data.get("cagr", "N/A")),
                sharpe=metrics.get("Sharpe", port_data.get("sharpe", "N/A")),
                stability=metrics.get("Stability", port_data.get("stability_score", "N/A")),
                composite_rating=port_data.get("composite_rating", "N/A"),
                governance_score=port_data.get("governance_score", "N/A"),
            ).strip()

        return GENERIC_EXPLANATION_TEMPLATE.format(context=json.dumps(outputs, indent=2)).strip()

    except Exception as e:
        return f"Could not format explanation safely. Reason: {e}. Raw data extracted."

def process_query(req: AssistantRequest) -> AssistantResponse:
    # 1 & 2. Parse and detect intent
    intent, function_calls = IntentRouter.route_query(req.user_query)
    
    # 3 & 4. Execute functions and merge outputs
    raw_outputs = {}
    called_funcs = []
    
    for call in function_calls:
        for func_name, args in call.items():
            if func_name in REGISTRY:
                try:
                    res = REGISTRY[func_name](args)
                    raw_outputs[func_name] = res
                    called_funcs.append(func_name)
                except Exception as e:
                    raw_outputs[func_name] = {"error": str(e)}

    # 3. Format Explanation
    explanation = format_explanation(intent, raw_outputs, req.user_query)
    
    return AssistantResponse(
        original_query=req.user_query,
        intent=intent,
        functions_called=called_funcs,
        explanation=explanation,
        raw_outputs=raw_outputs
    )
