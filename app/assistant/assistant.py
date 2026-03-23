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
    generate_portfolio_review_explanation
)
from app.assistant.llm_engine import llm_generate_explanation

def format_explanation(intent: str, outputs: Dict[str, Any], user_query: str = "") -> str:
    """
    Generate an explanation from the structured outputs.
    Routes to dynamic, rule-based response generators.
    """
    try:
        # 1. Try LLM formatting first
        llm_input_data = outputs.copy()
        
        # AGGRESSIVE PRUNING for Portfolio Comparison to avoid LLM timeout
        if intent == "Portfolio Comparison" and "compare_portfolios" in llm_input_data:
            comp_data = llm_input_data["compare_portfolios"]
            # Only send the summary and top performer, drop any huge logs or full result objects
            llm_input_data["compare_portfolios"] = {
                "comparison_summary": comp_data.get("comparison_summary", {}),
                "top_performer": comp_data.get("top_performer"),
                "period": comp_data.get("period")
            }

        if "get_price_movement" in llm_input_data:
            llm_input_data["get_price_movement"] = llm_input_data["get_price_movement"].copy()
            if "history" in llm_input_data["get_price_movement"]:
                del llm_input_data["get_price_movement"]["history"]

        print(f"[DEBUG] Generating LLM explanation for intent: {intent}")
        llm_response = llm_generate_explanation(intent, user_query, llm_input_data)
        if llm_response:
            print(f"[DEBUG] LLM explanation SUCCESS (length: {len(llm_response)})")
            return llm_response
        
        print("[DEBUG] LLM explanation returned empty. Falling back to python generator.")
        # 2. Fallback to python string builders
        if intent == "Portfolio Review":
            return generate_portfolio_review_explanation(outputs.get("get_portfolio_review", {}))

        if intent == "Multi-Stock Review":
            return generate_portfolio_review_explanation(outputs.get("get_multi_stock_review", {}))

        if intent == "Price Movement Analysis":
            data = outputs.get("get_price_movement", {})
            return f"Price Movement for {data.get('symbol')}:\n" \
                   f"- Start: {data.get('start_price')}\n" \
                   f"- End: {data.get('end_price')}\n" \
                   f"- Change: {data.get('pct_change')}%"
            
        if intent == "Market Overview":
            return generate_market_overview_explanation(outputs.get("list_stocks", {}))
            
        if intent in ["Stock Recommendation", "Stock Research"]:
            stock_data = outputs.get("get_stock_summary", {})
            if not stock_data:
                stock_data = outputs.get("get_stock_research", {})
            stop_data = outputs.get("get_stop_loss", {})
                
            return generate_stock_explanation(stock_data, stop_data)
            
        elif intent in ["Hybrid Portfolio Request", "Portfolio Comparison", "Meta Portfolio Request"]:
            port_data = outputs.get("get_hybrid_portfolio")
            if not port_data or "error" in port_data:
                port_data = outputs.get("compare_portfolios")
                
            if not port_data or "error" in port_data:
                port_data = outputs.get("get_meta_portfolio", {})
                
            # If it's a comparison, we have multiple portfolios in comparison_summary
            if "comparison_summary" in port_data:
                summary = port_data["comparison_summary"]
                top_name = port_data.get("top_performer")
                
                # If we have a top performer, use its metrics for the primary display
                if top_name and top_name in summary:
                    m = summary[top_name]
                    return PORTFOLIO_EXPLANATION_TEMPLATE.format(
                        portfolio_type=f"{intent} (Top: {top_name})",
                        cagr=f"{round(m.get('cagr', 0)*100, 2)}%",
                        sharpe=m.get('sharpe', 'N/A'),
                        stability=m.get('stability', 'N/A'),
                        composite_rating=m.get('composite_score', 'N/A'),
                        governance_score=port_data.get("governance_score", "N/A")
                    ).strip()
                
            # Single portfolio fallback
            metrics = port_data.get("expected_metrics", {})
            cagr = metrics.get("CAGR", port_data.get("cagr", "N/A"))
            sharpe = metrics.get("Sharpe", port_data.get("sharpe", "N/A"))
            stability = metrics.get("Stability", port_data.get("stability_score", "N/A"))
            comp_rating = port_data.get("composite_rating", "N/A")
            
            return PORTFOLIO_EXPLANATION_TEMPLATE.format(
                portfolio_type=intent,
                cagr=cagr,
                sharpe=sharpe,
                stability=stability,
                composite_rating=comp_rating,
                governance_score=port_data.get("governance_score", "N/A")
            ).strip()
            
        elif intent in ["Governance Check", "Calibration Explanation", "Stability Diagnostic"]:
            gov_data = outputs.get("get_governance", {})
            return generate_governance_explanation(gov_data)
            
        elif intent == "Risk Diagnostics":
            stop_data = outputs.get("get_stop_loss", {})
            # Look up symbol from outputs or context (if available). Usually the first key has it or we can fallback:
            symbol = stop_data.get("symbol", "Target Stock")
            return generate_risk_explanation(stop_data, symbol)
            
        elif intent == "Automatic N Explanation":
            auto_data = outputs.get("get_auto_n_analysis", {})
            return generate_auto_n_explanation(auto_data)

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
