from unittest.mock import patch

from app.assistant.assistant import process_query
from app.assistant.intent_router import IntentRouter
from app.assistant.query_catalog import (
    ASSISTANT_PROMPT_SUGGESTIONS,
    ASSISTANT_ROADMAP_QUERY_LIBRARY,
    ASSISTANT_RUNNABLE_QUERY_LIBRARY,
    ASSISTANT_SUPPORTED_INTENTS,
)
from app.assistant.schemas import AssistantRequest


SUPPORTED_INTENT_NAMES = {
    "Stock Recommendation",
    "Stock Research",
    "Market Overview",
    "Sector Ranking",
    "Portfolio Comparison",
    "Hybrid Portfolio Request",
    "Automatic N Explanation",
    "Governance Check",
    "Calibration Explanation",
    "Stability Diagnostic",
    "Risk Diagnostics",
}


def _all_runnable_queries():
    for prompt in ASSISTANT_PROMPT_SUGGESTIONS:
        yield prompt
    for queries in ASSISTANT_RUNNABLE_QUERY_LIBRARY.values():
        for query in queries:
            yield query


@patch("app.assistant.llm_engine.llm_route_query", return_value=("", "", "", [], ""))
def test_all_runnable_queries_route_to_supported_intents(_mock_llm):
    failures = []
    for query in _all_runnable_queries():
        intent, function_calls = IntentRouter.route_query(query)
        if intent not in SUPPORTED_INTENT_NAMES or not function_calls:
            failures.append((query, intent, function_calls))

    assert not failures, f"Unroutable runnable queries: {failures}"


def test_supported_intent_examples_are_runnable():
    examples = [row["Example"] for row in ASSISTANT_SUPPORTED_INTENTS]
    with patch("app.assistant.llm_engine.llm_route_query", return_value=("", "", "", [], "")):
        failures = []
        for example in examples:
            intent, function_calls = IntentRouter.route_query(example)
            if not function_calls:
                failures.append((example, intent))

    assert not failures, f"Supported intent examples missing function calls: {failures}"


def test_roadmap_queries_are_not_exposed_as_runnable():
    runnable = set(_all_runnable_queries())
    roadmap = {
        query
        for queries in ASSISTANT_ROADMAP_QUERY_LIBRARY.values()
        for query in queries
    }
    overlap = runnable.intersection(roadmap)
    assert not overlap, f"Roadmap queries should not be runnable: {sorted(overlap)}"


@patch("app.assistant.assistant.llm_generate_explanation", return_value="")
@patch("app.assistant.llm_engine.llm_route_query", return_value=("", "", "", [], ""))
def test_all_runnable_queries_produce_non_empty_response(_mock_llm_route, _mock_llm_explainer):
    mock_registry = {
        "get_stock_summary": lambda _args: {"symbol": "TCS.NS", "composite_score": 82.0, "rank": 5},
        "get_stop_loss": lambda _args: {"symbol": "TCS.NS", "stop_loss_score": 25.0},
        "get_stock_research": lambda _args: {"symbol": "INFY.NS", "composite_score": 78.0, "factors": {"quality": 80.0}},
        "list_stocks": lambda _args: {"stocks": [{"symbol": "TCS.NS", "composite_score": 82.0, "volatility_score": 22.0, "rank": 1}], "date": "2026-04-12"},
        "get_sector_rankings": lambda _args: {"rankings": [{"rank": 1, "sector": "IT", "n_stocks": 10, "cagr": 18.0, "sharpe": 1.4, "avg_score": 75.0}], "period": "2025-10-01 to 2026-04-12"},
        "compare_portfolios": lambda _args: {"comparison_summary": {"Core": {"cagr": 0.12, "sharpe": 1.1, "composite_score": 80.0, "grade": "A"}}, "top_performer": "Core"},
        "get_hybrid_portfolio": lambda _args: {"expected_metrics": {"CAGR": 0.15, "Sharpe": 1.2}, "composite_rating": 82.0},
        "get_auto_n_analysis": lambda _args: {"optimal_n": 12},
        "get_governance": lambda _args: {"warnings": "None detected"},
        "get_latest_calibration": lambda _args: {"calibration_date": "2026-04-12", "drift_score": 0.1},
        "get_stability_analysis": lambda _args: {"stability_score": 77.0},
    }

    failures = []
    with patch("app.assistant.assistant.REGISTRY", mock_registry):
        for query in _all_runnable_queries():
            response = process_query(AssistantRequest(user_query=query))
            if not response.functions_called or not response.explanation.strip():
                failures.append((query, response.intent, response.functions_called, response.explanation))

    assert not failures, f"Runnable queries without usable responses: {failures}"
