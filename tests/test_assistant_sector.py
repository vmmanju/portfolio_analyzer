import unittest
from unittest.mock import patch, MagicMock
from app.assistant.intent_router import IntentRouter
from app.assistant.function_registry import get_sector_rankings, REGISTRY
from app.assistant.assistant import format_explanation
import pandas as pd
from datetime import date

class TestAssistantSector(unittest.TestCase):

    @patch("app.assistant.llm_engine.llm_route_query")
    def test_intent_router_sector_ranking(self, mock_llm):
        """Test that sector ranking queries are correctly routed."""
        # Mock LLM to return empty so Regex fallback is tested
        mock_llm.return_value = ("", "", "", [], "")
        
        queries = [
            "Show me top 10 sectors",
            "What are the best sectors?",
            "rank sectors by performance",
            "top 5 sectors"
        ]
        for q in queries:
            intent, calls = IntentRouter.route_query(q)
            self.assertEqual(intent, "Sector Ranking", f"Failed for query: {q}")
            self.assertIn("get_sector_rankings", calls[0])
            
        # Test N extraction
        intent, calls = IntentRouter.route_query("top 5 sectors")
        self.assertEqual(calls[0]["get_sector_rankings"]["top_n"], 5)

    @patch("services.sector_analytics.compute_sector_relative_performance")
    @patch("services.portfolio.get_latest_scoring_date")
    def test_get_sector_rankings_function(self, mock_date, mock_service):
        """Test the get_sector_rankings registry function."""
        mock_date.return_value = date(2026, 3, 25)
        
        # Mock DataFrame return from service
        df_mock = pd.DataFrame([
            {"sector": "Technology", "n_stocks": 50, "cagr": 0.25, "sharpe": 1.2, "max_drawdown": -0.15, "avg_score": 0.75},
            {"sector": "Financials", "n_stocks": 40, "cagr": 0.15, "sharpe": 0.8, "max_drawdown": -0.10, "avg_score": 0.65}
        ])
        mock_service.return_value = df_mock
        
        result = get_sector_rankings({"top_n": 2})
        
        self.assertIn("rankings", result)
        self.assertEqual(len(result["rankings"]), 2)
        self.assertEqual(result["rankings"][0]["sector"], "Technology")
        self.assertEqual(result["rankings"][0]["cagr"], 25.0)
        self.assertEqual(result["rankings"][0]["sharpe"], 1.2)
        self.assertEqual(result["rankings"][0]["avg_score"], 75.0)

    def test_sector_ranking_formatting(self):
        """Test the markdown formatting for sector rankings."""
        data = {
            "rankings": [
                {"rank": 1, "sector": "Technology", "n_stocks": 50, "cagr": 25.0, "sharpe": 1.2, "avg_score": 75.0},
                {"rank": 2, "sector": "Financials", "n_stocks": 40, "cagr": 15.0, "sharpe": 0.8, "avg_score": 65.0}
            ],
            "period": "2025-09-26 to 2026-03-25"
        }
        
        # Force fallback to python generator
        with patch("app.assistant.assistant.llm_generate_explanation", return_value=""):
            explanation = format_explanation("Sector Ranking", {"get_sector_rankings": data})
            
            self.assertIn("### Top Sector Rankings", explanation)
            self.assertIn("| Rank | Sector |", explanation)
            self.assertIn("Technology", explanation)
            self.assertIn("25.0%", explanation)
            self.assertIn("Financials", explanation)

if __name__ == "__main__":
    unittest.main()
