import unittest
from unittest.mock import patch, MagicMock
from app.assistant.intent_router import IntentRouter
from app.assistant.function_registry import (
    get_portfolio_review, 
    get_multi_stock_review,
    REGISTRY
)
from app.assistant.assistant import format_explanation

class TestAssistant(unittest.TestCase):

    def test_intent_router_market_overview(self):
        """Test extraction of top_n from market overview queries."""
        # Simple regex check in IntentRouter
        intent, calls = IntentRouter.route_query("List top 15 stocks")
        self.assertEqual(intent, "Market Overview")
        self.assertEqual(calls[0]["list_stocks"]["top_n"], 15)

    def test_intent_router_market_overview_alt_phrasing(self):
        """Top N extraction should work even when the number comes before 'top stocks'."""
        intent, calls = IntentRouter.route_query("Show me 15 top stocks in the market")
        self.assertEqual(intent, "Market Overview")
        self.assertEqual(calls[0]["list_stocks"]["top_n"], 15)

    @patch("app.assistant.llm_engine.llm_route_query")
    def test_multi_stock_review_routing(self, mock_route):
        """Test that multi-stock review intent correctly maps to symbols list."""
        mock_route.return_value = ("Multi-Stock Review", "", "", ["TCS.NS", "INFY.NS"])
        
        intent, calls = IntentRouter.route_query("Review TCS and INFY")
        self.assertEqual(intent, "Multi-Stock Review")
        self.assertIn("symbols", calls[0]["get_multi_stock_review"])
        self.assertEqual(calls[0]["get_multi_stock_review"]["symbols"], ["TCS.NS", "INFY.NS"])

    @patch("services.recommendation.get_stock_signal")
    def test_get_multi_stock_review_function(self, mock_signal):
        """Test the logic of get_multi_stock_review wrapper."""
        mock_signal.return_value = {
            "composite_score": 0.85,
            "recommendation": "Buy",
            "rank": 5,
            "risk_level": "Low"
        }
        
        params = {"symbols": ["RELIANCE.NS"]}
        result = get_multi_stock_review(params)
        
        self.assertIn("stock_reviews", result)
        self.assertEqual(len(result["stock_reviews"]), 1)
        self.assertEqual(result["stock_reviews"][0]["symbol"], "RELIANCE.NS")
        self.assertEqual(result["stock_reviews"][0]["score"], 85.0)

    @patch("services.user_portfolios.load_user_portfolios")
    @patch("services.recommendation.get_stock_signal")
    def test_get_portfolio_review_found(self, mock_signal, mock_load):
        """Test getting review for an existing portfolio."""
        mock_load.return_value = [
            {"name": "Test_P", "symbols": ["AAPL.NS"], "strategy": "equal_weight"}
        ]
        mock_signal.return_value = {"composite_score": 0.5, "recommendation": "Hold"}
        
        res = get_portfolio_review({"portfolio_name": "Test_P"})
        self.assertEqual(res["portfolio_name"], "Test_P")
        self.assertEqual(len(res["stock_reviews"]), 1)
        self.assertEqual(res["stock_reviews"][0]["score"], 50.0) # 0.5 * 100

    def test_portfolio_review_fallback_formatting(self):
        """Test that the python fallback generates a table."""
        data = {
            "portfolio_name": "Manju_Portfolio",
            "strategy": "equal_weight",
            "stock_reviews": [
                {"symbol": "TCS.NS", "score": 0.4, "recommendation": "Hold", "rank": 150, "risk": "Low"}
            ]
        }
        # Force fallback by passing empty user_query (so LLM logic isn't triggered or we mock it)
        with patch("app.assistant.assistant.llm_generate_explanation", return_value=""):
            explanation = format_explanation("Portfolio Review", {"get_portfolio_review": data})
            self.assertIn("Manju_Portfolio Review", explanation)
            self.assertIn("| Symbol | Score |", explanation)
            self.assertIn("TCS.NS", explanation)

    def test_intent_router_recommend_stocks(self):
        """Test that 'Recommend 10 stocks' is Market Overview, not a symbol."""
        intent, _ = IntentRouter.route_query("Recommend 10 stocks to buy")
        self.assertEqual(intent, "Market Overview")
        
        # Verify 'Recommend' is NOT extracted as a symbol
        symbol = IntentRouter.extract_symbol("Recommend stocks")
        self.assertEqual(symbol, "UNKNOWN")

    def test_portfolio_comparison_summary_formatting(self):
        """Test that comparison summary formats CAGR as percentage and identifies Top Performer."""
        data = {
            "comparison_summary": {
                "Alpha_P": {"cagr": 0.25, "sharpe": 1.5, "composite_score": 80.0, "grade": "A"},
                "Beta_P": {"cagr": 0.10, "sharpe": 0.8, "composite_score": 60.0, "grade": "B"}
            },
            "top_performer": "Alpha_P"
        }
        with patch("app.assistant.assistant.llm_generate_explanation", return_value=""):
            explanation = format_explanation("Portfolio Comparison", {"compare_portfolios": data})
            self.assertIn("Top: Alpha_P", explanation)
            self.assertIn("25.0%", explanation) # 0.25 * 100

    @patch("app.database.get_db_context")
    @patch("services.portfolio.get_latest_scoring_date")
    @patch("services.recommendation.get_stock_signal")
    def test_get_stock_research_fallback(self, mock_signal, mock_date, mock_db):
        """Test factor fallback for LT.NS zero-score case."""
        from app.assistant.function_registry import get_stock_research
        import datetime
        
        mock_signal.return_value = {"composite_score": 0.04, "trend_direction": "Up"}
        mock_date.return_value = datetime.date(2026, 3, 19)
        
        # Mock DB results
        mock_stock = MagicMock(id=101)
        mock_f_latest = MagicMock(quality_score=0, value_score=0, growth_score=0, date=datetime.date(2026, 3, 19))
        mock_f_old = MagicMock(quality_score=0.45, value_score=0.3, growth_score=0.5, date=datetime.date(2026, 3, 10))
        
        context = mock_db.return_value.__enter__.return_value
        # 1. Stock lookup
        # 2. Latest Factor lookup (is all zero)
        # 3. Fallback Factor lookup
        # 4. Fundamental lookup (None)
        context.execute.return_value.scalars.return_value.first.side_effect = [mock_stock, mock_f_latest, mock_f_old, None]
        
        result = get_stock_research({"symbol": "LT.NS"})
        
        self.assertEqual(result["analysis_date"], "2026-03-10")
        self.assertEqual(result["factors"]["quality"], 45.0)
        self.assertEqual(result["composite_score"], 4.0)

    def test_scaling_research(self):
        """Verify 100x scaling in get_stock_research."""
        signal_data = {
            "symbol": "MTARTECH.NS",
            "composite_score": 1.308, 
            "recommendation": "Buy",
            "trend_direction": "Up",
            "rank": 1,
            "regime": "Low Vol",
            "momentum_score": 0.5,
            "volatility_score": 0.2
        }
        
        from unittest.mock import patch
        with patch("services.recommendation.get_stock_signal") as mock_signal:
            with patch("app.database.get_db_context") as mock_db:
                mock_signal.side_effect = [signal_data] * 20
                mock_stock = MagicMock(id=1)
                mock_f = MagicMock(quality_score=0.8, value_score=0.5, growth_score=0.7)
                context = mock_db.return_value.__enter__.return_value
                context.execute.return_value.scalars.return_value.first.side_effect = [mock_stock, mock_f, None] * 20
                
                from app.assistant.function_registry import get_stock_research
                res = get_stock_research({"symbol": "MTARTECH.NS"})
                self.assertEqual(res["composite_score"], 130.8)

    @patch("services.recommendation.get_stock_signal")
    def test_scaling_multi_review(self, mock_signal):
        """Verify 100x scaling in get_multi_stock_review."""
        mock_signal.return_value = {"composite_score": 1.308, "recommendation": "Buy"}
        res = get_multi_stock_review({"symbols": ["MTARTECH.NS"]})
        self.assertEqual(res["stock_reviews"][0]["score"], 130.8)

    @patch("app.database.get_db_context")
    @patch("services.financial_utils.fetch_live_fundamentals")
    @patch("services.recommendation.get_stock_signal")
    def test_fundamental_live_fallback(self, mock_signal, mock_live, mock_db):
        """Test that missing DB fundamentals trigger live yfinance fetch."""
        from app.assistant.function_registry import get_stock_research
        # 1. Stock lookup -> Found
        # 2. Latest Factor lookup -> Found
        # 3. Fundamental lookup -> NONE (triggers live fetch)
        mock_stock = MagicMock(id=2)
        mock_f = MagicMock(quality_score=0.8, value_score=0.5, growth_score=0.7)
        # Add a placeholder for any other accidental calls
        context = mock_db.return_value.__enter__.return_value
        context.execute.return_value.scalars.return_value.first.side_effect = [mock_stock, mock_f, None, None, None]
        
        mock_live.return_value = {"eps": 10.0, "roe": 20.0, "source": "yfinance_live"}
        
        result = get_stock_research({"symbol": "TCS.NS"})
        self.assertEqual(result["fundamentals"]["source"], "yfinance_live")
        self.assertEqual(result["fundamentals"]["eps"], 10.0)

    @patch("app.assistant.llm_engine.llm_route_query")
    def test_intent_router_disambiguation(self, mock_llm):
        """Test differentiation between stock compare and portfolio compare."""
        # 1. Stock Compare (should NOT hit portfolio regex)
        mock_llm.return_value = ("Multi-Stock Review", "", "", ["TCS.NS", "INFY.NS"])
        intent, _ = IntentRouter.route_query("Compare TCS and INFY")
        self.assertEqual(intent, "Multi-Stock Review")
        
        # 2. Portfolio Compare (should hit intent)
        mock_llm.return_value = ("Portfolio Comparison", "", "Primary", [])
        intent, _ = IntentRouter.route_query("Compare my portfolios")
        self.assertEqual(intent, "Portfolio Comparison")

if __name__ == "__main__":
    unittest.main()
