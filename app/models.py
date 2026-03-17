"""ORM models for users, stocks, prices, fundamentals, factors, scores,
portfolio_allocations, monthly_allocations, portfolio_metrics."""

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# User model (multi-user auth)
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    """Application user, populated from Google OAuth profile."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    picture_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    # Relationships to per-user data
    portfolios: Mapped[List["UserPortfolio"]] = relationship(
        "UserPortfolio", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email!r})>"


class Stock(Base):
    """Stock master table."""

    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships (bidirectional)
    prices: Mapped[List["Price"]] = relationship(
        "Price", back_populates="stock", cascade="all, delete-orphan"
    )
    fundamentals: Mapped[List["Fundamental"]] = relationship(
        "Fundamental", back_populates="stock", cascade="all, delete-orphan"
    )
    factors: Mapped[List["Factor"]] = relationship(
        "Factor", back_populates="stock", cascade="all, delete-orphan"
    )
    scores: Mapped[List["Score"]] = relationship(
        "Score", back_populates="stock", cascade="all, delete-orphan"
    )
    portfolio_allocations: Mapped[List["PortfolioAllocation"]] = relationship(
        "PortfolioAllocation", back_populates="stock", cascade="all, delete-orphan"
    )
    monthly_allocations: Mapped[List["MonthlyAllocation"]] = relationship(
        "MonthlyAllocation", back_populates="stock", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Stock(id={self.id}, symbol={self.symbol!r})>"


class Price(Base):
    """Daily OHLCV price data per stock."""

    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_prices_stock_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="prices")

    def __repr__(self) -> str:
        return f"<Price(id={self.id}, stock_id={self.stock_id}, date={self.date})>"


class Fundamental(Base):
    """Quarterly fundamental metrics per stock."""

    __tablename__ = "fundamentals"
    __table_args__ = (UniqueConstraint("stock_id", "quarter", name="uq_fundamentals_stock_quarter"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    quarter: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # e.g. "2024-Q1"
    revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_income: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roce: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    debt_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    free_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="fundamentals")

    def __repr__(self) -> str:
        return f"<Fundamental(id={self.id}, stock_id={self.stock_id}, quarter={self.quarter!r})>"


class Factor(Base):
    """Factor scores per stock per date."""

    __tablename__ = "factors"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_factors_stock_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    value_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    growth_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    momentum_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="factors")

    def __repr__(self) -> str:
        return f"<Factor(id={self.id}, stock_id={self.stock_id}, date={self.date})>"


class Score(Base):
    """Composite score and rank per stock per date."""

    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_scores_stock_date"),
        Index("ix_scores_date_rank", "date", "rank"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="scores")

    def __repr__(self) -> str:
        return f"<Score(id={self.id}, stock_id={self.stock_id}, date={self.date})>"


class PortfolioAllocation(Base):
    """Portfolio weight per stock per date and strategy."""

    __tablename__ = "portfolio_allocations"
    __table_args__ = (
        Index("ix_portfolio_allocations_date_strategy", "date", "strategy_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=False)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strategy_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="portfolio_allocations")

    def __repr__(self) -> str:
        return f"<PortfolioAllocation(id={self.id}, date={self.date}, stock_id={self.stock_id})>"


class UserPortfolio(Base):
    """User-defined portfolio definitions."""

    __tablename__ = "user_portfolios"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_portfolio_user_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    name: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    strategy_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    regime_mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    top_n: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped[Optional["User"]] = relationship("User", back_populates="portfolios")
    stocks: Mapped[List["UserPortfolioStock"]] = relationship(
        "UserPortfolioStock", back_populates="portfolio", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<UserPortfolio(id={self.id}, name={self.name!r}, user_id={self.user_id})>"


class UserPortfolioStock(Base):
    """Mapping table: user portfolio -> stock symbol (and optional stock_id)."""

    __tablename__ = "user_portfolio_stocks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("user_portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    stock_id: Mapped[Optional[int]] = mapped_column(ForeignKey("stocks.id", ondelete="CASCADE"), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    portfolio: Mapped["UserPortfolio"] = relationship("UserPortfolio", back_populates="stocks")
    stock: Mapped[Optional["Stock"]] = relationship("Stock")

    def __repr__(self) -> str:
        return f"<UserPortfolioStock(id={self.id}, portfolio_id={self.portfolio_id}, symbol={self.symbol!r})>"


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Monthly allocation persistence
# ─────────────────────────────────────────────────────────────────────────────

class MonthlyAllocation(Base):
    """Persisted weight per stock for each monthly portfolio rebalance.

    Design decisions
    ----------------
    * Historical records are NEVER overwritten.  A new rebalance on the same
      (portfolio_type, portfolio_name, rebalance_date, stock_id) tuple is blocked
      by the unique constraint — callers must supply a new rebalance_date or use
      upsert logic when they intentionally want to replace a snapshot.
    * portfolio_name is nullable to support anonymous strategy runs (e.g. the
      system auto-hybrid that has no user-defined name).
    * stock_id FK references stocks(id) with SET NULL on delete so that removing
      a delisted stock does not cascade-delete the historical weight record.
    """

    __tablename__ = "monthly_allocations"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "portfolio_type", "portfolio_name", "rebalance_date", "stock_id",
            name="uq_monthly_alloc_user_type_name_date_stock",
        ),
        Index("ix_monthly_alloc_type_date", "portfolio_type", "rebalance_date"),
        Index("ix_monthly_alloc_stock",     "stock_id"),
        Index("ix_monthly_alloc_user",      "user_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True,
    )

    # Portfolio identity
    portfolio_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="e.g. 'auto_diversified_hybrid', 'equal_weight', 'user'",
    )
    portfolio_name: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True,
        comment="User-defined name; NULL for system portfolios",
    )

    # Rebalance context
    rebalance_date: Mapped[date] = mapped_column(
        Date, nullable=False,
        comment="Month-end trading date on which this allocation was built",
    )

    # Allocation
    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="SET NULL"), nullable=False,
    )
    weight: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Portfolio weight in [0, 1]; weights for a given snapshot sum to ~1",
    )

    optimal_n_used: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
        comment="If auto-n was used, the selected N. Otherwise null.",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Whether this was a forced rebalance (bypassed the monthly guard)
    forced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="monthly_allocations")

    def __repr__(self) -> str:
        return (
            f"<MonthlyAllocation(id={self.id}, type={self.portfolio_type!r}, "
            f"date={self.rebalance_date}, stock_id={self.stock_id}, weight={self.weight:.4f})>"
        )


class PortfolioMetrics(Base):
    """Performance and stability snapshot taken at each monthly rebalance.

    Design decisions
    ----------------
    * Records are INSERT-only — historical snapshots are immutable.
    * The unique constraint on (portfolio_type, portfolio_name, rebalance_date)
      prevents duplicate snapshots for the same period.  If a forced rebalance
      replaces the same month's run, callers should delete the old record first
      or update it explicitly (upsert).
    * All metric columns are nullable to gracefully handle cases where a metric
      could not be computed (e.g. insufficient data for Sharpe).
    """

    __tablename__ = "portfolio_metrics"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "portfolio_type", "portfolio_name", "rebalance_date",
            name="uq_portfolio_metrics_user_type_name_date",
        ),
        Index("ix_portfolio_metrics_type_date", "portfolio_type", "rebalance_date"),
        Index("ix_portfolio_metrics_user",     "user_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True,
    )

    # Portfolio identity
    portfolio_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    portfolio_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Period
    rebalance_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Standard performance metrics (annualised where applicable)
    cagr: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Compound Annual Growth Rate"
    )
    volatility: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Annualised volatility"
    )
    sharpe: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Annualised Sharpe ratio (rf=0)"
    )
    max_drawdown: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Maximum drawdown (negative value)"
    )
    calmar: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="CAGR / abs(Max Drawdown)"
    )
    sortino: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Annualised Sortino ratio"
    )
    total_return: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Total return over backtest window"
    )

    # Stability layer (from stability_analyzer)
    stability_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Composite stability score 0-100"
    )
    stability_grade: Mapped[Optional[str]] = mapped_column(
        String(4), nullable=True, comment="A/B/C/D/F letter grade"
    )
    sharpe_consistency_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    drawdown_stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    turnover_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corr_stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regime_robustness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Composite rating (from portfolio_comparison.rate_portfolios)
    composite_rating: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Composite rating score (0-1 normalised)"
    )
    composite_grade: Mapped[Optional[str]] = mapped_column(
        String(4), nullable=True, comment="A/B/C/D peer-relative grade"
    )

    # Portfolio construction context
    n_stocks: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Number of stocks in this allocation"
    )
    expected_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_pairwise_corr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    forced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return (
            f"<PortfolioMetrics(id={self.id}, type={self.portfolio_type!r}, "
            f"date={self.rebalance_date}, sharpe={self.sharpe})>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Error-model Bayesian calibration history
# ─────────────────────────────────────────────────────────────────────────────

class ModelCalibration(Base):
    """Persisted record of each error-model Bayesian coefficient calibration.

    Design decisions
    ----------------
    * One row per calibration run.  The Bayesian blending rule
      (posterior = w × new + (1−w) × old) is applied at write time so the
      stored coefficients are always the *posterior* values that the system
      will use when making predictions.
    * Unique constraint on calibration_end_date ensures only one calibration
      is recorded per boundary date, preventing duplicate runs on the same day.
    * All coefficient columns are nullable to gracefully handle edge cases
      (e.g. insufficient data for a reliable OLS fit).
    * raw_* columns store the raw OLS output BEFORE Bayesian blending, enabling
      forensic analysis of how much the prior was influencing results over time.
    """

    __tablename__ = "model_calibration"
    __table_args__ = (
        UniqueConstraint("calibration_end_date", name="uq_model_calibration_end_date"),
        Index("ix_model_calibration_end_date", "calibration_end_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Estimation window
    calibration_start_date: Mapped[date] = mapped_column(
        Date, nullable=False,
        comment="First date of the 5-year rolling estimation window",
    )
    calibration_end_date: Mapped[date] = mapped_column(
        Date, nullable=False, index=True,
        comment="Last date of the estimation window (anchor for +6-month trigger)",
    )

    # Bayesian-posterior coefficients (what the system uses)
    intercept: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Posterior β0 after Bayesian blending"
    )
    beta_momentum: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Posterior β_MOM after Bayesian blending"
    )
    beta_quality: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Posterior β_QUAL after Bayesian blending"
    )
    beta_value: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Posterior β_VAL after Bayesian blending"
    )
    beta_volatility: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Posterior β_VOL after Bayesian blending"
    )

    # OLS quality metrics (for the new window, before blending)
    r_squared: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="OLS R² on the new estimation window"
    )
    adj_r_squared: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Adjusted R²"
    )
    n_observations: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Number of (stock, month) pairs in OLS"
    )
    residual_std: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Std deviation of OLS residuals"
    )
    mean_error: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Mean prediction error (bias) on the window"
    )

    # Raw OLS coefficients BEFORE Bayesian blending (for diagnostics)
    raw_intercept: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_beta_momentum: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_beta_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_beta_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_beta_volatility: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Blending metadata
    blend_weight: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="w used in posterior = w*new + (1-w)*old (0.0 means first calibration)",
    )
    prediction_method: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, comment="'shrinkage' or 'score'"
    )
    triggered_by: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        comment="'scheduled' (6-month trigger) or 'manual'",
    )

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelCalibration(id={self.id}, "
            f"window=[{self.calibration_start_date},{self.calibration_end_date}], "
            f"β_MOM={self.beta_momentum}, R²={self.r_squared})>"
        )

