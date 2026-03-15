from pydantic import BaseModel, Field


class GradingReport(BaseModel):
    """Strict JSON output schema for backtest grading results.

    All metrics are computed from a vectorized Polars-based backtest engine
    with realistic T+1 execution, configurable fees, and slippage modeling.
    """

    symbol: str = Field(description="Traded symbol, e.g. BTC/USDT")
    strategy_name: str = Field(description="Name of the strategy class")

    # Core performance metrics
    cagr: float = Field(
        description="Compound Annual Growth Rate as decimal (0.15 = 15%)"
    )
    max_drawdown: float = Field(
        description="Maximum drawdown magnitude as decimal (negative, e.g. -0.25 = -25%)"
    )
    max_drawdown_duration_candles: int = Field(
        description="Duration of the longest drawdown period in candles"
    )
    sharpe_ratio: float = Field(
        description="Annualized Sharpe ratio (risk-free rate = 0)"
    )
    sortino_ratio: float = Field(
        description="Annualized Sortino ratio (downside deviation only)"
    )
    win_rate: float = Field(
        description="Fraction of winning trades (0.0 to 1.0)"
    )

    # Agentic feedback
    failure_narrative: str = Field(
        description="Concise 2-sentence failure/success analysis relative to market regimes"
    )

    # Metadata
    total_trades: int = Field(description="Total number of round-trip trades executed")
    total_return: float = Field(
        description="Total cumulative return as decimal (0.5 = 50%)"
    )
    fee_bps: float = Field(description="Fee in basis points used for this backtest")
    slippage_bps: float = Field(
        description="Slippage in basis points used for this backtest"
    )
    candles_evaluated: int = Field(
        description="Total number of candles in the dataset"
    )
