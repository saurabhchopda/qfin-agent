"""Typed contracts shared across all qfin-agent components."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Recommendation(str, Enum):
    """Final action recommendation."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskDecision(str, Enum):
    """Risk manager policy decision."""

    APPROVE = "approve"
    FLAG = "flag"
    REJECT = "reject"


class TrendDirection(str, Enum):
    """Price trend category from technical analysis."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class VolatilityRegime(str, Enum):
    """Volatility regime inferred from ATR ratio."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class OHLCVBar(BaseModel):
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataRequest(BaseModel):
    """Request for historical market data."""

    ticker: str = Field(min_length=1)
    period: str = Field(default="6mo")
    interval: str = Field(default="1d")


class PortfolioPosition(BaseModel):
    """Open position in the portfolio."""

    ticker: str
    weight: float = Field(ge=0.0, le=1.0)


class PortfolioState(BaseModel):
    """Current portfolio and risk state."""

    cash_weight: float = Field(ge=0.0, le=1.0)
    positions: list[PortfolioPosition] = Field(default_factory=list)
    gross_exposure: float = Field(default=0.0, ge=0.0)
    max_drawdown_pct: float = Field(default=0.0, ge=0.0)


class TechnicalAnalysisInput(BaseModel):
    """Input contract for technical analysis agent."""

    ticker: str
    ohlcv: list[OHLCVBar]


class TechnicalIndicators(BaseModel):
    """Snapshot of key indicator values."""

    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    rsi_14: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    atr_14: float


class TechnicalAnalysisOutput(BaseModel):
    """Output contract for technical analysis agent."""

    ticker: str
    trend: TrendDirection
    momentum: str
    volatility_regime: VolatilityRegime
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: TechnicalIndicators
    rationale: list[str] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    """Evidence excerpt used by the fundamental analyst."""

    source: str
    title: str
    snippet: str
    published_at: datetime | None = None


class FundamentalAnalysisInput(BaseModel):
    """Input contract for fundamental analysis agent."""

    ticker: str
    query: str = Field(default="recent earnings guidance, news sentiment, and catalysts")
    top_k: int = Field(default=5, ge=1, le=20)


class FundamentalAnalysisOutput(BaseModel):
    """Output contract for fundamental analysis agent."""

    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    summary: str
    evidence: list[EvidenceSnippet] = Field(default_factory=list)


class RiskViolation(BaseModel):
    """Individual breached risk rule."""

    rule: str
    message: str
    severity: str = Field(default="medium")


class RiskEvaluationInput(BaseModel):
    """Input contract for risk manager agent."""

    ticker: str
    fundamental: FundamentalAnalysisOutput
    technical: TechnicalAnalysisOutput
    portfolio: PortfolioState


class RiskManagerOutput(BaseModel):
    """Output contract for risk manager agent."""

    ticker: str
    decision: RiskDecision
    recommendation: Recommendation
    target_position_weight: float = Field(ge=0.0, le=1.0)
    violations: list[RiskViolation] = Field(default_factory=list)
    rationale: str


class SupervisorRequest(BaseModel):
    """Top-level workflow input."""

    ticker: str
    technical_input: TechnicalAnalysisInput
    fundamental_input: FundamentalAnalysisInput
    portfolio: PortfolioState


class DebateTurn(BaseModel):
    """Single agent contribution in the supervisor debate."""

    speaker: str
    stance: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class DebateConflict(BaseModel):
    """Conflict detected across agent recommendations."""

    topic: str
    participants: list[str] = Field(default_factory=list)
    description: str
    resolution: str


class DebateTranscript(BaseModel):
    """Structured transcript of debate turns and resolution."""

    turns: list[DebateTurn] = Field(default_factory=list)
    conflicts: list[DebateConflict] = Field(default_factory=list)
    consensus_strength: float = Field(ge=0.0, le=1.0)
    final_resolution: str


class SupervisorOutput(BaseModel):
    """Top-level workflow output."""

    ticker: str
    final_recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    debate_transcript: DebateTranscript
    fundamental: FundamentalAnalysisOutput
    technical: TechnicalAnalysisOutput
    risk: RiskManagerOutput

    model_config = ConfigDict(use_enum_values=True)


class BacktestConfig(BaseModel):
    """Configuration for simple historical simulation."""

    ticker: str
    lookback_bars: int = Field(default=60, ge=20)
    initial_capital: float = Field(default=100_000.0, gt=0.0)
    position_size: float = Field(default=1.0, ge=0.0, le=1.0)
    transaction_cost_bps: float = Field(default=2.0, ge=0.0)


class BacktestSignal(BaseModel):
    """Single recommendation emitted by a recommendation engine."""

    recommendation: Recommendation
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class BacktestStep(BaseModel):
    """Per candlebar backtest record."""

    timestamp: datetime
    close: float
    market_return: float
    strategy_return: float
    position: float
    equity: float
    recommendation: Recommendation
    confidence: float


class BacktestResult(BaseModel):
    """Aggregate backtest output and equity curve."""

    config: BacktestConfig
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    steps: list[BacktestStep] = Field(default_factory=list)


class BacktestSummary(BaseModel):
    """Compact backtest metrics used by walk-forward evaluation."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    final_equity: float


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation."""

    ticker: str
    lookback_bars: int = Field(default=60, ge=20)
    train_bars: int = Field(default=126, ge=20)
    test_bars: int = Field(default=63, ge=20)
    step_bars: int = Field(default=63, ge=1)
    initial_capital: float = Field(default=100_000.0, gt=0.0)
    position_size: float = Field(default=1.0, ge=0.0, le=1.0)
    transaction_cost_bps: float = Field(default=2.0, ge=0.0)


class WalkForwardWindowResult(BaseModel):
    """Per-window walk-forward evaluation result."""

    window_index: int = Field(ge=1)
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    regime: str
    supervisor: BacktestSummary
    benchmark: BacktestSummary
    excess_return: float


class RegimeAggregate(BaseModel):
    """Aggregate performance within a market regime."""

    regime: str
    windows: int = Field(ge=0)
    avg_excess_return: float
    outperformance_rate: float = Field(ge=0.0, le=1.0)


class WalkForwardAggregate(BaseModel):
    """Portfolio-level aggregates across all walk-forward windows."""

    total_windows: int = Field(ge=0)
    supervisor_avg_return: float
    benchmark_avg_return: float
    avg_excess_return: float
    outperformance_rate: float = Field(ge=0.0, le=1.0)
    regimes: list[RegimeAggregate] = Field(default_factory=list)


class WalkForwardResult(BaseModel):
    """Full walk-forward output with windows and aggregated metrics."""

    config: WalkForwardConfig
    windows: list[WalkForwardWindowResult] = Field(default_factory=list)
    aggregate: WalkForwardAggregate
