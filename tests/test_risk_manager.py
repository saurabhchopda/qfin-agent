from __future__ import annotations

from qfin_agent.agents.risk_manager import RiskManager
from qfin_agent.config import Settings
from qfin_agent.models.schemas import (
    FundamentalAnalysisOutput,
    PortfolioState,
    RiskDecision,
    RiskEvaluationInput,
    TechnicalAnalysisOutput,
    TechnicalIndicators,
    TrendDirection,
    VolatilityRegime,
)


def _technical_output(trend: TrendDirection, momentum: str) -> TechnicalAnalysisOutput:
    return TechnicalAnalysisOutput(
        ticker="AAPL",
        trend=trend,
        momentum=momentum,
        volatility_regime=VolatilityRegime.NORMAL,
        confidence=0.8,
        indicators=TechnicalIndicators(
            sma_fast=101,
            sma_slow=99,
            ema_fast=101,
            ema_slow=99,
            rsi_14=62,
            macd=1.2,
            macd_signal=1.0,
            bb_upper=104,
            bb_lower=96,
            atr_14=1.2,
        ),
        rationale=["test"],
    )


def test_risk_manager_flags_gross_exposure_breach() -> None:
    settings = Settings(max_gross_exposure=0.66, default_target_weight=0.05)
    manager = RiskManager(settings=settings)

    output = manager.evaluate(
        RiskEvaluationInput(
            ticker="AAPL",
            fundamental=FundamentalAnalysisOutput(
                ticker="AAPL",
                sentiment_score=0.6,
                summary="Positive guidance",
                evidence=[],
            ),
            technical=_technical_output(TrendDirection.BULLISH, "positive"),
            portfolio=PortfolioState(cash_weight=0.34, positions=[], gross_exposure=0.65, max_drawdown_pct=0.05),
        )
    )

    assert output.decision == RiskDecision.FLAG
    assert output.violations
