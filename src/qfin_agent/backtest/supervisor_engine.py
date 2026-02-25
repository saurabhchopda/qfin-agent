"""Recommendation engines backed by qfin-agent supervisor components."""

from __future__ import annotations

from qfin_agent.agents.risk_manager import RiskManager
from qfin_agent.agents.technical import TechnicalAnalyst
from qfin_agent.models.schemas import (
    BacktestSignal,
    FundamentalAnalysisOutput,
    OHLCVBar,
    PortfolioState,
    RiskEvaluationInput,
    TechnicalAnalysisInput,
    TrendDirection,
)


class LightweightSupervisorEngine:
    """Fast supervisor-style engine using technical + risk manager signals.

    This avoids per-bar LLM calls while preserving recommendation flow:
    technical analysis -> risk validation -> recommendation.
    """

    def __init__(self, technical_agent: TechnicalAnalyst, risk_manager: RiskManager):
        self._technical_agent = technical_agent
        self._risk_manager = risk_manager

    async def recommend(
        self,
        ticker: str,
        ohlcv_window: list[OHLCVBar],
        portfolio: PortfolioState,
    ) -> BacktestSignal:
        technical = self._technical_agent.analyze(
            TechnicalAnalysisInput(ticker=ticker, ohlcv=ohlcv_window)
        )
        fundamental = _heuristic_fundamental_from_technical(ticker=ticker, trend=technical.trend)
        risk = self._risk_manager.evaluate(
            RiskEvaluationInput(
                ticker=ticker,
                fundamental=fundamental,
                technical=technical,
                portfolio=portfolio,
            )
        )

        return BacktestSignal(
            recommendation=risk.recommendation,
            confidence=technical.confidence,
            rationale=risk.rationale,
        )


def _heuristic_fundamental_from_technical(ticker: str, trend: TrendDirection) -> FundamentalAnalysisOutput:
    if trend == TrendDirection.BULLISH:
        score = 0.35
    elif trend == TrendDirection.BEARISH:
        score = -0.35
    else:
        score = 0.0

    return FundamentalAnalysisOutput(
        ticker=ticker,
        sentiment_score=score,
        summary="Heuristic sentiment proxy for lightweight backtesting.",
        evidence=[],
    )
