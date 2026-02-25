"""Agent C: risk policy checks over agent proposals."""

from __future__ import annotations

from qfin_agent.config import Settings
from qfin_agent.models.schemas import (
    Recommendation,
    RiskDecision,
    RiskEvaluationInput,
    RiskManagerOutput,
    RiskViolation,
    TrendDirection,
)


class RiskManager:
    """Validates trading intent against portfolio risk constraints."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def evaluate(self, data: RiskEvaluationInput) -> RiskManagerOutput:
        """Assess recommendation viability under configured constraints.

        Args:
            data: Combined outputs from fundamental and technical agents.

        Returns:
            Risk manager decision and rationale.
        """

        recommendation = self._derive_recommendation(
            sentiment=data.fundamental.sentiment_score,
            trend=data.technical.trend,
            momentum=data.technical.momentum,
        )

        target_weight = self._target_weight_for_recommendation(recommendation)
        projected_gross = data.portfolio.gross_exposure + target_weight

        violations: list[RiskViolation] = []

        if target_weight > self._settings.max_position_weight:
            violations.append(
                RiskViolation(
                    rule="max_position_weight",
                    message=(
                        f"Proposed weight {target_weight:.2%} exceeds "
                        f"limit {self._settings.max_position_weight:.2%}."
                    ),
                    severity="high",
                )
            )

        if projected_gross > self._settings.max_gross_exposure:
            violations.append(
                RiskViolation(
                    rule="max_gross_exposure",
                    message=(
                        f"Projected gross exposure {projected_gross:.2f} exceeds "
                        f"limit {self._settings.max_gross_exposure:.2f}."
                    ),
                    severity="high",
                )
            )

        if data.portfolio.max_drawdown_pct > self._settings.max_drawdown_pct:
            violations.append(
                RiskViolation(
                    rule="max_drawdown_pct",
                    message=(
                        f"Observed drawdown {data.portfolio.max_drawdown_pct:.2%} exceeds "
                        f"limit {self._settings.max_drawdown_pct:.2%}."
                    ),
                    severity="critical",
                )
            )

        decision = self._decision_from_violations(violations)
        if decision == RiskDecision.REJECT:
            recommendation = Recommendation.HOLD
            target_weight = 0.0

        rationale = self._build_rationale(recommendation, data, violations)

        return RiskManagerOutput(
            ticker=data.ticker,
            decision=decision,
            recommendation=recommendation,
            target_position_weight=round(target_weight, 4),
            violations=violations,
            rationale=rationale,
        )

    @staticmethod
    def _derive_recommendation(sentiment: float, trend: TrendDirection, momentum: str) -> Recommendation:
        bullish = trend == TrendDirection.BULLISH and momentum == "positive" and sentiment > 0.2
        bearish = trend == TrendDirection.BEARISH and momentum == "negative" and sentiment < -0.2
        if bullish:
            return Recommendation.BUY
        if bearish:
            return Recommendation.SELL
        return Recommendation.HOLD

    def _target_weight_for_recommendation(self, recommendation: Recommendation) -> float:
        if recommendation in {Recommendation.BUY, Recommendation.SELL}:
            return self._settings.default_target_weight
        return 0.0

    @staticmethod
    def _decision_from_violations(violations: list[RiskViolation]) -> RiskDecision:
        if any(v.severity == "critical" for v in violations):
            return RiskDecision.REJECT
        if violations:
            return RiskDecision.FLAG
        return RiskDecision.APPROVE

    @staticmethod
    def _build_rationale(
        recommendation: Recommendation,
        data: RiskEvaluationInput,
        violations: list[RiskViolation],
    ) -> str:
        base = (
            f"Fundamental sentiment={data.fundamental.sentiment_score:.2f}, "
            f"technical trend={data.technical.trend.value}, "
            f"momentum={data.technical.momentum}. "
            f"Risk recommendation={recommendation.value}."
        )
        if not violations:
            return base + " No risk limits were violated."
        joined = " ".join([f"[{v.rule}] {v.message}" for v in violations])
        return f"{base} Violations: {joined}"
