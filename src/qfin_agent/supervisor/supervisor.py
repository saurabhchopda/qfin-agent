"""Supervisor orchestration powered by LangGraph."""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from qfin_agent.agents.fundamental import FundamentalAnalyst
from qfin_agent.agents.risk_manager import RiskManager
from qfin_agent.agents.technical import TechnicalAnalyst
from qfin_agent.config import Settings
from qfin_agent.models.schemas import (
    DebateConflict,
    DebateTranscript,
    DebateTurn,
    Recommendation,
    RiskDecision,
    RiskEvaluationInput,
    FundamentalAnalysisOutput,
    RiskManagerOutput,
    SupervisorOutput,
    SupervisorRequest,
    TechnicalAnalysisOutput,
    TrendDirection,
)


class WorkflowState(TypedDict):
    """Mutable state passed between graph nodes."""

    request: SupervisorRequest
    fundamental_output: FundamentalAnalysisOutput
    technical_output: TechnicalAnalysisOutput
    risk_output: RiskManagerOutput
    supervisor_output: SupervisorOutput


class SupervisorOrchestrator:
    """Build and execute the multi-agent workflow graph."""

    def __init__(
        self,
        settings: Settings,
        fundamental_agent: FundamentalAnalyst,
        technical_agent: TechnicalAnalyst,
        risk_manager: RiskManager,
    ):
        self._settings = settings
        self._fundamental_agent = fundamental_agent
        self._technical_agent = technical_agent
        self._risk_manager = risk_manager
        self._graph = self._build_graph()

    async def run(self, request: SupervisorRequest) -> SupervisorOutput:
        """Execute full supervisor workflow for one ticker."""
        result = await self._graph.ainvoke({"request": request})
        return result["supervisor_output"]

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("fundamental", self._run_fundamental)
        graph.add_node("technical", self._run_technical)
        graph.add_node("risk", self._run_risk)
        graph.add_node("supervisor", self._run_supervisor)

        graph.set_entry_point("fundamental")
        graph.add_edge("fundamental", "technical")
        graph.add_edge("technical", "risk")
        graph.add_edge("risk", "supervisor")
        graph.add_edge("supervisor", END)

        return graph.compile()

    async def _run_fundamental(self, state: WorkflowState) -> WorkflowState:
        output = await self._fundamental_agent.analyze(state["request"].fundamental_input)
        return {"fundamental_output": output}

    async def _run_technical(self, state: WorkflowState) -> WorkflowState:
        output = self._technical_agent.analyze(state["request"].technical_input)
        return {"technical_output": output}

    async def _run_risk(self, state: WorkflowState) -> WorkflowState:
        request = state["request"]
        risk_output = self._risk_manager.evaluate(
            RiskEvaluationInput(
                ticker=request.ticker,
                fundamental=state["fundamental_output"],
                technical=state["technical_output"],
                portfolio=request.portfolio,
            )
        )
        return {"risk_output": risk_output}

    async def _run_supervisor(self, state: WorkflowState) -> WorkflowState:
        request = state["request"]
        fundamental = state["fundamental_output"]
        technical = state["technical_output"]
        risk = state["risk_output"]

        if not self._settings.openai_api_key:
            recommendation = risk.recommendation
            confidence = round((technical.confidence + 0.55) / 2, 3)
            rationale = (
                "Fallback supervisor mode (no OPENAI_API_KEY). "
                f"Fundamental sentiment={fundamental.sentiment_score:.2f}; "
                f"technical trend={technical.trend.value}; "
                f"risk decision={risk.decision.value}."
            )
        else:
            recommendation, confidence, rationale = await self._llm_supervise(
                ticker=request.ticker,
                fundamental=fundamental,
                technical=technical,
                risk=risk,
            )

        output = SupervisorOutput(
            ticker=request.ticker,
            final_recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            debate_transcript=self._build_debate_transcript(
                fundamental=fundamental,
                technical=technical,
                risk=risk,
                final_recommendation=recommendation,
                final_confidence=confidence,
                final_rationale=rationale,
            ),
            fundamental=fundamental,
            technical=technical,
            risk=risk,
        )
        return {"supervisor_output": output}

    async def _llm_supervise(
        self,
        ticker: str,
        fundamental: FundamentalAnalysisOutput,
        technical: TechnicalAnalysisOutput,
        risk: RiskManagerOutput,
    ) -> tuple[Recommendation, float, str]:
        llm = ChatOpenAI(
            model=self._settings.supervisor_llm_model,
            api_key=self._settings.openai_api_key,
            temperature=0,
        )

        messages = [
            SystemMessage(
                content=(
                    "You are the supervisor for a trading research multi-agent system. "
                    "Return JSON with keys: final_recommendation (buy/sell/hold), confidence (0-1), rationale."
                )
            ),
            HumanMessage(
                content=(
                    f"Ticker: {ticker}\n"
                    f"Fundamental sentiment: {fundamental.sentiment_score}\n"
                    f"Fundamental summary: {fundamental.summary}\n"
                    f"Technical trend: {technical.trend.value}\n"
                    f"Technical momentum: {technical.momentum}\n"
                    f"Technical volatility: {technical.volatility_regime.value}\n"
                    f"Risk decision: {risk.decision.value}\n"
                    f"Risk recommendation: {risk.recommendation.value}\n"
                    f"Risk rationale: {risk.rationale}"
                )
            ),
        ]

        response = await llm.ainvoke(messages)
        payload = _safe_json_parse(str(response.content))

        recommendation_raw = str(payload.get("final_recommendation", risk.recommendation.value)).lower()
        recommendation = Recommendation(recommendation_raw)
        confidence = float(payload.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, round(confidence, 3)))
        rationale = str(payload.get("rationale", "Supervisor response did not include rationale."))

        return recommendation, confidence, rationale

    def _build_debate_transcript(
        self,
        fundamental: FundamentalAnalysisOutput,
        technical: TechnicalAnalysisOutput,
        risk: RiskManagerOutput,
        final_recommendation: Recommendation,
        final_confidence: float,
        final_rationale: str,
    ) -> DebateTranscript:
        fundamental_stance = self._recommendation_from_sentiment(fundamental.sentiment_score)
        technical_stance = self._recommendation_from_technical(
            technical.trend,
            technical.momentum,
        )

        turns = [
            DebateTurn(
                speaker="fundamental_analyst",
                stance=fundamental_stance,
                confidence=min(1.0, round(abs(fundamental.sentiment_score), 3)),
                thesis=fundamental.summary,
                key_points=[
                    f"Sentiment score={fundamental.sentiment_score:.2f}",
                    "Evidence selected from retrieved news and filings context.",
                ],
                citations=[f"{item.title} | {item.source}" for item in fundamental.evidence[:3]],
            ),
            DebateTurn(
                speaker="technical_analyst",
                stance=technical_stance,
                confidence=technical.confidence,
                thesis=(
                    f"Trend is {technical.trend.value}, momentum is {technical.momentum}, "
                    f"volatility regime is {technical.volatility_regime.value}."
                ),
                key_points=technical.rationale,
            ),
            DebateTurn(
                speaker="risk_manager",
                stance=risk.recommendation,
                confidence=self._risk_confidence(risk.decision),
                thesis=risk.rationale,
                key_points=[f"Risk decision={risk.decision.value}"],
                citations=[f"{item.rule}: {item.message}" for item in risk.violations],
            ),
            DebateTurn(
                speaker="supervisor",
                stance=final_recommendation,
                confidence=final_confidence,
                thesis=final_rationale,
                key_points=["Synthesized final action from all agent arguments."],
            ),
        ]

        conflicts: list[DebateConflict] = []
        if fundamental_stance != technical_stance:
            conflicts.append(
                DebateConflict(
                    topic="signal_divergence",
                    participants=["fundamental_analyst", "technical_analyst"],
                    description=(
                        "Fundamental and technical agents recommended different actions."
                    ),
                    resolution=f"Supervisor prioritized {final_recommendation.value}.",
                )
            )
        if risk.recommendation != final_recommendation:
            conflicts.append(
                DebateConflict(
                    topic="risk_override",
                    participants=["risk_manager", "supervisor"],
                    description="Supervisor output differs from risk manager recommendation.",
                    resolution=f"Final decision remains {final_recommendation.value}.",
                )
            )

        stances = [
            (fundamental_stance, turns[0].confidence),
            (technical_stance, turns[1].confidence),
            (risk.recommendation, turns[2].confidence),
            (final_recommendation, final_confidence),
        ]
        support = [confidence for stance, confidence in stances if stance == final_recommendation]
        consensus_strength = round(sum(support) / max(len(stances), 1), 3)

        return DebateTranscript(
            turns=turns,
            conflicts=conflicts,
            consensus_strength=max(0.0, min(1.0, consensus_strength)),
            final_resolution=(
                f"Final action {final_recommendation.value} at confidence {final_confidence:.2f}. "
                f"Risk decision was {risk.decision.value}."
            ),
        )

    @staticmethod
    def _recommendation_from_sentiment(sentiment: float) -> Recommendation:
        if sentiment > 0.2:
            return Recommendation.BUY
        if sentiment < -0.2:
            return Recommendation.SELL
        return Recommendation.HOLD

    @staticmethod
    def _recommendation_from_technical(trend: TrendDirection, momentum: str) -> Recommendation:
        if trend == TrendDirection.BULLISH and momentum == "positive":
            return Recommendation.BUY
        if trend == TrendDirection.BEARISH and momentum == "negative":
            return Recommendation.SELL
        return Recommendation.HOLD

    @staticmethod
    def _risk_confidence(decision: RiskDecision) -> float:
        if decision == RiskDecision.APPROVE:
            return 0.8
        if decision == RiskDecision.FLAG:
            return 0.7
        return 0.9


def _safe_json_parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
