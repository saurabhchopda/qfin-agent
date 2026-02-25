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
    Recommendation,
    RiskEvaluationInput,
    SupervisorOutput,
    SupervisorRequest,
    TechnicalAnalysisOutput,
    FundamentalAnalysisOutput,
    RiskManagerOutput,
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
