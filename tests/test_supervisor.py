from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from langchain_core.documents import Document

from qfin_agent.agents.fundamental import FundamentalAnalyst
from qfin_agent.agents.risk_manager import RiskManager
from qfin_agent.agents.technical import TechnicalAnalyst
from qfin_agent.config import Settings
from qfin_agent.models.schemas import (
    FundamentalAnalysisInput,
    OHLCVBar,
    PortfolioState,
    SupervisorRequest,
    TechnicalAnalysisInput,
)
from qfin_agent.supervisor.supervisor import SupervisorOrchestrator


class _FakeRetriever:
    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return [
            Document(
                page_content="Revenue growth is stable with improving margins.",
                metadata={"source": "https://example.com/news", "title": "Stable growth"},
            )
        ]


def _bars(count: int = 80) -> list[OHLCVBar]:
    start = datetime(2025, 1, 1)
    bars: list[OHLCVBar] = []
    close = 100.0
    for i in range(count):
        close += 0.2
        bars.append(
            OHLCVBar(
                timestamp=start + timedelta(days=i),
                open=close - 0.5,
                high=close + 0.7,
                low=close - 0.8,
                close=close,
                volume=1_000_000,
            )
        )
    return bars


@pytest.mark.asyncio
async def test_supervisor_orchestrator_runs_without_openai() -> None:
    settings = Settings(openai_api_key=None)
    orchestrator = SupervisorOrchestrator(
        settings=settings,
        fundamental_agent=FundamentalAnalyst(settings=settings, retriever=_FakeRetriever()),
        technical_agent=TechnicalAnalyst(),
        risk_manager=RiskManager(settings=settings),
    )

    result = await orchestrator.run(
        SupervisorRequest(
            ticker="AAPL",
            technical_input=TechnicalAnalysisInput(ticker="AAPL", ohlcv=_bars()),
            fundamental_input=FundamentalAnalysisInput(ticker="AAPL"),
            portfolio=PortfolioState(cash_weight=0.3, positions=[], gross_exposure=0.6, max_drawdown_pct=0.05),
        )
    )

    assert result.ticker == "AAPL"
    assert result.final_recommendation in {"buy", "sell", "hold"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.debate_transcript.turns
    assert result.debate_transcript.turns[-1].speaker == "supervisor"
    assert 0.0 <= result.debate_transcript.consensus_strength <= 1.0
