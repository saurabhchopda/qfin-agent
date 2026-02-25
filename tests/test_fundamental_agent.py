from __future__ import annotations

from langchain_core.documents import Document

from qfin_agent.agents.fundamental import FundamentalAnalyst
from qfin_agent.config import Settings
from qfin_agent.models.schemas import FundamentalAnalysisInput
from qfin_agent.rag.retriever import NewsRetriever


class _FakeRetriever(NewsRetriever):
    def __init__(self) -> None:
        self._docs = [
            Document(
                page_content="Earnings beat expectations and raised forward guidance.",
                metadata={"source": "https://example.com", "title": "Strong quarter"},
            )
        ]

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self._docs[:top_k]


def test_fundamental_agent_returns_evidence_without_openai_key() -> None:
    analyst = FundamentalAnalyst(settings=Settings(openai_api_key=None), retriever=_FakeRetriever())
    output = __import__("asyncio").run(analyst.analyze(FundamentalAnalysisInput(ticker="AAPL")))

    assert output.ticker == "AAPL"
    assert output.evidence
    assert output.sentiment_score == 0.0
