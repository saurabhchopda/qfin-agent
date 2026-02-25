"""Agent A: fundamental analysis using RAG + LLM summarization."""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from qfin_agent.config import Settings
from qfin_agent.models.schemas import EvidenceSnippet, FundamentalAnalysisInput, FundamentalAnalysisOutput
from qfin_agent.rag.retriever import NewsRetriever


class _FundamentalResponse(BaseModel):
    """Expected JSON response from LLM summarization."""

    summary: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)


class FundamentalAnalyst:
    """Produces sentiment and supporting evidence from retrieved documents."""

    def __init__(self, settings: Settings, retriever: NewsRetriever):
        self._settings = settings
        self._retriever = retriever

    async def analyze(self, data: FundamentalAnalysisInput) -> FundamentalAnalysisOutput:
        """Generate structured fundamental analysis output.

        Args:
            data: Fundamental analysis input.

        Returns:
            Fundamental analysis output.
        """

        docs = await self._retriever.retrieve(
            query=f"{data.ticker} {data.query}",
            top_k=data.top_k,
        )

        if not docs:
            return FundamentalAnalysisOutput(
                ticker=data.ticker,
                sentiment_score=0.0,
                summary="No retrievable fundamental evidence was found.",
                evidence=[],
            )

        evidence = [
            EvidenceSnippet(
                source=str(doc.metadata.get("source", "")),
                title=str(doc.metadata.get("title", "Untitled")),
                snippet=doc.page_content[:300],
                published_at=_parse_datetime(doc.metadata.get("published_at")),
            )
            for doc in docs[:5]
        ]

        if not self._settings.openai_api_key:
            return FundamentalAnalysisOutput(
                ticker=data.ticker,
                sentiment_score=0.0,
                summary="OPENAI_API_KEY is not configured; returning evidence-only neutral analysis.",
                evidence=evidence,
            )

        llm = ChatOpenAI(
            model=self._settings.agent_llm_model,
            api_key=self._settings.openai_api_key,
            temperature=0,
        )

        context = "\n\n".join(
            [f"Title: {ev.title}\nSnippet: {ev.snippet}\nSource: {ev.source}" for ev in evidence]
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a fundamental analyst. Read evidence and return JSON with keys "
                    "summary and sentiment_score in range [-1, 1]."
                )
            ),
            HumanMessage(content=f"Ticker: {data.ticker}\n\nEvidence:\n{context}"),
        ]
        response = await llm.ainvoke(messages)

        payload = _safe_json_parse(str(response.content))
        parsed = _FundamentalResponse.model_validate(payload)

        return FundamentalAnalysisOutput(
            ticker=data.ticker,
            sentiment_score=parsed.sentiment_score,
            summary=parsed.summary,
            evidence=evidence,
        )


def _safe_json_parse(raw: str) -> dict:
    """Parse JSON payload from an LLM response string."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text[:500], "sentiment_score": 0.0}


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None
