"""CLI entry point for qfin-agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import json

import structlog

from qfin_agent.agents import FundamentalAnalyst, RiskManager, TechnicalAnalyst
from qfin_agent.config import configure_logging, get_settings
from qfin_agent.data import MarketDataClient
from qfin_agent.models.schemas import (
    FundamentalAnalysisInput,
    MarketDataRequest,
    PortfolioState,
    SupervisorRequest,
    TechnicalAnalysisInput,
)
from qfin_agent.rag import NewsRetriever, build_embeddings, ingest_news_documents
from qfin_agent.supervisor import SupervisorOrchestrator

logger = structlog.get_logger(__name__)


async def run_workflow(ticker: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Run end-to-end multi-agent workflow for a ticker."""
    settings = get_settings()

    market_data_client = MarketDataClient()
    bars = await market_data_client.fetch_ohlcv(
        MarketDataRequest(ticker=ticker, period=period, interval=interval)
    )
    if not bars:
        raise ValueError(f"No market data found for ticker={ticker}")

    documents = await ingest_news_documents(ticker=ticker, settings=settings)
    if not documents:
        raise ValueError("No news documents fetched for fundamental analysis.")

    embeddings = build_embeddings(settings)
    retriever = NewsRetriever.from_documents(documents=documents, embeddings=embeddings)

    orchestrator = SupervisorOrchestrator(
        settings=settings,
        fundamental_agent=FundamentalAnalyst(settings=settings, retriever=retriever),
        technical_agent=TechnicalAnalyst(),
        risk_manager=RiskManager(settings=settings),
    )

    response = await orchestrator.run(
        SupervisorRequest(
            ticker=ticker,
            technical_input=TechnicalAnalysisInput(ticker=ticker, ohlcv=bars),
            fundamental_input=FundamentalAnalysisInput(ticker=ticker),
            portfolio=PortfolioState(cash_weight=0.35, positions=[], gross_exposure=0.65, max_drawdown_pct=0.08),
        )
    )
    return response.model_dump(mode="json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qfin-agent on a ticker symbol.")
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="6mo")
    parser.add_argument("--interval", type=str, default="1d")
    return parser


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    result = asyncio.run(run_workflow(args.ticker.upper(), args.period, args.interval))
    logger.info("workflow_result", result=json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
