"""CLI entry point for qfin-agent workflow."""

from __future__ import annotations

import argparse
import asyncio
import json

import structlog

from qfin_agent.agents import FundamentalAnalyst, RiskManager, TechnicalAnalyst
from qfin_agent.backtest import (
    Backtester,
    BuyAndHoldEngine,
    LightweightSupervisorEngine,
    WalkForwardEvaluator,
)
from qfin_agent.config import configure_logging, get_settings
from qfin_agent.data import MarketDataClient
from qfin_agent.models.schemas import (
    BacktestConfig,
    FundamentalAnalysisInput,
    MarketDataRequest,
    PortfolioState,
    SupervisorRequest,
    TechnicalAnalysisInput,
    WalkForwardConfig,
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


async def run_backtest(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    lookback_bars: int = 60,
    position_size: float = 1.0,
    transaction_cost_bps: float = 2.0,
) -> dict:
    """Run lightweight historical simulation using supervisor-style recommendations."""
    settings = get_settings()

    market_data_client = MarketDataClient()
    bars = await market_data_client.fetch_ohlcv(
        MarketDataRequest(ticker=ticker, period=period, interval=interval)
    )
    if not bars:
        raise ValueError(f"No market data found for ticker={ticker}")

    engine = LightweightSupervisorEngine(
        technical_agent=TechnicalAnalyst(),
        risk_manager=RiskManager(settings=settings),
    )
    result = await Backtester().run(
        config=BacktestConfig(
            ticker=ticker,
            lookback_bars=lookback_bars,
            position_size=position_size,
            transaction_cost_bps=transaction_cost_bps,
        ),
        bars=bars,
        engine=engine,
    )
    return result.model_dump(mode="json")


async def run_walk_forward(
    ticker: str,
    period: str = "3y",
    interval: str = "1d",
    lookback_bars: int = 60,
    train_bars: int = 126,
    test_bars: int = 63,
    step_bars: int = 63,
    position_size: float = 1.0,
    transaction_cost_bps: float = 2.0,
) -> dict:
    """Run walk-forward evaluation against a buy-and-hold benchmark."""
    settings = get_settings()

    market_data_client = MarketDataClient()
    bars = await market_data_client.fetch_ohlcv(
        MarketDataRequest(ticker=ticker, period=period, interval=interval)
    )
    if not bars:
        raise ValueError(f"No market data found for ticker={ticker}")

    supervisor_engine = LightweightSupervisorEngine(
        technical_agent=TechnicalAnalyst(),
        risk_manager=RiskManager(settings=settings),
    )
    benchmark_engine = BuyAndHoldEngine()

    result = await WalkForwardEvaluator().evaluate(
        config=WalkForwardConfig(
            ticker=ticker,
            lookback_bars=lookback_bars,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            position_size=position_size,
            transaction_cost_bps=transaction_cost_bps,
        ),
        bars=bars,
        supervisor_engine=supervisor_engine,
        benchmark_engine=benchmark_engine,
    )
    return result.model_dump(mode="json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qfin-agent on a ticker symbol.")
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="6mo")
    parser.add_argument("--interval", type=str, default="1d")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--backtest", action="store_true", help="Run lightweight backtest mode.")
    mode_group.add_argument("--walk-forward", action="store_true", help="Run walk-forward evaluation mode.")
    parser.add_argument("--lookback-bars", type=int, default=60, help="Backtest indicator lookback window.")
    parser.add_argument("--position-size", type=float, default=1.0, help="Position size as portfolio fraction.")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=2.0,
        help="Transaction cost per unit turnover in basis points.",
    )
    parser.add_argument("--train-bars", type=int, default=126, help="Walk-forward training window in bars.")
    parser.add_argument("--test-bars", type=int, default=63, help="Walk-forward test window in bars.")
    parser.add_argument("--step-bars", type=int, default=63, help="Walk-forward step size in bars.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    if args.walk_forward:
        result = asyncio.run(
            run_walk_forward(
                ticker=args.ticker.upper(),
                period=args.period,
                interval=args.interval,
                lookback_bars=args.lookback_bars,
                train_bars=args.train_bars,
                test_bars=args.test_bars,
                step_bars=args.step_bars,
                position_size=args.position_size,
                transaction_cost_bps=args.transaction_cost_bps,
            )
        )
    elif args.backtest:
        result = asyncio.run(
            run_backtest(
                ticker=args.ticker.upper(),
                period=args.period,
                interval=args.interval,
                lookback_bars=args.lookback_bars,
                position_size=args.position_size,
                transaction_cost_bps=args.transaction_cost_bps,
            )
        )
    else:
        result = asyncio.run(run_workflow(args.ticker.upper(), args.period, args.interval))
    logger.info("workflow_result", result=json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
