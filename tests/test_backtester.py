from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from qfin_agent.agents.risk_manager import RiskManager
from qfin_agent.agents.technical import TechnicalAnalyst
from qfin_agent.backtest import Backtester, BuyAndHoldEngine, LightweightSupervisorEngine
from qfin_agent.config import Settings
from qfin_agent.models.schemas import BacktestConfig, OHLCVBar, Recommendation


def _bars(count: int = 140, slope: float = 0.25) -> list[OHLCVBar]:
    start = datetime(2024, 1, 1)
    price = 100.0
    rows: list[OHLCVBar] = []
    for i in range(count):
        price += slope
        rows.append(
            OHLCVBar(
                timestamp=start + timedelta(days=i),
                open=price - 0.2,
                high=price + 0.4,
                low=price - 0.5,
                close=price,
                volume=1_000_000 + i * 500,
            )
        )
    return rows


@pytest.mark.asyncio
async def test_backtester_buy_and_hold_positive_in_uptrend() -> None:
    bars = _bars()
    result = await Backtester().run(
        config=BacktestConfig(ticker="AAPL", lookback_bars=60, initial_capital=100_000),
        bars=bars,
        engine=BuyAndHoldEngine(),
    )

    assert result.total_return > 0
    assert result.trades == 1
    assert len(result.steps) == len(bars) - 60


@pytest.mark.asyncio
async def test_lightweight_supervisor_engine_emits_valid_signals() -> None:
    bars = _bars(slope=0.1)
    engine = LightweightSupervisorEngine(
        technical_agent=TechnicalAnalyst(),
        risk_manager=RiskManager(settings=Settings(openai_api_key=None)),
    )

    result = await Backtester().run(
        config=BacktestConfig(ticker="MSFT", lookback_bars=60, position_size=0.5),
        bars=bars,
        engine=engine,
    )

    assert result.steps
    assert result.steps[-1].recommendation in {
        Recommendation.BUY,
        Recommendation.SELL,
        Recommendation.HOLD,
    }
    assert 0.0 <= result.steps[-1].confidence <= 1.0
