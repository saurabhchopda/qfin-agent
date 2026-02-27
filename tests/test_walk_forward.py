from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from qfin_agent.backtest import BuyAndHoldEngine, WalkForwardEvaluator
from qfin_agent.models.schemas import OHLCVBar, WalkForwardConfig


def _bars(count: int = 260, slope: float = 0.2) -> list[OHLCVBar]:
    start = datetime(2023, 1, 1)
    price = 100.0
    rows: list[OHLCVBar] = []
    for i in range(count):
        price += slope
        rows.append(
            OHLCVBar(
                timestamp=start + timedelta(days=i),
                open=price - 0.3,
                high=price + 0.6,
                low=price - 0.7,
                close=price,
                volume=1_000_000 + i * 100,
            )
        )
    return rows


@pytest.mark.asyncio
async def test_walk_forward_evaluator_generates_windows_and_aggregate() -> None:
    bars = _bars()
    evaluator = WalkForwardEvaluator()

    result = await evaluator.evaluate(
        config=WalkForwardConfig(
            ticker="AAPL",
            lookback_bars=30,
            train_bars=90,
            test_bars=30,
            step_bars=30,
        ),
        bars=bars,
        supervisor_engine=BuyAndHoldEngine(),
        benchmark_engine=BuyAndHoldEngine(),
    )

    assert len(result.windows) == 5
    assert result.aggregate.total_windows == 5
    assert abs(result.aggregate.avg_excess_return) < 1e-12
    assert result.aggregate.outperformance_rate == 0.0


@pytest.mark.asyncio
async def test_walk_forward_evaluator_raises_on_insufficient_data() -> None:
    bars = _bars(count=80)
    evaluator = WalkForwardEvaluator()

    with pytest.raises(ValueError, match="Not enough bars"):
        await evaluator.evaluate(
            config=WalkForwardConfig(
                ticker="AAPL",
                lookback_bars=30,
                train_bars=60,
                test_bars=40,
                step_bars=20,
            ),
            bars=bars,
            supervisor_engine=BuyAndHoldEngine(),
            benchmark_engine=BuyAndHoldEngine(),
        )
