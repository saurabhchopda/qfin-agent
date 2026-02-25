from __future__ import annotations

from datetime import datetime, timedelta

from qfin_agent.agents.technical import TechnicalAnalyst
from qfin_agent.models.schemas import OHLCVBar, TechnicalAnalysisInput, TrendDirection


def _build_bars(count: int = 80) -> list[OHLCVBar]:
    start = datetime(2025, 1, 1)
    bars: list[OHLCVBar] = []
    price = 100.0
    for i in range(count):
        price += 0.4
        bars.append(
            OHLCVBar(
                timestamp=start + timedelta(days=i),
                open=price - 0.3,
                high=price + 0.6,
                low=price - 0.8,
                close=price,
                volume=1_000_000 + i * 1000,
            )
        )
    return bars


def test_technical_analyst_returns_structured_output() -> None:
    analyst = TechnicalAnalyst()
    output = analyst.analyze(TechnicalAnalysisInput(ticker="AAPL", ohlcv=_build_bars()))

    assert output.ticker == "AAPL"
    assert output.confidence >= 0.0
    assert output.confidence <= 1.0
    assert output.trend in {TrendDirection.BULLISH, TrendDirection.SIDEWAYS, TrendDirection.BEARISH}
    assert len(output.rationale) >= 1
