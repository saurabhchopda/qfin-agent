"""Market data access layer backed by yfinance."""

from __future__ import annotations

import asyncio

import pandas as pd
import structlog
import yfinance as yf

from qfin_agent.models.schemas import MarketDataRequest, OHLCVBar

logger = structlog.get_logger(__name__)


class MarketDataClient:
    """Wrapper around yfinance to fetch OHLCV bars."""

    async def fetch_ohlcv(self, request: MarketDataRequest) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a ticker asynchronously.

        Args:
            request: Market data request.

        Returns:
            Historical OHLCV bars sorted by timestamp.
        """

        logger.info(
            "fetching_ohlcv",
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
        )
        df = await asyncio.to_thread(self._download_history, request)
        if df.empty:
            return []

        bars: list[OHLCVBar] = []
        for ts, row in df.iterrows():
            bars.append(
                OHLCVBar(
                    timestamp=pd.Timestamp(ts).to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )
        return bars

    @staticmethod
    def _download_history(request: MarketDataRequest) -> pd.DataFrame:
        ticker = yf.Ticker(request.ticker)
        history = ticker.history(period=request.period, interval=request.interval, auto_adjust=False)
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = [col[0] for col in history.columns]
        return history.dropna()
