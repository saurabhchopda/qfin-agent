"""Agent B: technical analysis of OHLCV data."""

from __future__ import annotations

import pandas as pd

from qfin_agent.models.schemas import (
    TechnicalAnalysisInput,
    TechnicalAnalysisOutput,
    TechnicalIndicators,
    TrendDirection,
    VolatilityRegime,
)


class TechnicalAnalyst:
    """Computes trend, momentum, and volatility signals from OHLCV."""

    def analyze(self, data: TechnicalAnalysisInput) -> TechnicalAnalysisOutput:
        """Run technical analysis on historical bars.

        Args:
            data: Technical analysis input.

        Returns:
            Structured technical agent output.
        """

        if len(data.ohlcv) < 60:
            raise ValueError("At least 60 OHLCV bars are required for technical analysis.")

        df = pd.DataFrame([bar.model_dump() for bar in data.ohlcv])
        df = df.sort_values("timestamp").reset_index(drop=True)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        sma_fast = close.rolling(20).mean()
        sma_slow = close.rolling(50).mean()
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_14 = 100 - (100 / (1 + rs))

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + (2 * bb_std)
        bb_lower = bb_mid - (2 * bb_std)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        latest_close = float(close.iloc[-1])
        latest_sma_fast = float(sma_fast.iloc[-1])
        latest_sma_slow = float(sma_slow.iloc[-1])
        latest_ema_fast = float(ema_fast.iloc[-1])
        latest_ema_slow = float(ema_slow.iloc[-1])
        latest_rsi = float(rsi_14.iloc[-1])
        latest_macd = float(macd.iloc[-1])
        latest_macd_signal = float(macd_signal.iloc[-1])
        latest_bb_upper = float(bb_upper.iloc[-1])
        latest_bb_lower = float(bb_lower.iloc[-1])
        latest_atr = float(atr_14.iloc[-1])

        trend = self._trend_from_averages(latest_close, latest_sma_fast, latest_sma_slow, latest_ema_fast, latest_ema_slow)
        momentum = self._momentum_from_indicators(latest_rsi, latest_macd, latest_macd_signal)
        volatility_regime = self._volatility_from_atr(latest_atr, latest_close)
        confidence = self._confidence(trend, momentum, volatility_regime, latest_rsi)

        rationale = [
            f"Price {latest_close:.2f}, SMA20 {latest_sma_fast:.2f}, SMA50 {latest_sma_slow:.2f}",
            f"RSI14 {latest_rsi:.2f}, MACD {latest_macd:.4f} vs signal {latest_macd_signal:.4f}",
            f"ATR14 {latest_atr:.4f} implies {volatility_regime.value} volatility",
        ]

        return TechnicalAnalysisOutput(
            ticker=data.ticker,
            trend=trend,
            momentum=momentum,
            volatility_regime=volatility_regime,
            confidence=confidence,
            indicators=TechnicalIndicators(
                sma_fast=latest_sma_fast,
                sma_slow=latest_sma_slow,
                ema_fast=latest_ema_fast,
                ema_slow=latest_ema_slow,
                rsi_14=latest_rsi,
                macd=latest_macd,
                macd_signal=latest_macd_signal,
                bb_upper=latest_bb_upper,
                bb_lower=latest_bb_lower,
                atr_14=latest_atr,
            ),
            rationale=rationale,
        )

    @staticmethod
    def _trend_from_averages(
        close: float,
        sma_fast: float,
        sma_slow: float,
        ema_fast: float,
        ema_slow: float,
    ) -> TrendDirection:
        if close > sma_fast > sma_slow and ema_fast > ema_slow:
            return TrendDirection.BULLISH
        if close < sma_fast < sma_slow and ema_fast < ema_slow:
            return TrendDirection.BEARISH
        return TrendDirection.SIDEWAYS

    @staticmethod
    def _momentum_from_indicators(rsi_14: float, macd: float, macd_signal: float) -> str:
        if rsi_14 > 60 and macd > macd_signal:
            return "positive"
        if rsi_14 < 40 and macd < macd_signal:
            return "negative"
        return "neutral"

    @staticmethod
    def _volatility_from_atr(atr_14: float, close: float) -> VolatilityRegime:
        atr_ratio = atr_14 / max(close, 1e-9)
        if atr_ratio < 0.015:
            return VolatilityRegime.LOW
        if atr_ratio > 0.03:
            return VolatilityRegime.HIGH
        return VolatilityRegime.NORMAL

    @staticmethod
    def _confidence(
        trend: TrendDirection,
        momentum: str,
        volatility: VolatilityRegime,
        rsi_14: float,
    ) -> float:
        score = 0.5
        if trend != TrendDirection.SIDEWAYS:
            score += 0.2
        if momentum in {"positive", "negative"}:
            score += 0.15
        if volatility == VolatilityRegime.NORMAL:
            score += 0.1
        if rsi_14 < 20 or rsi_14 > 80:
            score -= 0.1
        return max(0.0, min(1.0, round(score, 3)))
