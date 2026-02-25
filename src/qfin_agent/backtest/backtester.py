"""Lightweight backtesting for supervisor recommendation engines."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from qfin_agent.models.schemas import (
    BacktestConfig,
    BacktestResult,
    BacktestSignal,
    BacktestStep,
    OHLCVBar,
    PortfolioPosition,
    PortfolioState,
    Recommendation,
)


class RecommendationEngine(Protocol):
    """Interface for engines that emit supervisor-compatible signals."""

    async def recommend(
        self,
        ticker: str,
        ohlcv_window: list[OHLCVBar],
        portfolio: PortfolioState,
    ) -> BacktestSignal:
        """Generate a recommendation for the next bar."""


class Backtester:
    """Runs a single-asset bar-by-bar strategy simulation."""

    async def run(
        self,
        config: BacktestConfig,
        bars: list[OHLCVBar],
        engine: RecommendationEngine,
    ) -> BacktestResult:
        """Simulate strategy returns from recommendation signals.

        Args:
            config: Backtest parameters.
            bars: Historical bars sorted by timestamp.
            engine: Recommendation engine (typically supervisor-backed).

        Returns:
            Aggregate backtest result and per-step records.
        """

        if len(bars) <= config.lookback_bars:
            raise ValueError("Not enough bars to run backtest with the configured lookback.")

        ordered_bars = sorted(bars, key=lambda bar: bar.timestamp)

        equity = config.initial_capital
        peak_equity = equity
        max_drawdown = 0.0
        current_position = 0.0
        trades = 0
        strategy_returns: list[float] = []
        steps: list[BacktestStep] = []

        portfolio = PortfolioState(cash_weight=1.0, positions=[], gross_exposure=0.0, max_drawdown_pct=0.0)

        for idx in range(config.lookback_bars, len(ordered_bars)):
            prev_close = ordered_bars[idx - 1].close
            current_bar = ordered_bars[idx]
            market_return = (current_bar.close / prev_close) - 1.0

            signal = await engine.recommend(
                ticker=config.ticker,
                ohlcv_window=ordered_bars[idx - config.lookback_bars : idx],
                portfolio=portfolio,
            )

            target_position = _recommendation_to_position(signal.recommendation, config.position_size)
            turnover = abs(target_position - current_position)
            transaction_cost = turnover * (config.transaction_cost_bps / 10_000)
            if turnover > 0:
                trades += 1

            strategy_return = (target_position * market_return) - transaction_cost
            strategy_returns.append(strategy_return)

            equity *= 1.0 + strategy_return
            peak_equity = max(peak_equity, equity)
            drawdown = 1.0 - (equity / peak_equity)
            max_drawdown = max(max_drawdown, drawdown)

            current_position = target_position
            portfolio = PortfolioState(
                cash_weight=max(0.0, 1.0 - abs(current_position)),
                positions=(
                    [PortfolioPosition(ticker=config.ticker, weight=abs(current_position))]
                    if abs(current_position) > 0
                    else []
                ),
                gross_exposure=abs(current_position),
                max_drawdown_pct=max_drawdown,
            )

            steps.append(
                BacktestStep(
                    timestamp=current_bar.timestamp,
                    close=current_bar.close,
                    market_return=market_return,
                    strategy_return=strategy_return,
                    position=current_position,
                    equity=equity,
                    recommendation=signal.recommendation,
                    confidence=signal.confidence,
                )
            )

        total_return = (equity / config.initial_capital) - 1.0
        annualized_return, annualized_volatility, sharpe_ratio, win_rate = _compute_metrics(strategy_returns, total_return)

        return BacktestResult(
            config=config,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=trades,
            steps=steps,
        )


class BuyAndHoldEngine:
    """Baseline recommendation engine that always emits BUY."""

    async def recommend(
        self,
        ticker: str,
        ohlcv_window: list[OHLCVBar],
        portfolio: PortfolioState,
    ) -> BacktestSignal:
        return BacktestSignal(
            recommendation=Recommendation.BUY,
            confidence=1.0,
            rationale="Baseline buy-and-hold signal.",
        )


def _recommendation_to_position(recommendation: Recommendation, position_size: float) -> float:
    if recommendation == Recommendation.BUY:
        return position_size
    if recommendation == Recommendation.SELL:
        return -position_size
    return 0.0


def _compute_metrics(strategy_returns: list[float], total_return: float) -> tuple[float, float, float, float]:
    if not strategy_returns:
        return 0.0, 0.0, 0.0, 0.0

    n = len(strategy_returns)
    years = n / 252
    annualized_return = ((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    returns_arr = np.asarray(strategy_returns, dtype=float)
    daily_mean = float(np.mean(returns_arr))
    daily_vol = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0

    annualized_volatility = daily_vol * math.sqrt(252)
    sharpe_ratio = (daily_mean / daily_vol) * math.sqrt(252) if daily_vol > 0 else 0.0
    wins = int(np.sum(returns_arr > 0))
    win_rate = wins / n

    return annualized_return, annualized_volatility, sharpe_ratio, win_rate
