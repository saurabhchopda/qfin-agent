"""Walk-forward evaluation for supervisor recommendations."""

from __future__ import annotations

import math

import numpy as np

from qfin_agent.backtest.backtester import Backtester, BuyAndHoldEngine, RecommendationEngine
from qfin_agent.models.schemas import (
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    OHLCVBar,
    RegimeAggregate,
    WalkForwardAggregate,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindowResult,
)


class WalkForwardEvaluator:
    """Evaluate recommendation engines across rolling train/test windows."""

    def __init__(self, backtester: Backtester | None = None):
        self._backtester = backtester or Backtester()

    async def evaluate(
        self,
        config: WalkForwardConfig,
        bars: list[OHLCVBar],
        supervisor_engine: RecommendationEngine,
        benchmark_engine: RecommendationEngine | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward evaluation.

        Args:
            config: Walk-forward settings.
            bars: Historical OHLCV bars sorted by timestamp.
            supervisor_engine: Strategy under evaluation.
            benchmark_engine: Optional benchmark strategy. Defaults to buy-and-hold.

        Returns:
            Detailed per-window and aggregate walk-forward metrics.
        """

        if config.train_bars < config.lookback_bars:
            raise ValueError("train_bars must be greater than or equal to lookback_bars.")

        minimum_bars = config.train_bars + config.test_bars
        if len(bars) < minimum_bars:
            raise ValueError(
                "Not enough bars for walk-forward evaluation with the current train/test settings."
            )

        ordered_bars = sorted(bars, key=lambda bar: bar.timestamp)
        baseline_engine = benchmark_engine or BuyAndHoldEngine()

        windows: list[WalkForwardWindowResult] = []
        latest_start = len(ordered_bars) - (config.train_bars + config.test_bars)

        for window_index, start_idx in enumerate(
            range(0, latest_start + 1, config.step_bars),
            start=1,
        ):
            train_start_idx = start_idx
            train_end_idx = train_start_idx + config.train_bars
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + config.test_bars
            if test_end_idx > len(ordered_bars):
                break

            eval_slice_start = test_start_idx - config.lookback_bars
            eval_bars = ordered_bars[eval_slice_start:test_end_idx]
            test_bars = ordered_bars[test_start_idx:test_end_idx]

            backtest_config = BacktestConfig(
                ticker=config.ticker,
                lookback_bars=config.lookback_bars,
                initial_capital=config.initial_capital,
                position_size=config.position_size,
                transaction_cost_bps=config.transaction_cost_bps,
            )

            supervisor_result = await self._backtester.run(
                config=backtest_config,
                bars=eval_bars,
                engine=supervisor_engine,
            )
            benchmark_result = await self._backtester.run(
                config=backtest_config,
                bars=eval_bars,
                engine=baseline_engine,
            )

            supervisor_summary = _summarize_backtest(
                result=supervisor_result,
                initial_capital=config.initial_capital,
            )
            benchmark_summary = _summarize_backtest(
                result=benchmark_result,
                initial_capital=config.initial_capital,
            )
            excess_return = supervisor_summary.total_return - benchmark_summary.total_return

            windows.append(
                WalkForwardWindowResult(
                    window_index=window_index,
                    train_start=ordered_bars[train_start_idx].timestamp,
                    train_end=ordered_bars[train_end_idx - 1].timestamp,
                    test_start=ordered_bars[test_start_idx].timestamp,
                    test_end=ordered_bars[test_end_idx - 1].timestamp,
                    regime=_classify_regime(test_bars),
                    supervisor=supervisor_summary,
                    benchmark=benchmark_summary,
                    excess_return=excess_return,
                )
            )

        if not windows:
            raise ValueError("No walk-forward windows were generated with the provided settings.")

        return WalkForwardResult(
            config=config,
            windows=windows,
            aggregate=_aggregate_windows(windows),
        )


def _summarize_backtest(result: BacktestResult, initial_capital: float) -> BacktestSummary:
    """Create compact metrics from a detailed backtest result."""
    final_equity = result.steps[-1].equity if result.steps else initial_capital
    return BacktestSummary(
        total_return=result.total_return,
        annualized_return=result.annualized_return,
        annualized_volatility=result.annualized_volatility,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        trades=result.trades,
        final_equity=final_equity,
    )


def _classify_regime(test_bars: list[OHLCVBar]) -> str:
    """Classify market regime from test-window price action."""
    if len(test_bars) < 2:
        return "unknown"

    closes = np.asarray([bar.close for bar in test_bars], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    total_return = float((closes[-1] / closes[0]) - 1.0)
    annualized_volatility = float(np.std(returns, ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0

    if annualized_volatility >= 0.35:
        return "high_volatility"
    if total_return >= 0.05:
        return "bull"
    if total_return <= -0.05:
        return "bear"
    return "sideways"


def _aggregate_windows(windows: list[WalkForwardWindowResult]) -> WalkForwardAggregate:
    """Aggregate performance across all walk-forward windows."""
    total_windows = len(windows)
    supervisor_returns = [window.supervisor.total_return for window in windows]
    benchmark_returns = [window.benchmark.total_return for window in windows]
    excess_returns = [window.excess_return for window in windows]

    outperformed = sum(1 for value in excess_returns if value > 0)

    by_regime: dict[str, list[float]] = {}
    for window in windows:
        by_regime.setdefault(window.regime, []).append(window.excess_return)

    regime_aggregates: list[RegimeAggregate] = []
    for regime, values in sorted(by_regime.items()):
        regime_aggregates.append(
            RegimeAggregate(
                regime=regime,
                windows=len(values),
                avg_excess_return=float(np.mean(values)) if values else 0.0,
                outperformance_rate=(sum(1 for value in values if value > 0) / len(values)) if values else 0.0,
            )
        )

    return WalkForwardAggregate(
        total_windows=total_windows,
        supervisor_avg_return=float(np.mean(supervisor_returns)) if supervisor_returns else 0.0,
        benchmark_avg_return=float(np.mean(benchmark_returns)) if benchmark_returns else 0.0,
        avg_excess_return=float(np.mean(excess_returns)) if excess_returns else 0.0,
        outperformance_rate=(outperformed / total_windows) if total_windows else 0.0,
        regimes=regime_aggregates,
    )
