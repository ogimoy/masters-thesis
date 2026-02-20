# src/run_baselines.py
"""
Run baseline hedging strategies on validation and test periods and save results.

This script runs *non-learning* baselines (fixed rules) separately on:
- validation period (default 2024)
- test/evaluation period (default 2025-01-01..2025-08-31)

Outputs:
- baseline_validation_results.csv
- baseline_test_results.csv
- Optional per-episode step logs in baseline_logs/<period>/<baseline_name>_episode_<k>.csv

Baselines supported (from src.baselines.hedging_baselines):
- FixedFrequencyHedgingBaseline
- ThresholdHedgingBaseline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.hedging_baselines import (
    FixedFrequencyHedgingBaseline,
    ThresholdHedgingBaseline,
)
from src.envs.baseline_env import UniswapEnvBaselines


# ----------------------------
# Metrics
# ----------------------------

def compute_sortino(step_returns, target: float = 0.0) -> float:
    step_returns = np.asarray(step_returns, dtype=float)
    downside = step_returns[step_returns < target]
    if downside.size == 0:
        return float("inf")
    expected_return = float(np.mean(step_returns))
    downside_std = float(np.std(downside))
    return float((expected_return - target) / downside_std) if downside_std != 0 else float("inf")


def max_drawdown(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak != 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def std_of_returns(step_returns) -> float:
    step_returns = np.asarray(step_returns, dtype=float)
    if step_returns.size <= 1:
        return 0.0
    return float(np.std(step_returns, ddof=1))


# ----------------------------
# Data loading
# ----------------------------

def load_price_data_slice(data_dir: Path, start: str, end: str) -> dict:
    ethusdt_df = pd.read_csv(data_dir / "ethusdt_15min.csv", parse_dates=["timestamp"])
    btcusdt_df = pd.read_csv(data_dir / "btcusdt_15min.csv", parse_dates=["timestamp"])
    ethbtc_df = pd.read_csv(data_dir / "ethbtc_15min.csv", parse_dates=["timestamp"])
    eth_funding_df = pd.read_csv(data_dir / "eth_funding_15min.csv", parse_dates=["timestamp"])
    btc_funding_df = pd.read_csv(data_dir / "btc_funding_15min.csv", parse_dates=["timestamp"])

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    ethusdt = ethusdt_df[(ethusdt_df["timestamp"] >= start_ts) & (ethusdt_df["timestamp"] <= end_ts)]
    btcusdt = btcusdt_df[(btcusdt_df["timestamp"] >= start_ts) & (btcusdt_df["timestamp"] <= end_ts)]
    ethbtc = ethbtc_df[(ethbtc_df["timestamp"] >= start_ts) & (ethbtc_df["timestamp"] <= end_ts)]
    eth_funding = eth_funding_df[(eth_funding_df["timestamp"] >= start_ts) & (eth_funding_df["timestamp"] <= end_ts)]
    btc_funding = btc_funding_df[(btc_funding_df["timestamp"] >= start_ts) & (btc_funding_df["timestamp"] <= end_ts)]

    return {
        "ethusdt": ethusdt["close"].values,
        "btcusdt": btcusdt["close"].values,
        "ethbtc": ethbtc["close"].values,
        "eth_funding": eth_funding["fundingRate"].fillna(0).values,
        "btc_funding": btc_funding["fundingRate"].fillna(0).values,
        "timestamp": btc_funding["timestamp"].values,
    }


# ----------------------------
# Running + saving
# ----------------------------

def run_one_baseline(
    baseline_name: str,
    baseline_obj,
    n_episodes: int,
    episode_length: int,
    seeds: list[int],
    log_dir: Path | None,
) -> dict:
    episode_returns = []
    step_returns_all = []
    mdds = []

    for ep_idx, seed in enumerate(seeds):
        # Run baseline episode
        values = baseline_obj.run_episode(seed=seed, episode_length=episode_length)
        values = np.asarray(values, dtype=float)

        if values.size < 2 or values[0] == 0:
            ep_return = 0.0
            step_returns = np.asarray([0.0])
        else:
            ep_return = float(values[-1] / values[0] - 1.0)
            step_returns = values[1:] / values[:-1] - 1.0

        episode_returns.append(ep_return)
        step_returns_all.append(step_returns)
        mdds.append(max_drawdown(values))

        # Optional per-step logs
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            # baseline history is list of dataclass entries
            rows = []
            for h in baseline_obj.history:
                rows.append(
                    {
                        "episode": ep_idx + 1,
                        "step": h.step,
                        "reward": h.reward,
                        "portfolio_value": h.portfolio_value,
                        "hedge_token1_pct": h.hedge_token1_pct,
                        "hedge_token2_pct": h.hedge_token2_pct,
                        "hedge_token1_amt": h.hedge_token1_amt,
                        "hedge_token2_amt": h.hedge_token2_amt,
                        "transaction_cost": h.transaction_cost,
                        "rebalanced": h.rebalanced,
                    }
                )
            pd.DataFrame(rows).to_csv(log_dir / f"{baseline_name}_episode_{ep_idx+1}.csv", index=False)

    metrics = {
        "avg_return": float(np.mean(episode_returns)),
        "median_return": float(np.median(episode_returns)),
        "best_return": float(np.max(episode_returns)),
        "worst_return": float(np.min(episode_returns)),
        "episode_return_std": float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0,
        "avg_sortino": float(np.mean([compute_sortino(r) for r in step_returns_all])),
        "avg_std_returns": float(np.mean([std_of_returns(r) for r in step_returns_all])),
        "avg_max_drawdown": float(np.mean(mdds)),
        "worst_max_drawdown": float(np.max(mdds)) if mdds else 0.0,
    }
    return metrics


def run_period(
    period_name: str,
    price_data: dict,
    n_episodes: int,
    episode_length: int,
    n_jobs: int,  # kept for symmetry; baselines are typically fast, so sequential is fine
    out_csv: Path,
    logs_root: Path | None,
    fixed_frequency: int,
    threshold: float,
    target_pct: float,
) -> None:
    seeds = list(range(n_episodes))  # fixed episode seeds for reproducibility

    # Create fresh env per baseline run (baseline itself mutates env state)
    env_ff = UniswapEnvBaselines(price_data, auto_rebalance=False)
    env_th = UniswapEnvBaselines(price_data, auto_rebalance=False)

    ff = FixedFrequencyHedgingBaseline(env_ff, hedge_frequency=fixed_frequency, target_pct=target_pct)
    th = ThresholdHedgingBaseline(env_th, target_pct=target_pct, threshold=threshold)

    rows = []

    ff_logs = (logs_root / period_name) if logs_root else None
    th_logs = (logs_root / period_name) if logs_root else None

    ff_metrics = run_one_baseline(
        baseline_name=f"fixed_freq_{fixed_frequency}",
        baseline_obj=ff,
        n_episodes=n_episodes,
        episode_length=episode_length,
        seeds=seeds,
        log_dir=(ff_logs / f"fixed_freq_{fixed_frequency}") if ff_logs else None,
    )
    rows.append({"baseline": f"fixed_freq_{fixed_frequency}", **ff_metrics})

    th_metrics = run_one_baseline(
        baseline_name=f"threshold_{threshold}",
        baseline_obj=th,
        n_episodes=n_episodes,
        episode_length=episode_length,
        seeds=seeds,
        log_dir=(th_logs / f"threshold_{threshold}") if th_logs else None,
    )
    rows.append({"baseline": f"threshold_{threshold}", **th_metrics})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[{period_name}] Saved baseline results to {out_csv}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")

    # Validation period (defaults match your thesis scripts)
    parser.add_argument("--val-start", type=str, default="2024-01-01 00:00:00")
    parser.add_argument("--val-end", type=str, default="2024-12-31 23:45:00")

    # Test period
    parser.add_argument("--test-start", type=str, default="2025-01-01 00:00:00")
    parser.add_argument("--test-end", type=str, default="2025-08-31 23:45:00")

    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=9000)

    # Baseline params
    parser.add_argument("--target-pct", type=float, default=1.0)
    parser.add_argument("--fixed-frequency", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.01)

    # Logging
    parser.add_argument("--logs-dir", type=str, default="baseline_logs", help="Set empty string to disable logs")
    parser.add_argument("--out-val-csv", type=str, default="baseline_validation_results.csv")
    parser.add_argument("--out-test-csv", type=str, default="baseline_test_results.csv")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load periods
    price_data_val = load_price_data_slice(data_dir, args.val_start, args.val_end)
    price_data_test = load_price_data_slice(data_dir, args.test_start, args.test_end)

    logs_root = Path(args.logs_dir) if args.logs_dir else None

    run_period(
        period_name="validation",
        price_data=price_data_val,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_jobs=1,
        out_csv=Path(args.out_val_csv),
        logs_root=logs_root,
        fixed_frequency=args.fixed_frequency,
        threshold=args.threshold,
        target_pct=args.target_pct,
    )

    run_period(
        period_name="test",
        price_data=price_data_test,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_jobs=1,
        out_csv=Path(args.out_test_csv),
        logs_root=logs_root,
        fixed_frequency=args.fixed_frequency,
        threshold=args.threshold,
        target_pct=args.target_pct,
    )


if __name__ == "__main__":
    main()