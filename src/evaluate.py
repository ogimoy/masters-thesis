# src/evaluate.py
"""
Evaluate selected PPO checkpoints on a fixed test period.

What this script does:
1) Loads test price/funding data from CSVs in --data-dir for --test-start..--test-end
2) Loads a list of best checkpoints from --best-checkpoints-csv (typically validation_best_per_seed.csv)
3) Evaluates each checkpoint on the same fixed set of episode seeds
4) Saves:
   - test_best_checkpoints_results.csv (summary metrics per checkpoint)
   - per-episode action/hedge logs to --action-log-dir

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from stable_baselines3 import PPO

from src.envs.uniswap_env import UniswapEnv


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
# Evaluation
# ----------------------------

def evaluate_checkpoint(
    ckpt_file: str,
    price_data_test: dict,
    seeds: list[int],
    episode_length: int,
    action_log_dir: Path,
) -> tuple[str, dict]:
    """
    Evaluate one PPO checkpoint across multiple episodes (fixed seeds).
    Saves per-episode action logs as CSVs.
    Returns (ckpt_file, metrics_dict).
    """
    model = PPO.load(ckpt_file)

    all_episode_values: list[np.ndarray] = []
    episode_returns: list[float] = []
    step_returns_all: list[np.ndarray] = []

    action_log_dir.mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(ckpt_file))[0]

    for ep_idx, seed in enumerate(seeds):
        env = UniswapEnv(price_data_test)  # fresh env per episode
        obs, _ = env.reset(seed=seed)

        done = False
        portfolio_values = [env._portfolio_value()]

        while (not done) and (env.episode_step < episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            portfolio_values.append(env._portfolio_value())
            done = bool(terminated or truncated)

        # Save action/hedge logs for this episode (requires env.history to contain these keys)
        if (
            "action_token1_pct" in env.history
            and "action_token2_pct" in env.history
            and "hedge_token1_pct" in env.history
            and "hedge_token2_pct" in env.history
            and "portfolio_value_norm" in env.history
            and "timestamp" in env.history
        ):
            n_steps = len(env.history["action_token1_pct"])
            episode_actions = pd.DataFrame(
                {
                    "episode": ep_idx + 1,
                    "step": np.arange(n_steps),
                    "action_token1_pct": env.history["action_token1_pct"],
                    "action_token2_pct": env.history["action_token2_pct"],
                    "hedge_token1_pct": env.history["hedge_token1_pct"],
                    "hedge_token2_pct": env.history["hedge_token2_pct"],
                    "portfolio_value_norm": env.history["portfolio_value_norm"],
                    "timestamp": env.history["timestamp"],
                }
            )
            episode_actions.to_csv(action_log_dir / f"{base_name}_episode_{ep_idx+1}.csv", index=False)

        ep_values = np.asarray(portfolio_values, dtype=float)
        all_episode_values.append(ep_values)

        if ep_values.size > 1 and ep_values[0] != 0:
            episode_returns.append(float(ep_values[-1] / ep_values[0] - 1.0))
            step_returns_all.append(ep_values[1:] / ep_values[:-1] - 1.0)
        else:
            episode_returns.append(0.0)
            step_returns_all.append(np.asarray([0.0], dtype=float))

    metrics = {
        "avg_return": float(np.mean(episode_returns)),
        "median_return": float(np.median(episode_returns)),
        "best_return": float(np.max(episode_returns)),
        "worst_return": float(np.min(episode_returns)),
        "episode_return_std": float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0,
        "avg_sortino": float(np.mean([compute_sortino(r) for r in step_returns_all])),
        "avg_std_returns": float(np.mean([std_of_returns(r) for r in step_returns_all])),
        "avg_max_drawdown": float(np.mean([max_drawdown(ep) for ep in all_episode_values])),
        "worst_max_drawdown": float(np.max([max_drawdown(ep) for ep in all_episode_values])),
    }

    return ckpt_file, metrics


def test_best_checkpoints(
    price_data_test: dict,
    checkpoint_files: list[str],
    n_episodes: int,
    episode_length: int,
    n_jobs: int,
    action_log_dir: Path,
) -> dict[str, dict]:
    seeds = list(range(n_episodes))  # same episode seeds for every checkpoint

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_checkpoint)(ckpt_file, price_data_test, seeds, episode_length, action_log_dir)
        for ckpt_file in checkpoint_files
    )

    metrics_by_ckpt: dict[str, dict] = {}
    for ckpt_file, metrics in results:
        print(
            f"Checkpoint metrics -- {os.path.basename(ckpt_file)} "
            f"Sortino: {metrics['avg_sortino']:.3f}, "
            f"Avg Return: {metrics['avg_return']:.5f}, "
            f"MaxDD: {metrics['worst_max_drawdown']:.3f}"
        )
        metrics_by_ckpt[ckpt_file] = metrics

    return metrics_by_ckpt


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="Folder with CSV files")
    parser.add_argument("--test-start", type=str, default="2025-01-01 00:00:00")
    parser.add_argument("--test-end", type=str, default="2025-08-31 23:45:00")

    parser.add_argument(
        "--best-checkpoints-csv",
        type=str,
        default="validation_best_per_seed.csv",
        help="CSV produced by validate.py (expects a 'checkpoint' column or 'checkpoint_path')",
    )
    parser.add_argument("--best-checkpoints-dir", type=str, default="best_checkpoints", help="Folder with best .zip checkpoints")

    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=9000)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--action-log-dir", type=str, default="ppo_action_logs")

    parser.add_argument("--out-csv", type=str, default="test_best_checkpoints_results.csv")

    args = parser.parse_args()

    price_data_test = load_price_data_slice(Path(args.data_dir), args.test_start, args.test_end)

    # Load best checkpoints list
    best_df = pd.read_csv(args.best_checkpoints_csv)

    # Prefer checkpoint_path if present (more robust); fallback to checkpoint filename
    checkpoint_files: list[str] = []
    if "checkpoint_path" in best_df.columns:
        # checkpoint_path may already point to copied files in best_checkpoints_dir
        checkpoint_files = [str(p) for p in best_df["checkpoint_path"].tolist()]
    elif "checkpoint" in best_df.columns:
        best_ckpt_dir = Path(args.best_checkpoints_dir)
        checkpoint_files = [
            str(best_ckpt_dir / (os.path.splitext(os.path.basename(name))[0] + ".zip"))
            for name in best_df["checkpoint"].tolist()
        ]
    else:
        raise ValueError("best checkpoints CSV must contain either 'checkpoint_path' or 'checkpoint' column")

    # Filter missing files with warning
    existing = []
    for f in checkpoint_files:
        if os.path.exists(f):
            existing.append(f)
        else:
            print(f"[WARNING] Missing checkpoint file: {f}")
    checkpoint_files = existing

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found to evaluate.")

    metrics_by_ckpt = test_best_checkpoints(
        price_data_test=price_data_test,
        checkpoint_files=checkpoint_files,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_jobs=args.n_jobs,
        action_log_dir=Path(args.action_log_dir),
    )

    results_df = pd.DataFrame(
        [{"checkpoint": os.path.basename(ckpt), **metrics} for ckpt, metrics in metrics_by_ckpt.items()]
    )
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nTesting complete. Results saved to {args.out_csv}")


if __name__ == "__main__":
    main()