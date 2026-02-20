# src/validate.py
"""
Validate PPO checkpoints on a fixed validation period and select the best checkpoint per seed.

What this script does:
1) Loads validation price/funding data from CSVs in --data-dir
2) For each training seed directory under --checkpoint-root:
   - evaluates all matching checkpoint .zip files on the same fixed set of episode seeds
   - computes metrics (mean Sortino, mean return, worst max drawdown)
   - selects the checkpoint with the best mean Sortino
   - copies that checkpoint into --best-checkpoints-dir
3) Writes:
   - validation_results_fixed.csv (all checkpoints + metrics)
   - validation_best_per_seed.csv (one best checkpoint per seed; used by evaluate.py)

"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
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
    if step_returns.size == 0:
        return float("nan")

    downside = step_returns[step_returns < target]
    if downside.size == 0:
        return float("inf")  # "perfect" sortino (no downside)

    downside_std = float(np.std(downside))
    if downside_std == 0.0 or np.isnan(downside_std):
        return float("nan")

    expected_return = float(np.mean(step_returns))
    return float((expected_return - target) / downside_std)


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


# ----------------------------
# Data loading
# ----------------------------

def load_price_data_slice(
    data_dir: Path,
    start: str,
    end: str,
) -> dict:
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

    price_data = {
        "ethusdt": ethusdt["close"].values,
        "btcusdt": btcusdt["close"].values,
        "ethbtc": ethbtc["close"].values,
        "eth_funding": eth_funding["fundingRate"].fillna(0).values,
        "btc_funding": btc_funding["fundingRate"].fillna(0).values,
        "timestamp": btc_funding["timestamp"].values,
    }

    # Basic NaN warning (non-fatal)
    for k, arr in price_data.items():
        if isinstance(arr, (np.ndarray, pd.Series)):
            try:
                n_nans = int(np.isnan(arr).sum())
            except TypeError:
                n_nans = 0
            if n_nans > 0:
                print(f"[WARNING] {k} contains {n_nans} NaN values")

    return price_data


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_checkpoint_fixed_seeds(
    ckpt_file: str,
    price_data_val: dict,
    episode_seeds: list[int],
    episode_length: int,
) -> tuple[str, dict]:
    """
    Evaluate one checkpoint on multiple episodes with fixed seeds.
    Creates a fresh env for each episode (safe & deterministic).
    """
    model = PPO.load(ckpt_file)

    episode_returns: list[float] = []
    episode_sortinos: list[float] = []
    episode_max_dds: list[float] = []

    for ep_seed in episode_seeds:
        env = UniswapEnv(price_data_val)  # fresh env per episode
        obs, _ = env.reset(seed=ep_seed)

        done = False
        portfolio_values = [env._portfolio_value()]

        while (not done) and (env.episode_step < episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            portfolio_values.append(env._portfolio_value())
            done = bool(terminated or truncated)

        portfolio_values = np.asarray(portfolio_values, dtype=float)
        if portfolio_values.size < 2:
            ep_return = 0.0
            step_returns = np.asarray([], dtype=float)
        else:
            ep_return = float((portfolio_values[-1] / portfolio_values[0]) - 1.0)
            step_returns = portfolio_values[1:] / portfolio_values[:-1] - 1.0

        episode_returns.append(ep_return)
        episode_sortinos.append(compute_sortino(step_returns))
        episode_max_dds.append(max_drawdown(portfolio_values))

    metrics = {
        "mean_sortino": float(np.nanmean(episode_sortinos)),
        "mean_return": float(np.mean(episode_returns)),
        "max_drawdown": float(np.max(episode_max_dds)) if episode_max_dds else 0.0,
    }
    return ckpt_file, metrics


def validate_all_seeds(
    price_data_val: dict,
    seeds: list[int],
    checkpoint_root: Path,
    pattern: str,
    best_checkpoints_dir: Path,
    n_episodes: int,
    episode_length: int,
    n_jobs: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - best_checkpoints (dict)
      - results_df (all checkpoints + metrics)
      - best_df (best checkpoint per seed)
    """
    best_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    validation_episode_seeds = list(range(1, n_episodes + 1))

    all_rows = []
    best_rows = []
    best_checkpoints: dict[int, dict] = {}

    for seed in seeds:
        print(f"\n=== Validating seed {seed} ===")
        seed_dir = checkpoint_root / f"seed_{seed}"
        checkpoint_files = sorted(glob.glob(str(seed_dir / pattern)))

        if not checkpoint_files:
            print(f"[WARNING] No checkpoints found for seed {seed} in {seed_dir} (pattern: {pattern})")
            continue

        # Evaluate all checkpoints in parallel (each job creates fresh envs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_checkpoint_fixed_seeds)(ckpt_file, price_data_val, validation_episode_seeds, episode_length)
            for ckpt_file in checkpoint_files
        )

        seed_best_sortino = -np.inf
        seed_best_file = None

        for ckpt_file, metrics in results:
            print(
                f"Checkpoint metrics -- {os.path.basename(ckpt_file)} "
                f"Sortino: {metrics['mean_sortino']:.3f}, "
                f"Mean Episode Return: {metrics['mean_return']:.5f}, "
                f"Max DD: {metrics['max_drawdown']:.3f}"
            )

            all_rows.append(
                {
                    "seed": seed,
                    "checkpoint": os.path.basename(ckpt_file),
                    "checkpoint_path": ckpt_file,
                    **metrics,
                }
            )

            if metrics["mean_sortino"] > seed_best_sortino:
                seed_best_sortino = metrics["mean_sortino"]
                seed_best_file = ckpt_file

        if seed_best_file is None:
            print(f"[WARNING] Could not determine best checkpoint for seed {seed}")
            continue

        # Copy best checkpoint to common folder
        dest_path = best_checkpoints_dir / os.path.basename(seed_best_file)
        shutil.copy(seed_best_file, dest_path)
        print(f"Copied best checkpoint for seed {seed} to {best_checkpoints_dir}")

        best_checkpoints[seed] = {
            "checkpoint": seed_best_file,
            "sortino": float(seed_best_sortino),
        }
        best_rows.append(
            {
                "seed": seed,
                "checkpoint": os.path.basename(seed_best_file),
                "checkpoint_path": str(dest_path),
                "mean_sortino": float(seed_best_sortino),
            }
        )

        print(
            f"Best checkpoint for seed {seed}: {os.path.basename(seed_best_file)} "
            f"with Sortino {seed_best_sortino:.3f}"
        )

    results_df = pd.DataFrame(all_rows).sort_values(["seed", "mean_sortino"], ascending=[True, False])
    best_df = pd.DataFrame(best_rows).sort_values(["seed"])

    return best_checkpoints, results_df, best_df


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="Folder with CSV files")
    parser.add_argument("--val-start", type=str, default="2024-01-01 00:00:00")
    parser.add_argument("--val-end", type=str, default="2024-12-31 23:45:00")

    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Root folder containing seed_* dirs")
    parser.add_argument("--pattern", type=str, default="*fees*.zip", help="Glob pattern for checkpoint files within seed dir")
    parser.add_argument("--best-checkpoints-dir", type=str, default="best_checkpoints", help="Where to copy best checkpoints")

    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=10)

    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=9000)

    parser.add_argument("--n-jobs", type=int, default=-1)

    args = parser.parse_args()

    price_data_val = load_price_data_slice(Path(args.data_dir), args.val_start, args.val_end)

    seeds = list(range(args.seed_start, args.seed_end + 1))

    _, results_df, best_df = validate_all_seeds(
        price_data_val=price_data_val,
        seeds=seeds,
        checkpoint_root=Path(args.checkpoint_root),
        pattern=args.pattern,
        best_checkpoints_dir=Path(args.best_checkpoints_dir),
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_jobs=args.n_jobs,
    )

    # Save outputs
    results_df.to_csv("validation_results_fixed.csv", index=False)
    best_df.to_csv("validation_best_per_seed.csv", index=False)

    print("\nValidation complete.")
    print("Saved: validation_results_fixed.csv")
    print("Saved: validation_best_per_seed.csv")


if __name__ == "__main__":
    main()