import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from src.envs.uniswap_env import UniswapEnv


def check_price_data_nan(price_data: dict) -> None:
    for key, series in price_data.items():
        if isinstance(series, (np.ndarray, pd.Series)):
            n_nans = np.isnan(series).sum()
            if n_nans > 0:
                print(f"[WARNING] {key} contains {n_nans} NaN values")


def load_train_price_data(data_dir: Path, train_end: str) -> dict:
    # Load CSVs
    ethusdt_df = pd.read_csv(data_dir / "ethusdt_15min.csv", parse_dates=["timestamp"])
    btcusdt_df = pd.read_csv(data_dir / "btcusdt_15min.csv", parse_dates=["timestamp"])
    ethbtc_df = pd.read_csv(data_dir / "ethbtc_15min.csv", parse_dates=["timestamp"])
    eth_funding_df = pd.read_csv(data_dir / "eth_funding_15min.csv", parse_dates=["timestamp"])
    btc_funding_df = pd.read_csv(data_dir / "btc_funding_15min.csv", parse_dates=["timestamp"])

    # Filter training period
    train_end_ts = pd.Timestamp(train_end)
    ethusdt_train = ethusdt_df[ethusdt_df["timestamp"] <= train_end_ts]
    btcusdt_train = btcusdt_df[btcusdt_df["timestamp"] <= train_end_ts]
    ethbtc_train = ethbtc_df[ethbtc_df["timestamp"] <= train_end_ts]
    eth_funding_train = eth_funding_df[eth_funding_df["timestamp"] <= train_end_ts]
    btc_funding_train = btc_funding_df[btc_funding_df["timestamp"] <= train_end_ts]

    price_data = {
        "ethusdt": ethusdt_train["close"].values,
        "btcusdt": btcusdt_train["close"].values,
        "ethbtc": ethbtc_train["close"].values,
        "eth_funding": eth_funding_train["fundingRate"].fillna(0).values,
        "btc_funding": btc_funding_train["fundingRate"].fillna(0).values,
        "timestamp": btc_funding_train["timestamp"].values,
    }

    check_price_data_nan(price_data)
    return price_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing CSVs")
    parser.add_argument("--train-end", type=str, default="2023-12-31 23:45:00")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint freq in environment steps")
    parser.add_argument("--seed-start", type=int, default=9)
    parser.add_argument("--seed-end", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="checkpoints")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    price_data = load_train_price_data(data_dir=data_dir, train_end=args.train_end)

    for seed in range(args.seed_start, args.seed_end + 1):
        print(f"\n=== Training run with seed {seed} ===")

        env = make_vec_env(lambda: UniswapEnv(price_data), n_envs=args.n_envs)

        model = PPO("MlpPolicy", env, verbose=1, seed=seed)

        checkpoint_dir = outdir / f"seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.save_freq // args.n_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"ppo_ethbtc_fees_lp_model_seed{seed}",
        )

        print(f"Training PPO model for seed {seed}...")
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

        final_model_path = checkpoint_dir / f"ppo_ethbtc_fees_lp_model_seed{seed}_final"
        model.save(str(final_model_path))


if __name__ == "__main__":
    main()