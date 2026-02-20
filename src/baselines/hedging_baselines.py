# src/baselines/hedging_baselines.py
"""
Simple baseline hedging strategies for the Uniswap hedging environment.

These are NOT training-based. They run a fixed policy against an environment instance.

Baselines included:
- FixedFrequencyHedgingBaseline: rebalance to target hedge ratio every N steps
- ThresholdHedgingBaseline: rebalance to target hedge ratio when deviation exceeds threshold

Expected env interface (matches your thesis envs):
- env.reset(seed=..., start_index=...)  -> (obs, info)
- env.step(action) -> (obs, reward, terminated, truncated, info)
- env.price_data, env.token1_key, env.token2_key, env.pair_key
- env.counter, env.transaction_cost
- env.token1_amount, env.token2_amount
- env.hedge_token1, env.hedge_token2
- env.DISCRETE_LEVELS
- env._portfolio_value(...) or env._portfolio_value() depending on env implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.utils.lp_math import calculate_hedge_cost


def _nearest_level_index(levels: list[float], value: float) -> int:
    return min(range(len(levels)), key=lambda i: abs(levels[i] - value))


def _safe_portfolio_value(env, price_token12: float, price_token1: float, price_token2: float) -> float:
    """
    Handle both env._portfolio_value() signatures:
      - some env versions: _portfolio_value()
      - baseline env version: _portfolio_value(price, token1_price, token2_price)
    """
    try:
        return float(env._portfolio_value(price_token12, price_token1, price_token2))
    except TypeError:
        return float(env._portfolio_value())


@dataclass
class BaselineStepLog:
    step: int
    reward: float
    portfolio_value: float
    hedge_token1_pct: float
    hedge_token2_pct: float
    hedge_token1_amt: float
    hedge_token2_amt: float
    transaction_cost: float
    rebalanced: bool


class FixedFrequencyHedgingBaseline:
    def __init__(self, env, hedge_frequency: int = 96, target_pct: float = 1.0):
        self.env = env
        self.hedge_frequency = int(hedge_frequency)
        self.target_pct = float(target_pct)
        self.history: list[BaselineStepLog] = []
        self.last_rebalance_step: Optional[int] = None

    def run_episode(
        self,
        seed: Optional[int] = None,
        print_hedges: bool = False,
        debug_tx: bool = False,
        start_index: Optional[int] = None,
        episode_length: Optional[int] = None,
    ) -> np.ndarray:
        self.history = []
        obs, _ = self.env.reset(seed=seed, start_index=start_index)

        done = False
        step = 0
        episode_values: list[float] = []

        self.last_rebalance_step = 0

        prev_hedge_token1 = float(self.env.hedge_token1)
        prev_hedge_token2 = float(self.env.hedge_token2)

        while not done:
            if episode_length is not None and getattr(self.env, "episode_step", 0) >= episode_length:
                break

            # Rebalance rule: every N steps
            if step % self.hedge_frequency == 0:
                target_hedge_token1 = self.target_pct * float(self.env.token1_amount)
                target_hedge_token2 = self.target_pct * float(self.env.token2_amount)
                rebalanced = True
            else:
                target_hedge_token1 = prev_hedge_token1
                target_hedge_token2 = prev_hedge_token2
                rebalanced = False

            # Prices at current env.counter
            price_token1 = float(self.env.price_data[self.env.token1_key][self.env.counter])
            price_token2 = float(self.env.price_data[self.env.token2_key][self.env.counter])
            price_token12 = float(self.env.price_data[self.env.pair_key][self.env.counter])

            portfolio_before = (
                float(self.env.portfolio_value)
                if float(getattr(self.env, "portfolio_value", 0.0)) != 0.0
                else _safe_portfolio_value(self.env, price_token12, price_token1, price_token2)
            )

            tx_cost_abs = 0.0
            total_cost_normalized = 0.0
            tc1 = 0.0
            tc2 = 0.0

            if rebalanced:
                tx_fee = float(self.env.transaction_cost)
                tc1 = calculate_hedge_cost(prev_hedge_token1, target_hedge_token1, price_token1, portfolio_before, tx_fee)
                tc2 = calculate_hedge_cost(prev_hedge_token2, target_hedge_token2, price_token2, portfolio_before, tx_fee)
                total_cost_normalized = float(tc1 + tc2)
                tx_cost_abs = float(total_cost_normalized * portfolio_before)

            # Apply baseline decision to env state BEFORE stepping
            self.env.hedge_token1 = target_hedge_token1
            self.env.hedge_token2 = target_hedge_token2

            # Split normalized costs across token1/token2 (kept close to your implementation)
            if total_cost_normalized > 0:
                self.env.token1_cost = total_cost_normalized * (tc1 / total_cost_normalized)
                self.env.token2_cost = total_cost_normalized * (tc2 / total_cost_normalized)
            else:
                self.env.token1_cost = 0.0
                self.env.token2_cost = 0.0

            self.env.total_cost = tx_cost_abs
            self.env.portfolio_value = portfolio_before - tx_cost_abs

            # Convert continuous hedge pct -> nearest discrete action (for env.step compatibility)
            hedge_pct1 = float(self.env.hedge_token1) / float(self.env.token1_amount) if float(self.env.token1_amount) else 0.0
            hedge_pct2 = float(self.env.hedge_token2) / float(self.env.token2_amount) if float(self.env.token2_amount) else 0.0

            token1_idx = _nearest_level_index(self.env.DISCRETE_LEVELS, hedge_pct1)
            token2_idx = _nearest_level_index(self.env.DISCRETE_LEVELS, hedge_pct2)
            action = (token1_idx, token2_idx)

            # Step environment (PnL/funding applied inside env.step())
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            hedge_pct1 = float(self.env.hedge_token1) / float(self.env.token1_amount) if float(self.env.token1_amount) else 0.0
            hedge_pct2 = float(self.env.hedge_token2) / float(self.env.token2_amount) if float(self.env.token2_amount) else 0.0

            self.history.append(
                BaselineStepLog(
                    step=step,
                    reward=float(reward),
                    portfolio_value=float(self.env.portfolio_value),
                    hedge_token1_pct=hedge_pct1,
                    hedge_token2_pct=hedge_pct2,
                    hedge_token1_amt=float(self.env.hedge_token1),
                    hedge_token2_amt=float(self.env.hedge_token2),
                    transaction_cost=float(tx_cost_abs),
                    rebalanced=bool(rebalanced),
                )
            )

            if debug_tx and rebalanced:
                print(f"\n=== Step {step} DEBUG TX ===")
                print(f"Portfolio BEFORE: {portfolio_before:.2f}")
                print(f"Tx cost applied: {tx_cost_abs:.2f}")
                print(f"Portfolio AFTER:  {self.env.portfolio_value:.2f}")
                print(f"prev_hedge_token1: {prev_hedge_token1:.6f} -> {target_hedge_token1:.6f}")
                print(f"prev_hedge_token2: {prev_hedge_token2:.6f} -> {target_hedge_token2:.6f}")
                print(f"total_cost_norm: {total_cost_normalized:.8f}")
                print("====================================\n")

            if print_hedges:
                print(
                    f"Step {step}: Hedge1={hedge_pct1:.3f}, Hedge2={hedge_pct2:.3f}, "
                    f"Rebalanced={rebalanced}, TxCost={tx_cost_abs:.2f}, "
                    f"Portfolio={self.env.portfolio_value:.2f}"
                )

            prev_hedge_token1 = float(self.env.hedge_token1)
            prev_hedge_token2 = float(self.env.hedge_token2)

            episode_values.append(float(self.env.portfolio_value))
            step += 1

        return np.asarray(episode_values, dtype=float)


class ThresholdHedgingBaseline:
    def __init__(self, env, target_pct: float = 1.0, threshold: float = 0.01):
        """
        target_pct: target hedge ratio (1.0 = fully hedged)
        threshold: absolute deviation from target_pct to trigger rebalance
        """
        self.env = env
        self.target_pct = float(target_pct)
        self.threshold = float(threshold)
        self.history: list[BaselineStepLog] = []

    def run_episode(
        self,
        seed: Optional[int] = None,
        print_hedges: bool = False,
        debug_tx: bool = False,
        start_index: Optional[int] = None,
        episode_length: Optional[int] = None,
    ) -> np.ndarray:
        self.history = []
        obs, _ = self.env.reset(seed=seed, start_index=start_index)

        done = False
        step = 0
        episode_values: list[float] = []

        prev_hedge_token1 = float(self.env.hedge_token1)
        prev_hedge_token2 = float(self.env.hedge_token2)

        while not done:
            if episode_length is not None and getattr(self.env, "episode_step", 0) >= episode_length:
                break

            current_pct1 = prev_hedge_token1 / float(self.env.token1_amount) if float(self.env.token1_amount) else 0.0
            current_pct2 = prev_hedge_token2 / float(self.env.token2_amount) if float(self.env.token2_amount) else 0.0

            dev1 = abs(current_pct1 - self.target_pct)
            dev2 = abs(current_pct2 - self.target_pct)

            if dev1 >= self.threshold or dev2 >= self.threshold:
                target_hedge_token1 = self.target_pct * float(self.env.token1_amount)
                target_hedge_token2 = self.target_pct * float(self.env.token2_amount)
                rebalanced = True
            else:
                target_hedge_token1 = prev_hedge_token1
                target_hedge_token2 = prev_hedge_token2
                rebalanced = False

            # Prices at current env.counter
            price_token1 = float(self.env.price_data[self.env.token1_key][self.env.counter])
            price_token2 = float(self.env.price_data[self.env.token2_key][self.env.counter])
            price_token12 = float(self.env.price_data[self.env.pair_key][self.env.counter])

            portfolio_before = (
                float(self.env.portfolio_value)
                if float(getattr(self.env, "portfolio_value", 0.0)) != 0.0
                else _safe_portfolio_value(self.env, price_token12, price_token1, price_token2)
            )

            tx_cost_abs = 0.0
            total_cost_normalized = 0.0
            tc1 = 0.0
            tc2 = 0.0

            if rebalanced:
                tx_fee = float(self.env.transaction_cost)
                tc1 = calculate_hedge_cost(prev_hedge_token1, target_hedge_token1, price_token1, portfolio_before, tx_fee)
                tc2 = calculate_hedge_cost(prev_hedge_token2, target_hedge_token2, price_token2, portfolio_before, tx_fee)
                total_cost_normalized = float(tc1 + tc2)
                tx_cost_abs = float(total_cost_normalized * portfolio_before)

            # Apply baseline decision BEFORE stepping
            self.env.hedge_token1 = target_hedge_token1
            self.env.hedge_token2 = target_hedge_token2

            if total_cost_normalized > 0:
                self.env.token1_cost = total_cost_normalized * (tc1 / total_cost_normalized)
                self.env.token2_cost = total_cost_normalized * (tc2 / total_cost_normalized)
            else:
                self.env.token1_cost = 0.0
                self.env.token2_cost = 0.0

            self.env.total_cost = tx_cost_abs
            self.env.portfolio_value = portfolio_before - tx_cost_abs

            hedge_pct1 = float(self.env.hedge_token1) / float(self.env.token1_amount) if float(self.env.token1_amount) else 0.0
            hedge_pct2 = float(self.env.hedge_token2) / float(self.env.token2_amount) if float(self.env.token2_amount) else 0.0

            token1_idx = _nearest_level_index(self.env.DISCRETE_LEVELS, hedge_pct1)
            token2_idx = _nearest_level_index(self.env.DISCRETE_LEVELS, hedge_pct2)
            action = (token1_idx, token2_idx)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            hedge_pct1 = float(self.env.hedge_token1) / float(self.env.token1_amount) if float(self.env.token1_amount) else 0.0
            hedge_pct2 = float(self.env.hedge_token2) / float(self.env.token2_amount) if float(self.env.token2_amount) else 0.0

            self.history.append(
                BaselineStepLog(
                    step=step,
                    reward=float(reward),
                    portfolio_value=float(self.env.portfolio_value),
                    hedge_token1_pct=hedge_pct1,
                    hedge_token2_pct=hedge_pct2,
                    hedge_token1_amt=float(self.env.hedge_token1),
                    hedge_token2_amt=float(self.env.hedge_token2),
                    transaction_cost=float(tx_cost_abs),
                    rebalanced=bool(rebalanced),
                )
            )

            if print_hedges:
                print(
                    f"Step {step}: Hedge1={hedge_pct1:.3f}, Hedge2={hedge_pct2:.3f}, "
                    f"Rebalanced={rebalanced}, TxCost={tx_cost_abs:.2f}, "
                    f"Portfolio={self.env.portfolio_value:.2f}"
                )

            prev_hedge_token1 = float(self.env.hedge_token1)
            prev_hedge_token2 = float(self.env.hedge_token2)

            episode_values.append(float(self.env.portfolio_value))
            step += 1

        return np.asarray(episode_values, dtype=float)