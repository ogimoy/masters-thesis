# src/envs/baseline_env.py
"""
Baseline environment for Uniswap v3 LP hedging (Gymnasium).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete

from src.utils.lp_math import (
    uniswap_v3_composition,
    rolling_volatility,
    uniswap_v3_value,
    calculate_ema,
    calculate_hedge_cost,
    calculate_hedge_pnl,
    compute_rolling_sortino,  # kept (may be used for diagnostics)
)


class UniswapEnvBaselines(gym.Env):
    """
    price_data expects (at minimum) the following keys:
      - "ethusdt": np.ndarray of close prices (token1 in USD)
      - "btcusdt": np.ndarray of close prices (token2 in USD)
      - "ethbtc":  np.ndarray of close prices (pair price)
      - "eth_funding": np.ndarray of funding rates aligned to 15-min timestamps
      - "btc_funding": np.ndarray of funding rates aligned to 15-min timestamps
      - "timestamp": np.ndarray of datetime-like timestamps aligned to 15-min grid

    Baseline behavior:
      - The environment can auto-rebalance hedge back to 1.0x of LP token amounts
        when hedge ratios deviate outside a tolerance band (auto_rebalance=True).
      - The action space is still defined, but the env may override hedge levels if
        auto_rebalance is enabled (matches your original baseline intent).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_data: dict,
        liquidity: float = 100000,
        transaction_cost: float = 0.003,
        step_length: int = 1,
        rebalance_tolerance: float = 0.025,
        auto_rebalance: bool = True,
    ):
        super().__init__()

        self.step_length = int(step_length)
        self.rebalance_tolerance = float(rebalance_tolerance)
        self.auto_rebalance = bool(auto_rebalance)

        self.price_data = price_data

        # Use explicit keys (safer than relying on dict insertion order)
        self.token1_key = "ethusdt"
        self.token2_key = "btcusdt"
        self.pair_key = "ethbtc"
        self.token1_funding_key = "eth_funding"
        self.token2_funding_key = "btc_funding"
        self.timestamp_key = "timestamp"

        self.lp_liquidity = liquidity
        self.lp_lower_bound = 0.0
        self.lp_upper_bound = 0.0
        self.total_funding_pnl = 0.0
        self.transaction_cost = transaction_cost

        self.max_steps = len(self.price_data[self.token1_key]) - 1
        self.step_counter = 0
        self.counter = 0
        self.episode_step = 0

        # Hedge state
        self.hedge_token1 = 0.0
        self.hedge_token2 = 0.0
        self.prev_token1_price = 0.0
        self.prev_token2_price = 0.0
        self.hedge_value = 0.0
        self.prev_hedge_pnl = 0.0

        # Costs (kept; baseline may or may not set these explicitly)
        self.token1_cost = 0.0
        self.token2_cost = 0.0
        self.total_cost = 0.0

        # LP / portfolio state
        self.initial_token1_price = 0.0
        self.initial_token2_price = 0.0
        self.initial_token1token2_price = 0.0
        self.initial_value = 0.0

        self.token1_amount = 0.0
        self.token2_amount = 0.0
        self.lp_value = 0.0
        self.portfolio_value = 0.0

        # Diagnostics (kept; not required)
        self.recent_returns = []
        self.sortino_window = 20

        # Discrete hedge levels (percentages)
        self.DISCRETE_LEVELS = [round(x, 2) for x in np.arange(0.0, 1.25, 0.05)]
        self.action_space = MultiDiscrete([len(self.DISCRETE_LEVELS), len(self.DISCRETE_LEVELS)])

        # Observation is 20-dim (same as main env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.history = {
            "portfolio_value_norm": [],
            "hedge_token1_norm": [],
            "hedge_token2_norm": [],
            "token1_norm": [],
            "token2_norm": [],
            "token1token2_norm": [],
            "lp_token1_value_norm": [],
            "lp_token2_value_norm": [],
            "reward": [],
            "hedge_token1_pct": [],
            "hedge_token2_pct": [],
            "hedge_token1_amt": [],
            "hedge_token2_amt": [],
            "timestamp": [],
            "lp_value": [],
        }

    def _set_range(self, idx: int) -> None:
        window = 24 * 30 * 4  # 30 days worth of 15-min steps

        if idx < window:
            volatility = 0.08
        else:
            prices = self.price_data[self.pair_key][idx - window : idx]
            daily_prices = prices[::96]
            daily_returns = np.diff(np.log(daily_prices))
            volatility = float(np.std(daily_returns) * np.sqrt(30))

        current_price = float(self.price_data[self.pair_key][idx])
        self.lp_lower_bound = current_price * (1 - 2 * volatility)
        self.lp_upper_bound = current_price * (1 / (1 - 2 * volatility))

    def reset(self, *, seed=None, options=None, start_index: int | None = None):
        super().reset(seed=seed)

        self.episode_step = 0

        if start_index is not None:
            self.step_counter = int(start_index)
            self.counter = int(start_index)
        else:
            max_start = (
                min(
                    len(self.price_data[self.pair_key]),
                    len(self.price_data[self.token1_key]),
                    len(self.price_data[self.token2_key]),
                )
                - 9000
                - 1
            )
            if max_start <= 0:
                raise ValueError("Price data too short for 9000-step episodes.")

            # Kept from your baseline: deterministic start at 1 (instead of random)
            self.step_counter = 1
            self.counter = self.step_counter

        self._set_range(self.counter)

        price = float(self.price_data[self.pair_key][self.counter])
        token1_price = float(self.price_data[self.token1_key][self.counter])
        token2_price = float(self.price_data[self.token2_key][self.counter])

        self.initial_token1_price = token1_price
        self.initial_token2_price = token2_price
        self.initial_token1token2_price = price

        self.token1_amount, self.token2_amount = uniswap_v3_composition(
            price, self.lp_lower_bound, self.lp_upper_bound, self.lp_liquidity
        )

        # Fully hedged at reset
        self.hedge_token1 = self.token1_amount
        self.hedge_token2 = self.token2_amount

        # Use current prices as previous prices at reset
        self.prev_token1_price = token1_price
        self.prev_token2_price = token2_price

        self.hedge_value = 0.0
        self.prev_hedge_pnl = 0.0
        self.token1_cost = 0.0
        self.token2_cost = 0.0
        self.total_cost = 0.0
        self.total_funding_pnl = 0.0

        self.initial_value = self._portfolio_value(price, token1_price, token2_price)
        self.lp_value = 0.0
        self.portfolio_value = 0.0
        self.cumulative_return = 0.0

        return self._get_obs(), {}

    def step(self, action, debug: bool = False):
        token1_idx, token2_idx = action
        target_token1_pct = self.DISCRETE_LEVELS[int(token1_idx)]
        target_token2_pct = self.DISCRETE_LEVELS[int(token2_idx)]
        # Note: in this baseline, actions are effectively ignored if auto_rebalance=True,
        # because rebalance_hedge() targets 1.0. We keep the action interface for compatibility.

        price = float(self.price_data[self.pair_key][self.counter])
        current_token1_price = float(self.price_data[self.token1_key][self.counter])
        current_token2_price = float(self.price_data[self.token2_key][self.counter])

        # LP composition
        self.token1_amount, self.token2_amount = uniswap_v3_composition(
            price, self.lp_lower_bound, self.lp_upper_bound, self.lp_liquidity
        )

        prev_value = self._portfolio_value(price, current_token1_price, current_token2_price)

        # Current hedge percentages
        current_hedge_token1_pct = self.hedge_token1 / self.token1_amount if self.token1_amount != 0 else 0.0
        current_hedge_token2_pct = self.hedge_token2 / self.token2_amount if self.token2_amount != 0 else 0.0

        token1_prev = self.hedge_token1
        token2_prev = self.hedge_token2

        # Baseline: auto rebalance towards 1.0 hedge if outside tolerance
        token1_adj, token2_adj = self.rebalance_hedge(
            current_hedge_pct1=current_hedge_token1_pct,
            current_hedge_pct2=current_hedge_token2_pct,
            target_hedge_pct=1.0,
            token1_amount=self.token1_amount,
            token2_amount=self.token2_amount,
            prev_token1=token1_prev,
            prev_token2=token2_prev,
            tolerance=self.rebalance_tolerance,
        )

        self.hedge_token1 = token1_adj
        self.hedge_token2 = token2_adj

        # Hedge PnL (uses previous prices)
        hedge_pnl_token1 = calculate_hedge_pnl(self.prev_token1_price, current_token1_price, self.hedge_token1)
        hedge_pnl_token2 = calculate_hedge_pnl(self.prev_token2_price, current_token2_price, self.hedge_token2)

        # Funding payments (8h cadence)
        current_timestamp = self.price_data[self.timestamp_key][self.counter]
        if hasattr(current_timestamp, "astype"):
            current_timestamp = current_timestamp.astype("M8[ms]").astype("O")

        hour = current_timestamp.hour
        minute = current_timestamp.minute

        if (hour % 8 == 0) and (minute == 0):
            eth_funding = float(self.price_data[self.token1_funding_key][self.counter])
            btc_funding = float(self.price_data[self.token2_funding_key][self.counter])
            funding_pnl_token1 = self.hedge_token1 * eth_funding * current_token1_price
            funding_pnl_token2 = self.hedge_token2 * btc_funding * current_token2_price
            hedge_pnl_token1 += funding_pnl_token1
            hedge_pnl_token2 += funding_pnl_token2
            self.total_funding_pnl += funding_pnl_token1 + funding_pnl_token2

        # Update hedge value
        current_hedge_pnl = hedge_pnl_token1 + hedge_pnl_token2
        self.hedge_value += current_hedge_pnl
        self.prev_hedge_pnl = current_hedge_pnl

        # Cost handling (kept): if token1_cost/token2_cost are set externally, apply them
        absolute_token1_cost = self.token1_cost * prev_value
        absolute_token2_cost = self.token2_cost * prev_value
        self.total_cost = absolute_token1_cost + absolute_token2_cost
        self.hedge_value -= self.total_cost

        # Update previous prices for next step
        self.prev_token1_price = current_token1_price
        self.prev_token2_price = current_token2_price

        # LP and portfolio values
        self.lp_value = self.calculate_lp_value(price, current_token1_price, current_token2_price)
        self.portfolio_value = self._portfolio_value(price, current_token1_price, current_token2_price)

        pnl = self.portfolio_value - prev_value
        return_pct = pnl / prev_value
        reward = 10.0 * return_pct

        # Advance time
        out_of_range = price < self.lp_lower_bound or price > self.lp_upper_bound
        self.counter += self.step_length
        self.step_counter += self.step_length
        self.episode_step += self.step_length

        terminated = bool(out_of_range)
        truncated = bool(
            (self.episode_step >= 9000)
            or (self.step_counter >= len(self.price_data[self.pair_key]) - self.step_length)
        )

        obs = self._get_obs()

        # History
        ts_next = self.price_data[self.timestamp_key][self.counter]
        self.history["timestamp"].append(ts_next)

        self.hedge_token1_pct = self.hedge_token1 / self.token1_amount if self.token1_amount else 0.0
        self.hedge_token2_pct = self.hedge_token2 / self.token2_amount if self.token2_amount else 0.0

        self.history["lp_value"].append(self.lp_value)
        self.history["portfolio_value_norm"].append(self.portfolio_value / self.initial_value)
        self.history["hedge_token1_norm"].append(float(obs[5]))
        self.history["hedge_token2_norm"].append(float(obs[6]))
        self.history["token1_norm"].append(float(obs[0]))
        self.history["token2_norm"].append(float(obs[1]))
        self.history["token1token2_norm"].append(float(obs[2]))
        self.history["lp_token1_value_norm"].append(float(obs[3]))
        self.history["lp_token2_value_norm"].append(float(obs[4]))
        self.history["reward"].append(float(reward))
        self.history["hedge_token1_pct"].append(float(self.hedge_token1_pct))
        self.history["hedge_token2_pct"].append(float(self.hedge_token2_pct))
        self.history["hedge_token1_amt"].append(float(self.hedge_token1))
        self.history["hedge_token2_amt"].append(float(self.hedge_token2))

        return obs, float(reward), terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        idx = self.counter

        token1_price = float(self.price_data[self.token1_key][idx])
        token2_price = float(self.price_data[self.token2_key][idx])
        token1token2 = float(self.price_data[self.pair_key][idx])

        token1_price_norm = token1_price / self.initial_token1_price
        token2_price_norm = token2_price / self.initial_token2_price
        token1token2_norm = token1token2 / self.initial_token1token2_price

        window = max(100, 20) + 1
        start_idx = max(0, idx - window)

        prices_token1 = self.price_data[self.token1_key][start_idx : idx + 1]
        prices_token2 = self.price_data[self.token2_key][start_idx : idx + 1]
        prices_pair = self.price_data[self.pair_key][start_idx : idx + 1]

        ema20_token1 = calculate_ema(prices_token1, 20)[-1]
        ema100_token1 = calculate_ema(prices_token1, 100)[-1]
        ema20_token2 = calculate_ema(prices_token2, 20)[-1]
        ema100_token2 = calculate_ema(prices_token2, 100)[-1]
        ema20_pair = calculate_ema(prices_pair, 20)[-1]
        ema100_pair = calculate_ema(prices_pair, 100)[-1]

        token1_amt, token2_amt = uniswap_v3_composition(
            price=token1token2,
            lower_bound=self.lp_lower_bound,
            upper_bound=self.lp_upper_bound,
            liquidity=self.lp_liquidity,
        )

        lp_token1_value = token1_amt * token1_price / self.initial_value
        lp_token2_value = token2_amt * token2_price / self.initial_value

        hedge_token1_value = self.hedge_token1 * token1_price / self.initial_value
        hedge_token2_value = self.hedge_token2 * token2_price / self.initial_value

        hedge_token1_pct = self.hedge_token1 / token1_amt if token1_amt != 0 else 0.0
        hedge_token2_pct = self.hedge_token2 / token2_amt if token2_amt != 0 else 0.0

        vol5 = rolling_volatility(self.price_data[self.pair_key][: idx + 1])
        vol5_scaled = np.clip(vol5, 0, 0.1) / 0.05

        vol20 = rolling_volatility(self.price_data[self.pair_key][: idx + 1], 20)
        vol20_scaled = np.clip(vol20, 0, 0.1) / 0.05

        width_norm = (self.lp_upper_bound - self.lp_lower_bound) / self.initial_token1token2_price

        eth_funding = float(self.price_data[self.token1_funding_key][idx])
        btc_funding = float(self.price_data[self.token2_funding_key][idx])
        clip_value = 0.01
        eth_funding_scaled = np.clip(eth_funding, -clip_value, clip_value) / clip_value
        btc_funding_scaled = np.clip(btc_funding, -clip_value, clip_value) / clip_value

        ema_features = [
            (ema20_token1 / token1_price) - 1,
            (ema100_token1 / token1_price) - 1,
            (ema20_token2 / token2_price) - 1,
            (ema100_token2 / token2_price) - 1,
            (ema20_pair / token1token2) - 1,
            (ema100_pair / token1token2) - 1,
        ]

        return np.array(
            [
                token1_price_norm,
                token2_price_norm,
                token1token2_norm,
                lp_token1_value,
                lp_token2_value,
                hedge_token1_value,
                hedge_token2_value,
                hedge_token1_pct,
                hedge_token2_pct,
                vol5_scaled,
                vol20_scaled,
                width_norm,
                eth_funding_scaled,
                btc_funding_scaled,
                *ema_features,
            ],
            dtype=np.float32,
        )

    def calculate_lp_value(self, price: float, token1_price: float, token2_price: float) -> float:
        return float(
            uniswap_v3_value(
                price=price,
                lower_bound=self.lp_lower_bound,
                upper_bound=self.lp_upper_bound,
                liquidity=self.lp_liquidity,
                token1_usd_price=token1_price,
                token2_usd_price=token2_price,
            )
        )

    def _portfolio_value(self, price: float, token1_price: float, token2_price: float) -> float:
        lp_val = self.calculate_lp_value(price, token1_price, token2_price)
        hedge_val = self.hedge_value
        return float(lp_val + hedge_val)

    def rebalance_hedge(
        self,
        current_hedge_pct1: float,
        current_hedge_pct2: float,
        target_hedge_pct: float,
        token1_amount: float,
        token2_amount: float,
        prev_token1: float,
        prev_token2: float,
        tolerance: float | None = None,
    ) -> tuple[float, float]:
        if not self.auto_rebalance:
            return float(prev_token1), float(prev_token2)

        if tolerance is None:
            tolerance = self.rebalance_tolerance

        if (abs(current_hedge_pct1 - target_hedge_pct) > tolerance) or (abs(current_hedge_pct2 - target_hedge_pct) > tolerance):
            token1_new = target_hedge_pct * token1_amount
            token2_new = target_hedge_pct * token2_amount
            return float(token1_new), float(token2_new)

        return float(prev_token1), float(prev_token2)

    def save_history(self, filename: str = "training_history.npz") -> None:
        np.savez(
            filename,
            portfolio_value_norm=np.array(self.history["portfolio_value_norm"]),
            hedge_token1_norm=np.array(self.history["hedge_token1_norm"]),
            hedge_token2_norm=np.array(self.history["hedge_token2_norm"]),
            token1_norm=np.array(self.history["token1_norm"]),
            token2_norm=np.array(self.history["token2_norm"]),
            token1token2_norm=np.array(self.history["token1token2_norm"]),
            lp_token1_value_norm=np.array(self.history["lp_token1_value_norm"]),
            lp_token2_value_norm=np.array(self.history["lp_token2_value_norm"]),
            reward=np.array(self.history["reward"]),
        )