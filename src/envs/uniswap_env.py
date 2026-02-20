# src/envs/uniswap_env.py
"""
Uniswap v3 LP hedging environment (Gymnasium).

Notes:
- Two hedge actions (token1, token2) as discrete multipliers of current LP token amounts.
- Episode terminates if LP goes out of range; truncates after max steps.
- Funding PnL is applied at 00:00, 08:00, 16:00 UTC based on the provided timestamp series.
- Observation is a 20-dim float32 vector (see _get_obs()).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete

# If you place lp_math in src/utils/lp_math.py, use this import:
from src.utils.lp_math import (
    uniswap_v3_composition,
    rolling_volatility,
    uniswap_v3_value,
    calculate_ema,
    rebalance_hedge,
    calculate_hedge_cost,
    calculate_hedge_pnl,
    compute_rolling_sortino,  # kept (used for logging/experiments)
)


class UniswapEnv(gym.Env):
    """
    price_data expects (at minimum) the following keys:
      - "ethusdt": np.ndarray of close prices (token1 in USD)
      - "btcusdt": np.ndarray of close prices (token2 in USD)
      - "ethbtc":  np.ndarray of close prices (pair price)
      - "eth_funding": np.ndarray of funding rates aligned to 15-min timestamps
      - "btc_funding": np.ndarray of funding rates aligned to 15-min timestamps
      - "timestamp": np.ndarray of datetime-like timestamps aligned to 15-min grid
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_data: dict,
        liquidity: float = 100000,
        transaction_cost: float = 0.003,
        step_length: int = 1,
    ):
        super().__init__()

        self.step_length = int(step_length)
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

        # Costs
        self.token1_cost = 0.0
        self.token2_cost = 0.0
        self.total_cost = 0.0

        # LP state
        self.token1_amount = 0.0
        self.token2_amount = 0.0

        # Initial prices (will be overwritten in reset() after random start selection)
        self.initial_token1_price = float(self.price_data[self.token1_key][0])
        self.initial_token2_price = float(self.price_data[self.token2_key][0])
        self.initial_token1token2_price = float(self.price_data[self.pair_key][0])

        self.lp_value = 0.0
        self.portfolio_value = 0.0
        self.initial_value = self._portfolio_value()

        # Reward-shaping / diagnostics (kept; reward currently uses scaled return only)
        self.recent_returns = []
        self.sortino_window = 20

        # Action space: discrete hedge multipliers
        self.DISCRETE_LEVELS = [round(x, 2) for x in np.arange(0.5, 1.25, 0.05)]
        self.action_space = MultiDiscrete([len(self.DISCRETE_LEVELS), len(self.DISCRETE_LEVELS)])

        # Observation: 20 dims (14 core + 6 EMA features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32,
        )

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
        }

    def _set_range(self, idx: int) -> None:
        # 30 days worth of 15-min steps: 30*24*4 = 2880
        window = 24 * 30 * 4

        if idx < window:
            volatility = 0.08
        else:
            prices = self.price_data[self.pair_key][idx - window : idx]
            daily_prices = prices[::96]  # 96x 15-min per day
            daily_returns = np.diff(np.log(daily_prices))
            volatility = float(np.std(daily_returns) * np.sqrt(30))

        current_price = float(self.price_data[self.pair_key][idx])
        self.lp_lower_bound = current_price * (1 - 2 * volatility)
        self.lp_upper_bound = current_price * (1 / (1 - 2 * volatility))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_step = 0

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

        self.step_counter = int(self.np_random.integers(0, max_start))
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

        # Initialize to fully hedged
        self.hedge_token1 = self.token1_amount
        self.hedge_token2 = self.token2_amount
        self.prev_token1_price = token1_price
        self.prev_token2_price = token2_price
        self.hedge_value = 0.0

        self.prev_hedge_pnl = 0.0
        self.token1_cost = 0.0
        self.token2_cost = 0.0
        self.total_cost = 0.0

        self.recent_returns = []

        self.lp_value = self.calculate_lp_value(self.counter)
        self.initial_value = self._portfolio_value()
        self.portfolio_value = self.initial_value
        self.cumulative_return = 0.0

        return self._get_obs(), {}

    def step(self, action):
        token1_idx, token2_idx = action
        target_token1_pct = self.DISCRETE_LEVELS[int(token1_idx)]
        target_token2_pct = self.DISCRETE_LEVELS[int(token2_idx)]

        self.prev_hedge_pnl = 0.0
        self.total_cost = 0.0

        price = float(self.price_data[self.pair_key][self.counter])
        current_token1_price = float(self.price_data[self.token1_key][self.counter])
        current_token2_price = float(self.price_data[self.token2_key][self.counter])

        # Update LP composition
        self.token1_amount, self.token2_amount = uniswap_v3_composition(
            price, self.lp_lower_bound, self.lp_upper_bound, self.lp_liquidity
        )

        prev_value = self._portfolio_value()

        current_hedge_token1_pct = self.hedge_token1 / self.token1_amount if self.token1_amount != 0 else 0.0
        current_hedge_token2_pct = self.hedge_token2 / self.token2_amount if self.token2_amount != 0 else 0.0

        token1_prev = self.hedge_token1
        token2_prev = self.hedge_token2

        token1_adj = rebalance_hedge(current_hedge_token1_pct, target_token1_pct, self.token1_amount, token1_prev)
        token2_adj = rebalance_hedge(current_hedge_token2_pct, target_token2_pct, self.token2_amount, token2_prev)

        if token1_adj != token1_prev:
            self.token1_cost = calculate_hedge_cost(
                token1_prev, token1_adj, current_token1_price, prev_value, self.transaction_cost
            )
            self.hedge_token1 = token1_adj
        else:
            self.token1_cost = 0.0

        if token2_adj != token2_prev:
            self.token2_cost = calculate_hedge_cost(
                token2_prev, token2_adj, current_token2_price, prev_value, self.transaction_cost
            )
            self.hedge_token2 = token2_adj
        else:
            self.token2_cost = 0.0

        # Store prices for PnL calculation (next-step price used below after counter increment)
        self.prev_token1_price = current_token1_price
        self.prev_token2_price = current_token2_price

        out_of_range = price < self.lp_lower_bound or price > self.lp_upper_bound

        # Advance time
        self.counter += self.step_length

        # Current prices after stepping forward
        current_token1_price = float(self.price_data[self.token1_key][self.counter])
        current_token2_price = float(self.price_data[self.token2_key][self.counter])

        hedge_pnl_token1 = calculate_hedge_pnl(self.prev_token1_price, current_token1_price, token1_prev)
        hedge_pnl_token2 = calculate_hedge_pnl(self.prev_token2_price, current_token2_price, token2_prev)

        # Funding at 8h cadence based on timestamp series
        current_timestamp = self.price_data[self.timestamp_key][self.counter]
        # Convert numpy datetime64 → python datetime (if needed)
        if hasattr(current_timestamp, "astype"):
            current_timestamp = current_timestamp.astype("M8[ms]").astype("O")

        hour = current_timestamp.hour
        minute = current_timestamp.minute

        if (hour % 8 == 0) and (minute == 0):
            eth_funding = float(self.price_data[self.token1_funding_key][self.counter])
            btc_funding = float(self.price_data[self.token2_funding_key][self.counter])
            hedge_pnl_token1 += token1_prev * eth_funding * current_token1_price
            hedge_pnl_token2 += token2_prev * btc_funding * current_token2_price

        current_hedge_pnl = hedge_pnl_token1 + hedge_pnl_token2
        self.hedge_value += current_hedge_pnl
        self.prev_hedge_pnl += current_hedge_pnl

        absolute_token1_cost = self.token1_cost * prev_value
        absolute_token2_cost = self.token2_cost * prev_value
        self.total_cost = absolute_token1_cost + absolute_token2_cost

        self.lp_value = self.calculate_lp_value(self.counter)
        self.portfolio_value = self._portfolio_value()

        pnl = self.portfolio_value - prev_value
        return_pct = pnl / prev_value

        # Optional diagnostics (kept; not currently used in reward)
        variance_lambda = 100
        _downside_variance_penalty = (return_pct**2) * variance_lambda if return_pct < 0 else 0.0
        self.recent_returns.append(return_pct)
        if len(self.recent_returns) > self.sortino_window:
            self.recent_returns.pop(0)
        _sortino_ratio = compute_rolling_sortino(self.recent_returns)

        self.cumulative_return += return_pct

        scaled_factor = 10
        reward = scaled_factor * return_pct

        self.step_counter += self.step_length
        self.episode_step += self.step_length

        terminated = bool(out_of_range)
        truncated = bool(
            (self.episode_step >= 10000)
            or (self.step_counter >= len(self.price_data[self.pair_key]) - self.step_length)
        )

        obs = self._get_obs()

        self.history["portfolio_value_norm"].append(self.portfolio_value / self.initial_value)
        self.history["hedge_token1_norm"].append(float(obs[5]))
        self.history["hedge_token2_norm"].append(float(obs[6]))
        self.history["token1_norm"].append(float(obs[0]))
        self.history["token2_norm"].append(float(obs[1]))
        self.history["token1token2_norm"].append(float(obs[2]))
        self.history["lp_token1_value_norm"].append(float(obs[3]))
        self.history["lp_token2_value_norm"].append(float(obs[4]))
        self.history["reward"].append(float(reward))
        self.history["hedge_token1_pct"].append(float(obs[7]))
        self.history["hedge_token2_pct"].append(float(obs[8]))

        return obs, float(reward), terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        idx = self.counter

        token1_price = float(self.price_data[self.token1_key][idx])
        token2_price = float(self.price_data[self.token2_key][idx])
        token1token2 = float(self.price_data[self.pair_key][idx])

        token1_price_norm = token1_price / self.initial_token1_price
        token2_price_norm = token2_price / self.initial_token2_price
        token1token2_norm = token1token2 / self.initial_token1token2_price

        # EMA features
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
        current_lp_value = lp_token1_value + lp_token2_value

        # (pct values computed in thesis code; only hedge pct values are used in observation)
        _lp_token1_pct = lp_token1_value / current_lp_value if current_lp_value != 0 else 0.0
        _lp_token2_pct = lp_token2_value / current_lp_value if current_lp_value != 0 else 0.0

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

    def calculate_lp_value(self, idx: int) -> float:
        token1_price = float(self.price_data[self.token1_key][idx])
        token2_price = float(self.price_data[self.token2_key][idx])
        token1token2 = float(self.price_data[self.pair_key][idx])

        return float(
            uniswap_v3_value(
                price=token1token2,
                lower_bound=self.lp_lower_bound,
                upper_bound=self.lp_upper_bound,
                liquidity=self.lp_liquidity,
                token1_usd_price=token1_price,
                token2_usd_price=token2_price,
            )
        )

    def _portfolio_value(self) -> float:
        idx = self.counter
        lp_val = self.calculate_lp_value(idx)
        hedge_val = self.hedge_value
        total_value = lp_val + hedge_val - self.total_cost
        return float(total_value)

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