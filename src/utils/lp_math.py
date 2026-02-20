# src/utils/lp_math.py
from __future__ import annotations

import numpy as np


def uniswap_v3_composition(
    price: float,
    lower_bound: float,
    upper_bound: float,
    liquidity: float,
) -> tuple[float, float]:
    """
    Compute Uniswap v3 token amounts (token1, token2) for a given price and range.

    Uses the standard concentrated liquidity formulas in terms of sqrt price.
    Assumes:
      - price is token1/token2 (e.g., ETH/BTC)
      - token1/token2 naming matches your thesis environment
    """
    sqrt_price = np.sqrt(price)
    sqrt_lower = np.sqrt(lower_bound)
    sqrt_upper = np.sqrt(upper_bound)

    if price < lower_bound:
        # Below range: all token1
        token1_amount = liquidity * (1.0 / sqrt_lower - 1.0 / sqrt_upper)
        token2_amount = 0.0
    elif price > upper_bound:
        # Above range: all token2
        token1_amount = 0.0
        token2_amount = liquidity * (sqrt_upper - sqrt_lower)
    else:
        # In range: mixed
        token1_amount = liquidity * (1.0 / sqrt_price - 1.0 / sqrt_upper)
        token2_amount = liquidity * (sqrt_price - sqrt_lower)

    return float(token1_amount), float(token2_amount)


def rolling_volatility(price_series: np.ndarray, window: int = 5) -> float:
    """
    Rolling volatility proxy: std of last `window` log returns.
    Returns 0.0 if there is insufficient history.
    """
    price_series = np.asarray(price_series, dtype=float)

    if price_series.size < 2:
        return 0.0

    log_returns = np.diff(np.log(price_series))
    if log_returns.size < window:
        return 0.0

    return float(np.std(log_returns[-window:]))


def uniswap_v3_value(
    price: float,
    lower_bound: float,
    upper_bound: float,
    liquidity: float,
    token1_usd_price: float,
    token2_usd_price: float,
) -> float:
    """
    LP position value in USD using current token USD prices.
    """
    token1_amount, token2_amount = uniswap_v3_composition(price, lower_bound, upper_bound, liquidity)
    token1_value = token1_amount * token1_usd_price
    token2_value = token2_amount * token2_usd_price
    return float(token1_value + token2_value)


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential moving average (EMA) for a 1D price series.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        return prices

    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(prices, dtype=float)
    ema[0] = prices[0]
    for i in range(1, prices.size):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def rebalance_hedge(
    current_hedge_pct: float,
    target_hedge_pct: float,
    token_amount: float,
    prev_hedge_amount: float,
    tolerance: float = 0.1,
) -> float:
    """
    Adjust hedge amount to target percentage of the token amount if outside tolerance band.
    Returns the new hedge amount (in token units).
    """
    if abs(current_hedge_pct - target_hedge_pct) > tolerance:
        return float(target_hedge_pct * token_amount)
    return float(prev_hedge_amount)


def calculate_hedge_cost(
    previous_hedge_amount: float,
    current_hedge_amount: float,
    current_price: float,
    prev_portfolio_value: float,
    transaction_cost: float,
) -> float:
    """
    Compute normalized transaction cost for changing hedge position.
    Returns a cost as a fraction of previous portfolio value.
    """
    hedge_change = abs(current_hedge_amount - previous_hedge_amount)
    hedge_cost_usd = hedge_change * current_price * transaction_cost
    if prev_portfolio_value == 0:
        return 0.0
    return float(hedge_cost_usd / prev_portfolio_value)


def calculate_hedge_pnl(previous_price: float, current_price: float, hedge_amount: float) -> float:
    """
    Hedge PnL with your thesis sign convention.
    (Kept exactly as your original implementation.)
    """
    return float((previous_price - current_price) * hedge_amount)


def compute_rolling_sortino(returns, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sortino ratio over a window of returns.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0

    downside_returns = returns[returns < risk_free_rate]
    downside_std = float(downside_returns.std()) if downside_returns.size > 0 else 0.0
    avg_return = float(returns.mean())

    if downside_std == 0.0:
        return 0.0

    return float((avg_return - risk_free_rate) / downside_std)