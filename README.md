

# Uniswap DRL Hedging

This repository contains the codebase for a master's thesis on **data-driven hedging of Uniswap v3 liquidity provision** using **Deep Reinforcement Learning (DRL)**.

The project studies how a Uniswap v3 liquidity provider (LP) position can be dynamically hedged using futures, and compares learned strategies against simple rule-based baselines under realistic transaction costs and funding payments.

---

## Motivation

Providing liquidity on Uniswap v3 exposes LPs to complex and path-dependent risk.
Hedging such positions requires making sequential decisions in a stochastic and high-dimensional environment.

This project frames LP hedging as a **sequential decision-making problem under uncertainty** and applies **deep reinforcement learning** to learn adaptive hedging strategies directly from data.

---

## High-level Approach

1. A custom **Gymnasium environment** simulates a Uniswap v3 LP position with hedging via futures.
2. **PPO (Proximal Policy Optimization)** agents learn hedge adjustments over time.
3. Models are selected using a validation period and evaluated on a held-out test period.
4. Learned strategies are compared against **rule-based hedging baselines**.

---

## Repository Structure
```
uniswap-drl-hedging/
│
├── src/
│ ├── envs/ # Gymnasium environments
│ │ ├── uniswap_env.py # RL environment
│ │ └── baseline_env.py # Baseline-compatible environment
│ │
│ ├── baselines/ # Rule-based hedging strategies
│ │ └── hedging_baselines.py
│ │
│ ├── utils/ # LP math and helper functions
│ │ └── lp_math.py
│ │
│ ├── train.py # PPO training
│ ├── validate.py # Model selection (validation period)
│ ├── evaluate.py # Final evaluation (test period)
│ └── run_baselines.py # Baseline evaluation
│
├── data/ # CSV price & funding data (not included)
├── checkpoints/ # Training checkpoints (generated)
├── best_checkpoints/ # Selected models (generated)
├── requirements.txt
└── README.md
```

---

## Data

The project expects historical cryptocurrency price and funding data sampled at 15-minute frequency.

Required CSV files (not included in the repository):
- `ethusdt_15min.csv`
- `btcusdt_15min.csv`
- `ethbtc_15min.csv`
- `eth_funding_15min.csv`
- `btc_funding_15min.csv`

See `data/README.md` for details.

---

## Baseline Strategies

Two non-learning baselines are implemented for comparison:

- **Fixed-frequency hedging**  
  Rebalances hedge positions at a fixed time interval.

- **Threshold-based hedging**  
  Rebalances only when the hedge ratio deviates beyond a predefined threshold.

Baselines are evaluated on the same validation and test periods as the DRL agents.

---

## Experimental Pipeline

The typical workflow is:

1. **Train agents**
   ```bash
   python -m src.train
   ```

2. Select best checkpoints (validation period)
    ```bash
    python -m src.validate
    ```

3. Evaluate on test period
    ```bash
    python -m src.evaluate
    ```
4. Run baselines
    ```bash
    python -m src.run_baselines
    ```

## Reproducibility

Fixed random seeds are used for training, validation, and evaluation.
Validation and test periods are strictly separated.
All strategies are evaluated with identical episode lengths and market data.

## Disclaimer

This repository is provided for research and educational purposes only and does not constitute financial or investment advice.


