## Setup

```bash
pip install -r requirements.txt
python simulate.py
```

## Parameters

| Parameter | Value | Description |
|---|---|---|
| S0 | $100 | Initial stock price |
| K | $100 | Strike price (ATM) |
| T | 30 days | Option expiry |
| sigma | 20% | Annualized volatility |
| mu | 5% | Stock drift |
| gamma | 0.01 | AS risk aversion parameter |
| kappa | 10.0 | Order arrival sensitivity to distance from mid |
| A | 0.001 | Noise trader arrival rate (per minute) |
| A_informed | 0.00005 | Informed trader arrival rate (per minute) |
| eta | 1.2 | Informed trader price impact multiplier |
| SKEW_WEIGHT | 0.5 | Inventory skew scaling in combined strategy |

## Connection to Theory

Poisson process (MIT 6.041 Lectures 13-15): order arrivals modeled as Poisson 
processes. The probability of at least one arrival in timestep dt is 
1 - exp(-lambda * dt), derived from the memoryless property of exponential 
inter-arrival times. Noise trader arrival rate decays exponentially with 
distance from mid; informed trader rate is constant.

Central Limit Theorem (MIT 6.041 Lectures 19-20): with 1,000 independent 
simulation runs, the distribution of mean P&L is approximately normal by CLT, 
allowing confidence interval construction via CI = mean +/- 1.96 * std / sqrt(n).

Black-Scholes: fair value and Greeks recomputed at every timestep throughout 
the full 30-day option life cycle. Delta is used for hedging; gamma and vega 
are tracked as residual risk exposures after hedging.

Geometric Brownian Motion: stock price update uses the Ito-corrected formula 
S(t+dt) = S(t) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z), ensuring 
the expected price grows at rate mu rather than mu + sigma^2/2.

## Requirements

numpy
scipy
matplotlib
