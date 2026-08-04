import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Computes the fair value and Greeks of a European call option.

    Args:
        S     : current stock price
        K     : strike price
        T     : time to expiry in years
        r     : risk-free rate (annualized)
        sigma : volatility (annualized)

    Returns:
        price : fair value of the call
        delta : dV/dS — sensitivity to stock price
        gamma : d²V/dS² — rate of change of delta
        vega  : dV/dsigma — sensitivity to volatility
        theta : dV/dT — time decay (per year)
    """
    if T <= 0:
        # at expiry, option is worth max(S-K, 0)
        price = max(S - K, 0)
        return price, 1.0 if S > K else 0.0, 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))

    return price, delta, gamma, vega, theta


if __name__ == "__main__":
    # test with a standard at-the-money call
    S     = 100.0   # stock price
    K     = 100.0   # strike (at the money)
    T     = 30/365  # 30 days to expiry
    r     = 0.05    # 5% risk-free rate
    sigma = 0.20    # 20% volatility

    price, delta, gamma, vega, theta = black_scholes_call(S, K, T, r, sigma)

    print("Black-Scholes Call Option")
    print(f"  Stock price  : ${S:.2f}")
    print(f"  Strike       : ${K:.2f}")
    print(f"  Time to expiry: {T*365:.0f} days")
    print(f"\nFair value : ${price:.4f}")
    print(f"Delta      : {delta:.4f}")
    print(f"Gamma      : {gamma:.4f}")
    print(f"Vega       : {vega:.4f}")
    print(f"Theta      : {theta:.4f} per year (${theta/365:.4f} per day)")

    print("\nSanity checks:")
    print(f"  ATM delta should be ~0.50 : {delta:.4f}")
    print(f"  Vega should be positive   : {vega:.4f}")
    print(f"  Theta should be negative  : {theta:.4f}")

    # show how price changes as stock moves
    print("\nOption price as stock moves:")
    print(f"  {'Stock':>8} {'Option':>8} {'Delta':>8}")
    for s in [90, 95, 100, 105, 110]:
        p, d, _, _, _ = black_scholes_call(s, K, T, r, sigma)
        print(f"  ${s:>6.0f}   ${p:>6.4f}   {d:>6.4f}")
