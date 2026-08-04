import numpy as np
from black_scholes import black_scholes_call

# ── PARAMETERS ────────────────────────────────────────────────────────────────
GAMMA = 0.5    # risk aversion — higher = wider spreads, more aggressive skew
K_PARAM = 1.5  # order arrival sensitivity to price distance from mid
A = 10.0        # baseline order arrival intensity (orders per second)
# ─────────────────────────────────────────────────────────────────────────────


def reservation_price(mid, inventory, gamma, sigma, T_remaining):
    """
    The price the market maker would be indifferent to trading at,
    given current inventory.

    r = mid - inventory * gamma * sigma^2 * T_remaining

    When inventory is positive (long), reservation price is below mid —
    the market maker wants to sell, so they lower their quotes to attract
    buyers. When inventory is negative (short), reservation price is above
    mid — they want to buy, so they raise their quotes.

    Args:
        mid         : current fair value of the option (from Black-Scholes)
        inventory   : current number of contracts held (positive = long)
        gamma       : risk aversion parameter
        sigma       : volatility of the underlying
        T_remaining : time left in trading day (years)

    Returns:
        r : reservation price
    """
    return mid - inventory * gamma * sigma**2 * T_remaining


def optimal_spread(gamma, sigma, T_remaining, k):
    """
    The optimal bid-ask spread derived from the AS framework.

    delta = gamma * sigma^2 * T_remaining + (2/gamma) * ln(1 + gamma/k)

    First term: widens with volatility, risk aversion, and time remaining
    (more uncertainty = wider spread needed to compensate).
    Second term: accounts for order arrival dynamics — how sensitive
    arrivals are to your distance from mid.

    Args:
        gamma       : risk aversion parameter
        sigma       : volatility of the underlying
        T_remaining : time left in trading day (years)
        k           : order arrival sensitivity parameter

    Returns:
        spread : total bid-ask spread in dollars
    """
    return (gamma * sigma**2 * T_remaining +
            (2 / gamma) * np.log(1 + gamma / k))


def compute_quotes(mid, inventory, gamma, sigma, T_remaining, k):
    """
    Computes the bid and ask quotes for the market maker.

    Bid = reservation_price - spread/2
    Ask = reservation_price + spread/2

    The spread is centered on the reservation price (not mid),
    which is what gives the inventory strategy its edge over
    the symmetric strategy.

    Returns:
        bid, ask, reservation_price, spread
    """
    r      = reservation_price(mid, inventory, gamma, sigma, T_remaining)
    spread = optimal_spread(gamma, sigma, T_remaining, k)

    bid = r - spread / 2
    ask = r + spread / 2

    return bid, ask, r, spread


if __name__ == "__main__":
    # test the quoting engine at different inventory levels
    S        = 100.0
    K_strike = 100.0
    T_expiry = 30/365
    r_rate   = 0.05
    sigma    = 0.20
    T_remaining = 1  # one trading day remaining in session

    price, delta, gamma, vega, theta = black_scholes_call(
        S, K_strike, T_expiry, r_rate, sigma)

    print(f"Option fair value (mid): ${price:.4f}")
    print(f"Delta: {delta:.4f}  Gamma: {gamma:.4f}\n")

    print(f"{'Inventory':>12} {'Reservation':>14} {'Bid':>10} {'Ask':>10} {'Spread':>10}")
    print("-" * 60)

    for inventory in [-5, -3, -1, 0, 1, 3, 5]:
        bid, ask, r, spread = compute_quotes(
            price, inventory, GAMMA, sigma, T_remaining, K_PARAM)
        print(f"{inventory:>12}   ${r:>10.4f}   ${bid:>7.4f}   ${ask:>7.4f}   ${spread:>7.4f}")

def simulate_order_arrivals(bid, ask, mid, A, k, dt):
    """
    Simulates whether a buy or sell order arrives in this timestep
    using a Poisson process.

    In the AS model, order arrival intensity depends on how far your
    quote is from the fair value (mid). The further away, the less
    likely someone hits your quote:

        lambda_bid = A * exp(-k * (mid - bid))  # arrival rate for buy orders
        lambda_ask = A * exp(-k * (ask - mid))  # arrival rate for sell orders

    The probability of at least one arrival in timestep dt is:
        P(arrival) = 1 - exp(-lambda * dt)

    This comes directly from the Poisson process: if events arrive at
    rate lambda, the probability of zero arrivals in time dt is exp(-lambda*dt),
    so the probability of at least one arrival is 1 - exp(-lambda*dt).

    Args:
        bid, ask : current quotes
        mid      : current fair value
        A        : baseline arrival intensity
        k        : sensitivity of arrivals to distance from mid
        dt       : timestep size

    Returns:
        buy_arrival  : True if a buy order hit our ask
        sell_arrival : True if a sell order hit our bid
    """
    # distance from mid to our quotes
    delta_bid = mid - bid   # how far our bid is below mid
    delta_ask = ask - mid   # how far our ask is above mid

    # arrival rates — decay exponentially with distance from mid
    lambda_bid = A * np.exp(-k * delta_bid)
    lambda_ask = A * np.exp(-k * delta_ask)

    # probability of at least one arrival this timestep
    prob_bid = 1 - np.exp(-lambda_bid * dt)
    prob_ask = 1 - np.exp(-lambda_ask * dt)

    # simulate arrivals
    sell_arrival = np.random.random() < prob_bid   # someone sold to us at bid
    buy_arrival  = np.random.random() < prob_ask   # someone bought from us at ask

    return buy_arrival, sell_arrival

def simulate_informed_arrivals(A_informed, dt):
    """
    Simulates whether an informed trader arrives this timestep.
    
    Informed traders arrive at a constant rate regardless of quote distance —
    they trade because they have information, not because of price attractiveness.
    
    Unlike noise traders, informed traders always trade on the side that
    hurts the market maker — they buy when they know price is going up,
    sell when they know price is going down.
    
    Args:
        A_informed : arrival intensity of informed traders (per second)
                     much lower than noise trader intensity
        dt         : timestep in seconds
    
    Returns:
        informed_buy  : True if informed trader bought from us (lifted ask)
        informed_sell : True if informed trader sold to us (hit bid)
        direction     : +1 if price will go up, -1 if price will go down
    """
    # probability of any informed arrival this timestep
    prob_informed = 1 - np.exp(-A_informed * dt)
    
    if np.random.random() < prob_informed:
        # informed trader arrived — determine direction
        # +1 means they know price is going up (they'll buy from us)
        # -1 means they know price is going down (they'll sell to us)
        direction = np.random.choice([1, -1])
        
        if direction == 1:
            return True, False, direction   # informed buy
        else:
            return False, True, direction   # informed sell
    
    return False, False, 0

if __name__ == "__main__":
    # ... keep existing test code, add this below it ...

    print("\n\nOrder Arrival Model Test")
    print("=" * 50)

    # test at different quote distances from mid
    mid    = 2.4934
    A_test = 0.1
    k_test = 1.5
    dt     = 1   # one second

    print(f"\n{'Distance from mid':>20} {'Arrival rate':>15} {'Prob per second':>18}")
    print("-" * 55)
    for distance in [0.0, 0.1, 0.2, 0.5, 1.0]:
        lam  = A_test * np.exp(-k_test * distance)
        prob = 1 - np.exp(-lam * dt)
        print(f"  ${distance:>6.2f}             {lam:>10.4f}          {prob:>16.10f}")

    # simulate 10000 timesteps and count arrivals
    print("\nSimulating 10,000 timesteps at mid quotes (distance=0):")
    bid_test = mid - 0.10
    ask_test = mid + 0.10
    buy_count, sell_count = 0, 0
    for _ in range(10000):
        b, s = simulate_order_arrivals(bid_test, ask_test, mid, A_test, k_test, dt)
        buy_count  += b
        sell_count += s
    print(f"  Buy arrivals : {buy_count}")
    print(f"  Sell arrivals: {sell_count}")
    print(f"  Expected ~{int(10000 * (1 - np.exp(-A_test * np.exp(-k_test * 0.10) * dt)))} each")
