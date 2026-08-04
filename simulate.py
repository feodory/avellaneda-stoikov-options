import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_call
from market_maker import compute_quotes, simulate_order_arrivals, simulate_informed_arrivals, optimal_spread

# ── PARAMETERS ────────────────────────────────────────────────────────────────
S0        = 100.0
K_STRIKE  = 100.0
T_EXPIRY  = 30/365
R_RATE    = 0.05
SIGMA     = 0.20
MU        = 0.05

# Simulation — 30 days at minute resolution
N_DAYS    = 30
N_STEPS   = N_DAYS * 390        # 11,700 timesteps
DT_YEARS  = 1/252/390           # one minute in years
DT_SEC    = 60.0                # one minute in seconds

# Market maker
GAMMA      = 0.01
K_PARAM    = 10.0
A          = 0.001              # noise trader arrivals per minute
A_INFORMED = 0.00005            # informed trader arrivals per minute
ETA        = 1.2
MAX_INV    = 10
SKEW_WEIGHT = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(seed=None, strategy='as', hedge=True):
    """
    Runs a full 30-day simulation at minute resolution.

    The option runs its full life cycle from 30 days to expiry,
    capturing theta decay, delta evolution, and accumulated drift.

    Args:
        seed     : random seed for reproducibility
        strategy : 'as'        — inventory-aware quoting
                   'symmetric' — naive quoting centered on mid
        hedge    : True  — delta hedge after every timestep
                   False — no delta hedging

    Returns:
        results : dict of time series arrays and summary statistics
    """
    if seed is not None:
        np.random.seed(seed)

    stock_price = S0
    inventory   = 0
    cash        = 0.0
    stock_hedge = 0.0

    T_session  = 1.0
    dt_session = 1.0 / N_STEPS

    stock_prices  = np.zeros(N_STEPS + 1)
    option_mids   = np.zeros(N_STEPS + 1)
    inventories   = np.zeros(N_STEPS + 1)
    pnls          = np.zeros(N_STEPS + 1)
    bids          = np.zeros(N_STEPS + 1)
    asks          = np.zeros(N_STEPS + 1)
    deltas        = np.zeros(N_STEPS + 1)
    stock_hedges  = np.zeros(N_STEPS + 1)

    stock_prices[0] = S0

    T_to_expiry = T_EXPIRY
    price, delta, gm, vega, theta = black_scholes_call(
        S0, K_STRIKE, T_to_expiry, R_RATE, SIGMA)
    option_mids[0] = price
    deltas[0]      = delta

    for i in range(N_STEPS):
        # ── 1. STOCK PRICE MOVES ─────────────────────────────────────────────
        Z = np.random.standard_normal()
        stock_price = stock_price * np.exp(
            (MU - 0.5 * SIGMA**2) * DT_YEARS + SIGMA * np.sqrt(DT_YEARS) * Z
        )

        # ── 2. RECOMPUTE OPTION FAIR VALUE ───────────────────────────────────
        T_to_expiry = max(T_EXPIRY - (i + 1) * DT_YEARS, 1e-6)
        mid, delta, gm, vega, theta = black_scholes_call(
            stock_price, K_STRIKE, T_to_expiry, R_RATE, SIGMA)

        # ── 3. COMPUTE QUOTES ────────────────────────────────────────────────
        T_session_remaining = max(T_session - (i + 1) * dt_session, 1e-6)
        spread = optimal_spread(GAMMA, SIGMA, T_session_remaining, K_PARAM)

        if strategy == 'as':
            effective_inventory = inventory * SKEW_WEIGHT if hedge else inventory
            r   = mid - effective_inventory * GAMMA * SIGMA**2 * T_session_remaining
            bid = r - spread / 2
            ask = r + spread / 2
        else:
            bid = mid - spread / 2
            ask = mid + spread / 2

        # enforce minimum spread
        bid = min(bid, mid - 0.005)
        ask = max(ask, mid + 0.005)

        # ── 4. SIMULATE ORDER ARRIVALS ───────────────────────────────────────
        buy_arrival, sell_arrival = simulate_order_arrivals(
            bid, ask, mid, A, K_PARAM, DT_SEC)

        informed_buy, informed_sell, inf_direction = simulate_informed_arrivals(
            A_INFORMED, DT_SEC)

        # ── 5. UPDATE INVENTORY AND CASH ─────────────────────────────────────
        if buy_arrival and inventory > -MAX_INV:
            inventory -= 1
            cash      += ask

        if sell_arrival and inventory < MAX_INV:
            inventory += 1
            cash      -= bid

        if informed_buy and inventory > -MAX_INV:
            inventory -= 1
            cash      += ask
            stock_price *= (1 + ETA * SIGMA / np.sqrt(252))

        if informed_sell and inventory < MAX_INV:
            inventory += 1
            cash      -= bid
            stock_price *= (1 - ETA * SIGMA / np.sqrt(252))

        # ── 5b. DELTA HEDGE ──────────────────────────────────────────────────
        if hedge:
            target_stock_hedge = -inventory * delta
            hedge_trade        = target_stock_hedge - stock_hedge
            stock_hedge        = target_stock_hedge
            cash              -= hedge_trade * stock_price

        # ── 6. MARK TO MARKET P&L ────────────────────────────────────────────
        pnl = cash + inventory * mid + stock_hedge * stock_price

        stock_prices[i+1]  = stock_price
        option_mids[i+1]   = mid
        inventories[i+1]   = inventory
        pnls[i+1]          = pnl
        bids[i+1]          = bid
        asks[i+1]          = ask
        deltas[i+1]        = delta
        stock_hedges[i+1]  = stock_hedge

    return {
        'stock_prices' : stock_prices,
        'option_mids'  : option_mids,
        'inventories'  : inventories,
        'pnls'         : pnls,
        'bids'         : bids,
        'asks'         : asks,
        'deltas'       : deltas,
        'stock_hedges' : stock_hedges,
        'final_pnl'    : pnls[-1],
        'final_inv'    : inventories[-1],
        'n_trades'     : int(np.sum(np.diff(inventories) != 0)),
    }


def run_many(n_sims=1000, strategy='as', hedge=True):
    label = f"{strategy.upper()} {'+ hedge' if hedge else 'no hedge'}"
    pnls = []
    for i in range(n_sims):
        if i % 50 == 0:
            print(f"  {label}: {i}/{n_sims}")
        r = run_simulation(strategy=strategy, hedge=hedge)
        pnls.append(r['final_pnl'])
    return np.array(pnls)


def analyze(pnls, strategy_name):
    n       = len(pnls)
    mean    = np.mean(pnls)
    std     = np.std(pnls)
    se      = std / np.sqrt(n)
    ci_low  = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    sharpe  = mean / std if std > 0 else 0

    print(f"\n{strategy_name}:")
    print(f"  Mean P&L   : ${mean:.4f}")
    print(f"  Std P&L    : ${std:.4f}")
    print(f"  95% CI     : ${ci_low:.4f} – ${ci_high:.4f}")
    print(f"  Sharpe     : {sharpe:.4f}")
    print(f"  % Positive : {100 * np.mean(pnls > 0):.1f}%")

    return {'mean': mean, 'std': std, 'ci_low': ci_low,
            'ci_high': ci_high, 'sharpe': sharpe}


if __name__ == "__main__":
    N_SIMS = 1000   # change to 1000 for final run

    print(f"Running {N_SIMS} simulations for each strategy...\n")

    naive_pnls    = run_many(N_SIMS, strategy='symmetric', hedge=False)
    hedge_pnls    = run_many(N_SIMS, strategy='symmetric', hedge=True)
    inv_pnls      = run_many(N_SIMS, strategy='as',        hedge=False)
    combined_pnls = run_many(N_SIMS, strategy='as',        hedge=True)

    np.save('naive_pnls.npy',    naive_pnls)
    np.save('hedge_pnls.npy',    hedge_pnls)
    np.save('inv_pnls.npy',      inv_pnls)
    np.save('combined_pnls.npy', combined_pnls)

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)

    naive_stats    = analyze(naive_pnls,    "Naive Baseline (no hedge, symmetric)")
    hedge_stats    = analyze(hedge_pnls,    "Delta Hedge Only (symmetric + hedge)")
    inv_stats      = analyze(inv_pnls,      "Inventory Management Only (AS, no hedge)")
    combined_stats = analyze(combined_pnls, "Combined (AS + hedge, balanced)")

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_pnls = np.concatenate([naive_pnls, hedge_pnls, inv_pnls, combined_pnls])
    bins     = np.linspace(all_pnls.min(), all_pnls.max(), 50)

    axes[0].hist(naive_pnls,    bins=bins, alpha=0.5, color='gray',
                 label=f'Naive (mean=${naive_stats["mean"]:.2f})')
    axes[0].hist(hedge_pnls,    bins=bins, alpha=0.5, color='blue',
                 label=f'Delta Hedge Only (mean=${hedge_stats["mean"]:.2f})')
    axes[0].hist(inv_pnls,      bins=bins, alpha=0.5, color='orange',
                 label=f'Inventory Only (mean=${inv_stats["mean"]:.2f})')
    axes[0].hist(combined_pnls, bins=bins, alpha=0.5, color='green',
                 label=f'Combined (mean=${combined_stats["mean"]:.2f})')
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].set_xlabel("Final P&L ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"P&L Distribution ({N_SIMS} simulations each)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    strategies = ['Naive', 'Delta Hedge\nOnly', 'Inventory\nOnly', 'Combined']
    means      = [naive_stats['mean'], hedge_stats['mean'],
                  inv_stats['mean'], combined_stats['mean']]
    stds       = [naive_stats['std'], hedge_stats['std'],
                  inv_stats['std'], combined_stats['std']]
    colors     = ['gray', 'blue', 'orange', 'green']

    bars = axes[1].bar(strategies, means, yerr=stds, capsize=8,
                       color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_ylabel("Mean P&L ($)")
    axes[1].set_title("Mean P&L ± Std Dev")
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars, means):
        ypos = mean + 1 if mean >= 0 else mean - 3
        axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                    f'${mean:.2f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("\nPlot saved to comparison.png")
