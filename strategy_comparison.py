"""
strategy_comparison.py
=======================
Compare Aggressive / Moderate / Delay launch strategies.

Share capture uncertainty propagates through every scenario via the
stochastic share-scale multiplier in monte_carlo.py. Competitive response
(triggered when Year-1 share exceeds 8%) activates in ~56% of scenarios
for the Aggressive strategy.

Three outputs:
1. Full distributional comparison table (mean/median/P10/P90/P(destruction))
2. Overlaid histogram showing all three strategies' NPV distributions
3. Risk-return scatter (mean NPV vs P10 downside) for visual strategy selection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from monte_carlo import run_simulation

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid']   = True
plt.rcParams['grid.alpha']  = 0.25
plt.rcParams['figure.dpi']  = 140

STRATEGY_COLORS = {
    "Aggressive": "#C00000",
    "Moderate":   "#2E75B6",
    "Delay":      "#70AD47",
}


def run_strategy_comparison(N: int = 10_000, seed: int = 42) -> dict:
    """
    Run all three strategies through the corrected MC engine and
    build the full comparison table.
    """
    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    all_results = {}
    for strat in ["Aggressive", "Moderate", "Delay"]:
        print(f"\nRunning: {strat}...")
        all_results[strat] = run_simulation(N=N, seed=seed, strategy=strat, verbose=True)

    print("\n" + "=" * 70)
    print("FULL COMPARISON TABLE")
    print("=" * 70)

    metrics = [
        ("Mean NPV (₹ Cr)",            "mean_npv",            "₹{:,.0f}"),
        ("Median NPV / P50 (₹ Cr)",    "median_npv",          "₹{:,.0f}"),
        ("Std Deviation (₹ Cr)",       "std_npv",             "₹{:,.0f}"),
        ("P10 — Downside (₹ Cr)",      "p10_npv",             "₹{:,.0f}"),
        ("P25 (₹ Cr)",                 "p25_npv",             "₹{:,.0f}"),
        ("P75 (₹ Cr)",                 "p75_npv",             "₹{:,.0f}"),
        ("P90 — Upside (₹ Cr)",        "p90_npv",             "₹{:,.0f}"),
        ("P(Value Destruction)",       "p_value_destruction", "{:.1%}"),
        ("P(Cannibalization Swing)",   "p_cannibal_swing",    "{:.1%}"),
        ("P(Competitive Response)",    "p_comp_response",     "{:.1%}"),
    ]

    print(f"\n{'Metric':<28} {'Aggressive':>16} {'Moderate':>16} {'Delay':>16}")
    print("─" * 78)
    table_rows = []
    for label, key, fmt in metrics:
        row_vals = {}
        line = f"  {label:<26}"
        for strat in ["Aggressive", "Moderate", "Delay"]:
            v = all_results[strat]["summary"][key]
            row_vals[strat] = v
            line += f"  {fmt.format(v):>16}"
        print(line)
        table_rows.append({"metric": label, **row_vals})

    # Risk-adjusted ranking: Sharpe-like ratio = mean / std
    print(f"\n{'Risk-Adjusted Metrics':<28}")
    print("─" * 78)
    for strat in ["Aggressive", "Moderate", "Delay"]:
        s = all_results[strat]["summary"]
        sharpe_like = s["mean_npv"] / s["std_npv"] if s["std_npv"] > 0 else float('nan')
        print(f"  {strat:<26}  Return/Risk ratio: {sharpe_like:.3f}   "
              f"(Mean ₹{s['mean_npv']:,.0f}Cr / Std ₹{s['std_npv']:,.0f}Cr)")

    return {"all_results": all_results, "table_rows": table_rows}


def plot_strategy_distributions(
    all_results: dict,
    save_path: str = "outputs/strategy_comparison_histogram.png",
) -> None:
    """Overlaid histograms of NPV distributions for all three strategies."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for strat in ["Delay", "Moderate", "Aggressive"]:   # draw order: back to front
        npv = all_results[strat]["npv_with_tv"]
        ax.hist(npv, bins=50, alpha=0.45, color=STRATEGY_COLORS[strat],
                label=f'{strat} (median ₹{np.median(npv):,.0f}Cr)',
                edgecolor='white', linewidth=0.3)

    ax.axvline(0, color='black', linewidth=1.3, linestyle='-')
    ax.set_xlabel('NPV (₹ Crore)', fontsize=11)
    ax.set_ylabel('Number of Scenarios', fontsize=11)
    ax.set_title('NPV Distribution by Launch Strategy\n(N=10,000 scenarios each, corrected share-uncertainty engine)',
                fontsize=12.5, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")


def plot_risk_return(
    all_results: dict,
    save_path: str = "outputs/strategy_risk_return.png",
) -> None:
    """
    Risk-return scatter: Mean NPV (y) vs P10 downside (x).
    This is the chart that actually drives the strategy recommendation —
    shows the tradeoff explicitly rather than just ranking by mean.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    offsets = {"Aggressive": (15, -55), "Moderate": (15, 15), "Delay": (15, 15)}
    for strat in ["Aggressive", "Moderate", "Delay"]:
        s = all_results[strat]["summary"]
        ax.scatter(s["p10_npv"], s["mean_npv"], s=400, color=STRATEGY_COLORS[strat],
                  edgecolor='black', linewidth=1.5, zorder=5, label=strat)
        ax.annotate(
            f"{strat}\nMean: ₹{s['mean_npv']:,.0f}Cr\nP10: ₹{s['p10_npv']:,.0f}Cr\n"
            f"P(loss): {s['p_value_destruction']:.0%}",
            xy=(s["p10_npv"], s["mean_npv"]),
            xytext=offsets[strat], textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=STRATEGY_COLORS[strat], alpha=0.9)
        )

    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('P10 NPV — Downside Risk (₹ Crore)', fontsize=11)
    ax.set_ylabel('Mean NPV — Expected Value (₹ Crore)', fontsize=11)
    ax.set_title('Risk-Return Tradeoff Across Launch Strategies',
                fontsize=13, fontweight='bold', pad=16)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))

    # Add headroom above the highest point so annotations never collide with the title
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * 1.18)
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.95, ylim[1]*0.95, 'High return,\nlow downside risk',
            ha='right', va='top', fontsize=8.5, color='green', style='italic', alpha=0.7)
    ax.text(xlim[0]*0.95, ylim[0]*0.95, 'Low return,\nhigh downside risk',
            ha='left', va='bottom', fontsize=8.5, color='red', style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    comparison = run_strategy_comparison(N=10_000, seed=42)

    print("\nGenerating distribution comparison histogram...")
    plot_strategy_distributions(comparison["all_results"])

    print("Generating risk-return scatter...")
    plot_risk_return(comparison["all_results"])

    # Interview-ready synthesis
    print("\n" + "=" * 70)
    print("HOUR 1 — INTERVIEW-READY FINDING")
    print("=" * 70)
    ar = comparison["all_results"]
    agg, mod, delay = ar["Aggressive"]["summary"], ar["Moderate"]["summary"], ar["Delay"]["summary"]
    print(f"""
  Aggressive has the highest mean NPV (₹{agg['mean_npv']:,.0f} Cr) but also the
  highest downside risk (P10 = ₹{agg['p10_npv']:,.0f} Cr, P(loss) = {agg['p_value_destruction']:.0%}).

  Moderate offers a lower mean (₹{mod['mean_npv']:,.0f} Cr) with a less severe downside
  (P10 = ₹{mod['p10_npv']:,.0f} Cr, P(loss) = {mod['p_value_destruction']:.0%}) — roughly half the capex risk
  for a smaller but still substantial expected return.

  Delay has the weakest case across the board (mean ₹{delay['mean_npv']:,.0f} Cr,
  P(loss) = {delay['p_value_destruction']:.0%}) — moving slowly sacrifices the network-effect
  upside without meaningfully reducing capital risk, since capex per
  subscriber is unchanged.

  This is a genuine risk-adjusted tradeoff, not an obvious choice —
  exactly the kind of judgment call a strategy team is paid to make.
""")
