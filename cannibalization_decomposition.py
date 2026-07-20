"""
cannibalization_decomposition.py
============================================
The core novelty of this project. Most market-entry models stop at
"here's the NPV distribution." This module answers a sharper question:

  "In how many scenarios does cannibalization ALONE determine whether
   the investment creates or destroys value?"

Method
------
For every Monte Carlo scenario i, we have two NPV figures:
  npv_with_tv[i]            — actual NPV including cannibalization effects
  npv_ex_all_losses[i]      — counterfactual NPV if cannibalization/erosion = 0

A scenario is a "cannibalization swing" scenario if:
  npv_with_tv[i] < 0  AND  npv_ex_all_losses[i] > 0

This means: absent cannibalization, the deal creates value. WITH
cannibalization, it destroys value. Cannibalization is the deciding factor.

We further decompose by segment (urban prepaid vs postpaid vs rural) to
show WHICH segment's cannibalization is doing the damage — this is the
analysis a Bain or ZS associate would actually build.

Outputs
-------
1. Waterfall chart: Gross Jio Revenue → minus segment cannibalization →
   minus comp response → Net NPV. Shows where value leaks.
2. Swing scenario count and characteristics (which inputs differ in
   swing vs non-swing scenarios).
3. Decomposition by segment: how much of total cannibalization cost
   comes from each of the three segments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from monte_carlo import run_simulation
from npv_engine import compute_npv

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid']   = True
plt.rcParams['grid.alpha']  = 0.25
plt.rcParams['figure.dpi']  = 140


# ── 1. SWING SCENARIO ANALYSIS ────────────────────────────────────────────────
def analyze_swing_scenarios(results: dict) -> dict:
    """
    Identify and characterize scenarios where cannibalization alone
    flips the NPV decision.
    """
    npv          = results["npv_with_tv"]
    npv_ex_loss  = results["npv_ex_all_losses"]
    swing_mask   = results["cannibalization_swing"]

    n_total = len(npv)
    n_swing = swing_mask.sum()
    pct_swing = n_swing / n_total

    print("\n" + "=" * 60)
    print("CANNIBALIZATION SWING ANALYSIS")
    print("=" * 60)
    print(f"  Total scenarios:                {n_total:,}")
    print(f"  Cannibalization-swing scenarios: {n_swing:,}  ({pct_swing:.1%})")
    print(f"  → In {pct_swing:.1%} of simulated futures, cannibalization risk")
    print(f"    ALONE is what separates a value-creating deal from a")
    print(f"    value-destroying one.")

    # Compare input characteristics: swing vs non-swing scenarios
    input_vars = [
        ("input_cannibal_up",   "Cannibalization — Urban Prepaid"),
        ("input_cannibal_pp",   "Cannibalization — Urban Postpaid"),
        ("input_cannibal_rural","Cannibalization — Rural"),
        ("input_share_up",      "Jio Share Capture"),
        ("input_jio_arpu",      "Jio ARPU Y2+"),
        ("input_capex_mult",    "Capex Overrun Multiplier"),
        ("input_wacc",          "WACC"),
    ]

    print(f"\n  {'Variable':<35} {'Mean (Swing)':>14} {'Mean (Non-swing)':>18} {'Δ':>10}")
    print("  " + "─" * 80)
    comparison = {}
    for key, label in input_vars:
        if key in results:
            swing_mean    = results[key][swing_mask].mean() if n_swing > 0 else np.nan
            nonswing_mean = results[key][~swing_mask].mean()
            delta = swing_mean - nonswing_mean
            comparison[label] = {"swing": swing_mean, "nonswing": nonswing_mean, "delta": delta}
            print(f"  {label:<35} {swing_mean:>14.4f} {nonswing_mean:>18.4f} {delta:>+10.4f}")

    return {
        "n_total": n_total, "n_swing": n_swing, "pct_swing": pct_swing,
        "comparison": comparison,
    }


# ── 2. SEGMENT-LEVEL CANNIBALIZATION DECOMPOSITION ────────────────────────────
def decompose_by_segment(
    cannibal_up:   float = 0.45,
    cannibal_pp:   float = 0.10,
    cannibal_rural:float = 0.25,
) -> dict:
    """
    For a given scenario, decompose total cannibalization cost by segment.
    Uses base-case point estimates by default; can be called with any
    scenario's draws to decompose that specific scenario.
    """
    from assumptions import SUBS, ARPU_PRE, SHARE_BY_YEAR, SEGMENTS

    seg_cannibal = {"urban_postpaid": cannibal_pp, "urban_prepaid": cannibal_up,
                     "rural_prepaid": cannibal_rural}

    # Sum across 8 years, segment by segment, discounted at base WACC
    wacc = 0.11
    seg_pv_loss = {seg: 0.0 for seg in SEGMENTS}

    for yr in range(1, 9):
        df = 1 / (1 + wacc) ** yr
        for seg in SEGMENTS:
            jio_subs_seg = SUBS[seg] * SHARE_BY_YEAR[yr][seg]
            cannibal_subs = jio_subs_seg * seg_cannibal[seg]
            loss_cr = cannibal_subs * ARPU_PRE[seg] * 1.2   # ₹/user/mo × 12mo → Cr
            seg_pv_loss[seg] += loss_cr * df

    total = sum(seg_pv_loss.values())
    pct_of_total = {seg: (v / total * 100 if total > 0 else 0) for seg, v in seg_pv_loss.items()}

    print("\n" + "=" * 60)
    print("CANNIBALIZATION COST DECOMPOSITION BY SEGMENT (Base Case, PV)")
    print("=" * 60)
    seg_labels = {"urban_postpaid": "Urban Postpaid", "urban_prepaid": "Urban Prepaid",
                   "rural_prepaid": "Rural Prepaid"}
    for seg in SEGMENTS:
        print(f"  {seg_labels[seg]:<20} ₹{seg_pv_loss[seg]:>10,.0f} Cr   "
              f"({pct_of_total[seg]:.1f}% of total cannibalization cost)")
    print(f"  {'TOTAL':<20} ₹{total:>10,.0f} Cr")

    return {"seg_pv_loss": seg_pv_loss, "pct_of_total": pct_of_total, "total": total}


def plot_segment_decomposition(
    decomp: dict,
    save_path: str = "outputs/cannibalization_by_segment.png",
) -> None:
    """Bar chart showing cannibalization cost contribution by segment."""
    seg_labels = {"urban_postpaid": "Urban\nPostpaid", "urban_prepaid": "Urban\nPrepaid",
                   "rural_prepaid": "Rural\nPrepaid"}
    segs = list(decomp["seg_pv_loss"].keys())
    values = [decomp["seg_pv_loss"][s] for s in segs]
    pcts = [decomp["pct_of_total"][s] for s in segs]
    labels = [seg_labels[s] for s in segs]
    colors = ['#2E75B6', '#C00000', '#70AD47']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.2, width=0.55)

    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.02,
                f'₹{val:,.0f} Cr\n({pct:.0f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Present Value of Cannibalization Loss (₹ Crore)', fontsize=11)
    ax.set_title('Where Does Cannibalization Cost Come From?\n(Base Case, 8-Year PV)',
                 fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── 3. WATERFALL CHART: GROSS REVENUE → NET NPV ───────────────────────────────
def plot_waterfall(
    save_path: str = "outputs/npv_waterfall.png",
) -> dict:
    """
    Waterfall showing how Jio's gross revenue is eroded by cannibalization
    and competitive response down to net contribution, then to NPV.
    Uses 8-year cumulative PV figures, base case assumptions.
    """
    result = compute_npv(include_terminal_value=True, verbose=False)

    pv_jio_revenue = sum(
        rev / (1.11 ** (i+1)) for i, rev in enumerate(result["jio_revenues"])
    )
    pv_cannibal    = result["pv_cannibalization"]
    pv_comp        = result["pv_comp_response"]
    pv_opex        = pv_jio_revenue - pv_cannibal - pv_comp - sum(result["pv_cash_flows"])
    pv_ebitda      = sum(result["pv_cash_flows"])
    tv             = result["terminal_value_pv"]
    capex          = result["capex_pv"]
    npv_final      = result["npv_with_tv"]

    # Waterfall steps
    labels = ['Gross Jio\nRevenue\n(PV)', 'Less:\nCannibal-\nization',
              'Less:\nComp.\nResponse', 'Less:\nOperating\nCosts',
              '= EBITDA\n(PV)', 'Plus:\nTerminal\nValue', 'Less:\nCapex\n(PV)',
              '= NPV']
    values = [pv_jio_revenue, -pv_cannibal, -pv_comp, -pv_opex,
              None, tv, -capex, None]  # None = computed subtotal

    # Compute running positions for waterfall
    running = 0
    bottoms, heights, colors_list = [], [], []
    pos_color, neg_color, total_color = '#2E75B6', '#C00000', '#1F4E79'

    cum = 0
    for i, (lab, val) in enumerate(zip(labels, values)):
        if val is None:  # subtotal bar
            bottoms.append(0)
            heights.append(cum)
            colors_list.append(total_color)
        else:
            if val >= 0:
                bottoms.append(cum)
                heights.append(val)
                colors_list.append(pos_color)
            else:
                bottoms.append(cum + val)
                heights.append(-val)
                colors_list.append(neg_color)
            cum += val

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, heights, bottom=bottoms, color=colors_list,
                  edgecolor='white', linewidth=1, width=0.65)

    # Connecting lines
    cum_track = 0
    cum_list = []
    for val in values:
        if val is not None:
            cum_track += val
        cum_list.append(cum_track)

    for i in range(len(labels) - 1):
        ax.plot([i + 0.32, i + 1 - 0.32], [cum_list[i], cum_list[i]],
               color='gray', linewidth=0.8, linestyle='--')

    # Value labels
    for i, (bot, hei, val) in enumerate(zip(bottoms, heights, values)):
        y_pos = bot + hei + max(heights) * 0.02
        display_val = val if val is not None else cum_list[i]
        ax.text(i, y_pos, f'₹{display_val:,.0f}Cr', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('₹ Crore (Present Value)', fontsize=11)
    ax.set_title('NPV Waterfall — Base Case (8-Year Horizon + Terminal Value)',
                fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        "pv_jio_revenue": pv_jio_revenue, "pv_cannibal": pv_cannibal,
        "pv_comp": pv_comp, "pv_opex": pv_opex, "pv_ebitda": pv_ebitda,
        "terminal_value": tv, "capex_pv": capex, "npv_final": npv_final,
    }


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("HOUR 5 — CANNIBALIZATION DECOMPOSITION")
    print("=" * 60)

    results = run_simulation(N=10_000, seed=42, strategy="Aggressive", verbose=False)

    # 1. Swing scenario analysis
    swing = analyze_swing_scenarios(results)

    # 2. Segment decomposition (base case)
    decomp = decompose_by_segment()
    plot_segment_decomposition(decomp)

    # 3. Waterfall chart
    print("\nGenerating NPV waterfall chart...")
    waterfall = plot_waterfall()

    # Final interview-ready summary
    print("\n" + "=" * 60)
    print("HOUR 5 — KEY TALKING POINTS")
    print("=" * 60)
    print(f"""
  1. In {swing['pct_swing']:.1%} of simulated scenarios, cannibalization
     ALONE is the deciding factor between value creation and destruction.

  2. Of total cannibalization cost (₹{decomp['total']:,.0f} Cr PV),
     urban prepaid contributes {decomp['pct_of_total']['urban_prepaid']:.0f}%
     — by far the largest share. This is where due diligence should focus.

  3. The waterfall shows gross Jio revenue of ₹{waterfall['pv_jio_revenue']:,.0f} Cr
     is eroded by ₹{waterfall['pv_cannibal']:,.0f} Cr of cannibalization before
     even reaching operating costs — cannibalization is a bigger drag than
     competitive response in the base case.
""")
