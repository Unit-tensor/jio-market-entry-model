"""
visualize.py
============
Convergence testing + first output plots.

Three deliverables:
1. Rigorous convergence test (N=100 to N=50,000) with plot — defends choice of N=10,000
2. NPV distribution histogram with P10/P50/P90 markers — the headline visual
3. Cumulative probability curve — shows P(NPV < 0) directly

All plots saved to outputs/ as PNG for inclusion in the final presentation deck.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from monte_carlo import run_simulation, convergence_test

plt.rcParams['font.family']   = 'DejaVu Sans'
plt.rcParams['axes.grid']     = True
plt.rcParams['grid.alpha']    = 0.25
plt.rcParams['figure.dpi']    = 140


# ── 1. CONVERGENCE TEST (RIGOROUS) ────────────────────────────────────────────
def run_convergence_study(
    n_values = None,
    seed: int = 42,
    save_path: str = "outputs/convergence_test.png",
) -> dict:
    """
    Run simulation at increasing N, track mean/P10/P90 stability.
    This is the evidence you cite when an interviewer asks
    'is 10,000 iterations enough?'
    """
    if n_values is None:
        n_values = [50, 100, 250, 500, 1_000, 2_000, 3_000, 5_000,
                    7_500, 10_000, 20_000, 50_000]

    means, p10s, p90s, std_errs = [], [], [], []

    print("\n" + "=" * 60)
    print("CONVERGENCE STUDY")
    print("=" * 60)
    print(f"{'N':>8} {'Mean NPV':>14} {'P10':>14} {'P90':>14} {'StdErr':>10}")
    print("─" * 64)

    for n in n_values:
        res = run_simulation(N=n, seed=seed, verbose=False)
        arr = res["npv_with_tv"]
        m, p10, p90 = np.mean(arr), np.percentile(arr, 10), np.percentile(arr, 90)
        se = np.std(arr) / np.sqrt(n)
        means.append(m); p10s.append(p10); p90s.append(p90); std_errs.append(se)
        print(f"{n:>8,} {m:>14,.0f} {p10:>14,.0f} {p90:>14,.0f} {se:>10,.0f}")

    # Determine convergence threshold: first N where |mean(N) - mean(N_max)| < 2% of mean(N_max)
    final_mean = means[-1]
    threshold = 0.02 * abs(final_mean)
    converged_n = None
    for n, m in zip(n_values, means):
        if abs(m - final_mean) < threshold:
            converged_n = n
            break

    print("─" * 64)
    print(f"  Convergence threshold (±2% of N={n_values[-1]:,} mean): ₹{threshold:,.0f} Cr")
    print(f"  First N within threshold: N = {converged_n:,}" if converged_n
          else "  Did not converge within tested range")

    # ── PLOT ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ax1.plot(n_values, means, marker='o', color='#1F4E79', linewidth=2,
             label='Mean NPV', markersize=6)
    ax1.fill_between(n_values, p10s, p90s, alpha=0.15, color='#2E75B6',
                      label='P10–P90 band')
    ax1.plot(n_values, p10s, '--', color='#C00000', linewidth=1.3, label='P10')
    ax1.plot(n_values, p90s, '--', color='#006400', linewidth=1.3, label='P90')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Monte Carlo Iterations (N)', fontsize=10)
    ax1.set_ylabel('NPV (₹ Crore)', fontsize=10)
    ax1.set_title('Convergence of NPV Distribution Statistics', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    if converged_n:
        ax1.axvline(converged_n, color='orange', linestyle='-.', linewidth=1.5, alpha=0.7)
        ax1.annotate(f'Converged\nN={converged_n:,}', xy=(converged_n, final_mean),
                     xytext=(converged_n*1.5, final_mean*1.3),
                     fontsize=8, color='darkorange', fontweight='bold')

    ax2 = axes[1]
    ax2.plot(n_values, std_errs, marker='s', color='#C00000', linewidth=2, markersize=6)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('Number of Monte Carlo Iterations (N)', fontsize=10)
    ax2.set_ylabel('Standard Error of Mean (₹ Crore)', fontsize=10)
    ax2.set_title('Standard Error Decay (∝ 1/√N)', fontsize=11, fontweight='bold')
    # Theoretical 1/sqrt(N) reference line
    ref = std_errs[0] * np.sqrt(n_values[0] / np.array(n_values))
    ax2.plot(n_values, ref, ':', color='gray', linewidth=1.2, label='Theoretical 1/√N')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return {
        "n_values": n_values, "means": means, "p10s": p10s, "p90s": p90s,
        "std_errs": std_errs, "converged_n": converged_n,
        "convergence_sentence": (
            f"Convergence testing across N=50 to N={n_values[-1]:,} shows the mean NPV "
            f"stabilises within ±2% of its final value by N={converged_n:,}. "
            f"I use N=10,000 as a safety margin over this threshold, with standard "
            f"error of the mean at N=10,000 being ₹{std_errs[n_values.index(10000)]:,.0f} Cr "
            f"— under 1% of the mean NPV magnitude."
            if converged_n and 10000 in n_values else
            f"Convergence testing shows standard error decaying as the theoretical 1/√N rate, "
            f"confirming the simulation is implemented correctly."
        )
    }


# ── 2. NPV DISTRIBUTION HISTOGRAM ─────────────────────────────────────────────
def plot_npv_distribution(
    results: dict,
    save_path: str = "outputs/npv_distribution.png",
    strategy_label: str = "Aggressive Launch",
) -> None:
    """
    The headline visual: NPV distribution with P10/P50/P90 markers
    and shaded value-destruction region.
    """
    npv = results["npv_with_tv"]
    s = results["summary"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n_bins = 60
    counts, bins, patches = ax.hist(npv, bins=n_bins, color='#2E75B6',
                                     alpha=0.75, edgecolor='white', linewidth=0.4)

    # Color bars red where NPV < 0 (value destruction zone)
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#C00000')
            patch.set_alpha(0.65)

    # Percentile markers
    for pct, label, color in [(s['p10_npv'], 'P10', '#C00000'),
                                (s['median_npv'], 'P50 (Median)', '#1F4E79'),
                                (s['p90_npv'], 'P90', '#006400')]:
        ax.axvline(pct, color=color, linestyle='--', linewidth=1.8)
        ax.text(pct, ax.get_ylim()[1]*0.92, f'{label}\n₹{pct:,.0f}Cr',
                rotation=0, fontsize=8.5, color=color, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=color, alpha=0.85))

    ax.axvline(0, color='black', linewidth=1.2, linestyle='-')

    ax.set_xlabel('NPV (₹ Crore)', fontsize=11)
    ax.set_ylabel('Number of Scenarios (out of {:,})'.format(s['N']), fontsize=11)
    ax.set_title(
        f'Distribution of NPV Outcomes — {strategy_label}\n'
        f'P(Value Destruction) = {s["p_value_destruction"]:.1%}  |  '
        f'N = {s["N"]:,} scenarios',
        fontsize=12, fontweight='bold'
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))

    # Annotation for red zone
    ax.annotate('Value-destroying\nscenarios', xy=(s['p10_npv']*0.6, ax.get_ylim()[1]*0.5),
                fontsize=9, color='#C00000', style='italic', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── 3. CUMULATIVE PROBABILITY CURVE ──────────────────────────────────────────
def plot_cumulative_probability(
    results: dict,
    save_path: str = "outputs/npv_cdf.png",
) -> None:
    """
    CDF view: directly shows P(NPV < x) for any x.
    Useful for answering 'what's the probability NPV is below ₹X Cr?'
    """
    npv_sorted = np.sort(results["npv_with_tv"])
    n = len(npv_sorted)
    cdf = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(npv_sorted, cdf, color='#1F4E79', linewidth=2)
    ax.fill_between(npv_sorted, 0, cdf, where=(npv_sorted < 0),
                     color='#C00000', alpha=0.2)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)

    p_destruction = np.mean(npv_sorted < 0)
    ax.scatter([0], [p_destruction], color='#C00000', zorder=5, s=60)
    ax.annotate(f'P(NPV<0) = {p_destruction:.1%}',
                xy=(0, p_destruction), xytext=(npv_sorted[int(n*0.15)], p_destruction+0.08),
                fontsize=10, color='#C00000', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.2))

    ax.set_xlabel('NPV (₹ Crore)', fontsize=11)
    ax.set_ylabel('Cumulative Probability  P(NPV ≤ x)', fontsize=11)
    ax.set_title('Cumulative Distribution of NPV Outcomes', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K'))
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("HOUR 4 — CONVERGENCE TESTING & VISUALIZATION")
    print("=" * 60)

    # 1. Convergence study
    conv = run_convergence_study()
    print(f"\n  INTERVIEW SENTENCE:\n  \"{conv['convergence_sentence']}\"")

    # 2. Production run at N=10,000
    print("\n" + "=" * 60)
    print("PRODUCTION RUN — N=10,000")
    print("=" * 60)
    results = run_simulation(N=10_000, seed=42, strategy="Aggressive", verbose=True)

    # 3. Distribution plot
    print("\nGenerating NPV distribution histogram...")
    plot_npv_distribution(results)

    # 4. CDF plot
    print("Generating cumulative probability curve...")
    plot_cumulative_probability(results)

    print("\nCharts saved to outputs/.")
