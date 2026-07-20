"""
break_even_frontier.py
=========================================
2D break-even frontier: maps the combinations of (Jio share capture,
urban prepaid cannibalization rate) at which NPV = 0.

This is the single most useful visual for an investment committee.
Instead of one break-even number per variable (as in a 1D analysis),
this shows the FULL boundary in two-dimensional assumption space —
answering "for any combination of these two key uncertain variables,
does this deal create or destroy value?"

Method
------
1. Build a grid over (share_scale, cannibal_urban_prepaid).
2. For each grid point, compute NPV (all other variables at base case).
3. Plot as a filled contour: green = value-creating, red = value-destroying.
4. Overlay the actual outcome point (Jio's realized share & inferred
   cannibalization) to show where reality landed relative to the frontier.
5. Overlay the Monte Carlo scenario cloud to show how much of the
   probability mass sits on each side of the line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from npv_engine import compute_npv
from monte_carlo import run_simulation

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid']   = True
plt.rcParams['grid.alpha']  = 0.2
plt.rcParams['figure.dpi']  = 140


def build_breakeven_grid(
    share_range:    tuple = (0.3, 2.2),    # share_scale multiplier
    cannibal_range: tuple = (0.05, 0.85),  # urban prepaid cannibalization rate
    n_points:       int   = 60,
) -> dict:
    """
    Build the NPV grid over share_scale × cannibalization_rate.
    All other variables held at base case (deterministic).
    """
    share_vals    = np.linspace(*share_range, n_points)
    cannibal_vals = np.linspace(*cannibal_range, n_points)

    npv_grid = np.zeros((n_points, n_points))

    print("Building break-even frontier grid "
          f"({n_points}x{n_points} = {n_points**2:,} model evaluations)...")

    for i, share in enumerate(share_vals):
        for j, can in enumerate(cannibal_vals):
            r = compute_npv(
                share_scale             = share,
                cannibal_urban_prepaid  = can,
                include_terminal_value  = True,
            )
            npv_grid[j, i] = r["npv_with_tv"]   # [row=cannibal, col=share]

    return {
        "share_vals": share_vals,
        "cannibal_vals": cannibal_vals,
        "npv_grid": npv_grid,
    }


def find_frontier_points(grid: dict) -> np.ndarray:
    """
    For each share_scale value, find the cannibalization rate at which
    NPV crosses zero (the break-even boundary), via linear interpolation
    between grid points that straddle zero.
    """
    share_vals    = grid["share_vals"]
    cannibal_vals = grid["cannibal_vals"]
    npv_grid      = grid["npv_grid"]

    frontier = []
    for i, share in enumerate(share_vals):
        col = npv_grid[:, i]
        # Find sign changes along the cannibalization axis
        sign_changes = np.where(np.diff(np.sign(col)) != 0)[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation between cannibal_vals[idx] and [idx+1]
            y0, y1 = col[idx], col[idx + 1]
            x0, x1 = cannibal_vals[idx], cannibal_vals[idx + 1]
            if y1 != y0:
                breakeven_cannibal = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                frontier.append((share, breakeven_cannibal))

    return np.array(frontier)


def plot_breakeven_frontier(
    grid: dict,
    mc_results: dict = None,
    save_path: str = "outputs/breakeven_frontier.png",
) -> dict:
    """
    Main deliverable: filled contour plot with break-even line,
    actual outcome marker, and optional MC scenario cloud overlay.
    """
    share_vals    = grid["share_vals"]
    cannibal_vals = grid["cannibal_vals"]
    npv_grid      = grid["npv_grid"]

    fig, ax = plt.subplots(figsize=(11, 8))

    # Custom diverging colormap: red (destroy) -> white (zero) -> green (create)
    cmap = LinearSegmentedColormap.from_list(
        "value_rg", ["#C00000", "#FFFFFF", "#1A7A1A"], N=256
    )
    vmax = np.percentile(np.abs(npv_grid), 95)

    cf = ax.contourf(share_vals, cannibal_vals, npv_grid, levels=40,
                     cmap=cmap, vmin=-vmax, vmax=vmax, extend='both')
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('NPV (₹ Crore)', fontsize=10)
    cbar.formatter = mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}K')
    cbar.update_ticks()

    # Break-even line (NPV = 0 contour)
    cs = ax.contour(share_vals, cannibal_vals, npv_grid, levels=[0],
                    colors='black', linewidths=2.5, linestyles='-')
    # Manually position labels to avoid the two branches' labels colliding
    # (the frontier has two branches near share_scale=1.0 — label only once,
    # at a point on the right/upper branch, away from the crowded left branch)
    try:
        label_pos = [(share_vals[int(len(share_vals)*0.55)], cannibal_vals[int(len(cannibal_vals)*0.75)])]
        ax.clabel(cs, fmt='NPV = 0', fontsize=9, manual=label_pos)
    except Exception:
        ax.clabel(cs, fmt='NPV = 0', fontsize=9)

    # Overlay Monte Carlo scenario cloud (subsample for visibility)
    if mc_results is not None:
        n_sample = min(2000, len(mc_results["input_share_scale_applied"]))
        idx = np.random.choice(len(mc_results["input_share_scale_applied"]), n_sample, replace=False)
        x_mc = mc_results["input_share_scale_applied"][idx]
        y_mc = mc_results["input_cannibal_up"][idx]
        npv_mc = mc_results["npv_with_tv"][idx]
        colors_mc = ['#00FF00' if v > 0 else '#FF00FF' for v in npv_mc]
        ax.scatter(x_mc, y_mc, c=colors_mc, s=4, alpha=0.35, zorder=3,
                  label='MC scenarios (sampled)')

    # Actual outcome marker: Jio's realized share_scale ≈ 1.0 (calibrated to
    # actuals), and inferred cannibalization ≈ 0.30-0.40 (Airtel lost only
    # 0.5pp despite Jio +6.4%, implying low TRUE cannibalization in Y1, but
    # rising over the full 8-year horizon as the price war deepened)
    ax.scatter([1.0], [0.38], marker='*', s=600, color='gold',
              edgecolor='black', linewidth=1.5, zorder=6,
              label='Approx. realized outcome')
    ax.annotate('Realized\noutcome\n(approx.)', xy=(1.0, 0.38), xytext=(1.25, 0.50),
               fontsize=9.5, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='black'))

    # Base case marker
    ax.scatter([1.0], [0.45], marker='D', s=140, color='#1F4E79',
              edgecolor='white', linewidth=1.5, zorder=6, label='Base case assumption')

    ax.set_xlabel('Jio Share Capture (scale relative to base-case table)', fontsize=11)
    ax.set_ylabel('Urban Prepaid Cannibalization Rate', fontsize=11)
    ax.set_title('Break-Even Frontier: Where Does the Investment Destroy Value?\n'
                'Green = value-creating  |  Red = value-destroying  |  Black line = NPV=0',
                fontsize=12.5, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    frontier_points = find_frontier_points(grid)
    return {"frontier_points": frontier_points}


def summarize_frontier(frontier_points: np.ndarray) -> None:
    """Print key frontier statistics for interview talking points."""
    print("\n" + "=" * 60)
    print("BREAK-EVEN FRONTIER — KEY POINTS")
    print("=" * 60)

    # At share_scale = 1.0 (base case strategy), find break-even cannibalization
    idx_base = np.argmin(np.abs(frontier_points[:, 0] - 1.0))
    share_at_base, cannibal_at_base = frontier_points[idx_base]
    print(f"  At base-case share capture (scale=1.0):")
    print(f"    Break-even cannibalization rate ≈ {cannibal_at_base:.1%}")

    # At cannibalization = 0.45 (base case), find break-even share
    cannibal_target = 0.45
    diffs = np.abs(frontier_points[:, 1] - cannibal_target)
    idx_can = np.argmin(diffs)
    share_at_can, _ = frontier_points[idx_can]
    print(f"  At base-case cannibalization (45%):")
    print(f"    Break-even share capture scale ≈ {share_at_can:.2f}x "
          f"(i.e. {share_at_can*0.12:.1%} Year-1 urban prepaid share)")

    print(f"\n  INTERVIEW SENTENCE:")
    print(f'  "The break-even frontier shows that at our base-case share')
    print(f'  capture, cannibalization would need to exceed {cannibal_at_base:.0%} for')
    print(f'  the deal to destroy value — well above our calibrated estimate')
    print(f'  of 38-45%. The investment has genuine margin of safety, but it')
    print(f'  is not a one-dimensional bet: lower share capture pulls the')
    print(f'  break-even cannibalization rate down sharply, which is exactly')
    print(f'  the interaction my correlation structure (ρ=-0.40 between TAM')
    print(f'  growth and cannibalization) is designed to capture."')


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(42)

    print("=" * 70)
    print("DAY 3, HOUR 2 — BREAK-EVEN FRONTIER")
    print("=" * 70)

    grid = build_breakeven_grid(n_points=60)

    print("\nGenerating Monte Carlo scenario cloud for overlay...")
    mc_results = run_simulation(N=10_000, seed=42, strategy="Aggressive", verbose=False)

    print("\nGenerating break-even frontier plot...")
    frontier_out = plot_breakeven_frontier(grid, mc_results=mc_results)

    summarize_frontier(frontier_out["frontier_points"])
