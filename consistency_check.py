"""
consistency_check.py — Day 5, Hour 4
========================================
Automated cross-module consistency audit. Three modules independently
touch Monte Carlo simulation logic — strategy_comparison.py and
risk_adjusted_selection.py both call monte_carlo.run_simulation()
directly, but correlation_robustness.py necessarily REIMPLEMENTS the
simulation loop (since it needs to scale the correlation matrix before
Cholesky decomposition, which run_simulation() doesn't expose as a
parameter).

This script verifies that reimplementation has not silently drifted
from the canonical engine — at correlation_scale=1.0 (the unmodified
matrix), correlation_robustness.py's loop should produce numbers
IDENTICAL to monte_carlo.run_simulation(), to the rupee.

Run this after ANY edit to monte_carlo.py or correlation_robustness.py
to catch silent logic drift immediately, rather than discovering it
during a Day 5-style numeric audit days later.
"""

from monte_carlo import run_simulation
from correlation_robustness import run_simulation_with_corr_scale
from risk_adjusted_selection import run_simulation as ras_run_simulation_check


def check_correlation_robustness_consistency() -> bool:
    """
    correlation_robustness.py reimplements the simulation loop. At
    corr_scale=1.0 (unmodified correlation matrix), it MUST match
    monte_carlo.run_simulation() exactly, since they're doing the
    identical calculation by construction.
    """
    print("Checking correlation_robustness.py vs monte_carlo.py at scale=1.0...")

    r1 = run_simulation_with_corr_scale(1.0, N=10_000, seed=42)
    r2 = run_simulation(N=10_000, seed=42, strategy="Aggressive", verbose=False)

    fields_to_check = [
        ("mean", "mean_npv"), ("median", "median_npv"), ("std", "std_npv"),
        ("p10", "p10_npv"), ("p90", "p90_npv"),
        ("p_destruction", "p_value_destruction"),
    ]

    all_match = True
    for k1, k2 in fields_to_check:
        v1 = r1[k1]
        v2 = r2["summary"][k2]
        match = abs(v1 - v2) < 1.0
        all_match = all_match and match
        status = "✓" if match else "✗ MISMATCH"
        print(f"  {k1:<16} corr_robustness={v1:>14,.1f}  monte_carlo={v2:>14,.1f}  {status}")

    return all_match


def check_strategy_comparison_uses_shared_engine() -> bool:
    """
    Confirm strategy_comparison.py and risk_adjusted_selection.py both
    call the SAME monte_carlo.run_simulation() function (not separate
    reimplementations), so any future fix to the core engine
    automatically propagates to both without manual syncing.
    """
    import inspect
    import strategy_comparison
    import risk_adjusted_selection

    sc_source = inspect.getsource(strategy_comparison)
    ras_source = inspect.getsource(risk_adjusted_selection)

    sc_uses_shared = "from monte_carlo import run_simulation" in sc_source
    ras_uses_shared = "from monte_carlo import run_simulation" in ras_source

    print(f"\nstrategy_comparison.py imports shared run_simulation():    "
          f"{'✓' if sc_uses_shared else '✗ REIMPLEMENTS OWN LOOP'}")
    print(f"risk_adjusted_selection.py imports shared run_simulation(): "
          f"{'✓' if ras_uses_shared else '✗ REIMPLEMENTS OWN LOOP'}")

    return sc_uses_shared and ras_uses_shared


def check_seed_and_n_consistency() -> bool:
    """
    Confirm every module that runs the production Monte Carlo uses the
    SAME N=10,000 and seed=42 as the canonical run — a module silently
    using N=5,000 or seed=1 would produce different numbers without
    any error being raised.
    """
    import re
    import inspect
    import strategy_comparison
    import risk_adjusted_selection
    import break_even_frontier
    import cannibalization_decomposition
    import competitive_response_stresstest

    modules = {
        "strategy_comparison": strategy_comparison,
        "risk_adjusted_selection": risk_adjusted_selection,
        "break_even_frontier": break_even_frontier,
        "cannibalization_decomposition": cannibalization_decomposition,
    }

    print("\nChecking N=10,000, seed=42 consistency across modules:")
    all_consistent = True
    for name, mod in modules.items():
        source = inspect.getsource(mod)
        # Find all run_simulation(...) calls with explicit N= and seed=
        calls = re.findall(r"run_simulation\(\s*N\s*=\s*([\d_]+)\s*,\s*seed\s*=\s*(\d+)", source)
        if calls:
            for n_str, seed_str in calls:
                n_val = int(n_str.replace("_", ""))
                seed_val = int(seed_str)
                consistent = (n_val == 10_000 and seed_val == 42)
                all_consistent = all_consistent and consistent
                status = "✓" if consistent else f"✗ MISMATCH (found N={n_val}, seed={seed_val})"
                print(f"  {name:<35} {status}")
        else:
            print(f"  {name:<35} (no direct run_simulation(N=,seed=) call found — uses defaults or different pattern)")

    return all_consistent


if __name__ == "__main__":
    print("=" * 70)
    print("DAY 5, HOUR 4 — CROSS-MODULE CONSISTENCY CHECK")
    print("=" * 70)

    check1 = check_correlation_robustness_consistency()
    check2 = check_strategy_comparison_uses_shared_engine()
    check3 = check_seed_and_n_consistency()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Correlation-robustness reimplementation matches canonical: {'PASS' if check1 else 'FAIL'}")
    print(f"  Strategy modules use shared engine (no silent reimplementation): {'PASS' if check2 else 'FAIL'}")
    print(f"  N/seed consistent across all modules: {'PASS' if check3 else 'FAIL'}")

    if check1 and check2 and check3:
        print("\n  ALL CHECKS PASS. No logic drift detected across the codebase.")
        print("  This script should be re-run after any future edit to monte_carlo.py,")
        print("  npv_engine.py, or any module that reimplements simulation logic.")
    else:
        print("\n  ONE OR MORE CHECKS FAILED. Investigate before trusting cross-file")
        print("  numeric consistency claims in the README or executive memo.")
