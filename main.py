"""
main.py
=======
Single entrypoint — runs the full analysis pipeline end to end.

Usage:
    python main.py                  # full run, N=10,000, generates all charts
    python main.py --quick          # N=2,000 smoke test, ~5 seconds
    python main.py --skip-plots     # skip chart generation, numbers only

All outputs (charts, validation logs) are written to an outputs/ subfolder.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Jio Market Entry Decision Model"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use N=2,000 for a fast smoke test (vs production N=10,000)"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip chart generation"
    )
    args = parser.parse_args()

    N = 2_000 if args.quick else 10_000
    t_start = time.time()
    os.makedirs("outputs", exist_ok=True)

    # ── STEP 1: BASE CASE & VALIDATION ───────────────────────────────────────
    section("STEP 1 / 7 — DETERMINISTIC BASE CASE & VALIDATION")
    from npv_engine import compute_npv, validate_base_case, break_even_analysis

    base_result = compute_npv(include_terminal_value=True, verbose=True)
    validate_base_case(verbose=True)
    break_even_analysis(verbose=True)

    # ── STEP 2: MONTE CARLO ───────────────────────────────────────────────────
    section(f"STEP 2 / 7 — MONTE CARLO SIMULATION  (N={N:,}, Cholesky-correlated)")
    from monte_carlo import run_simulation

    mc_results = run_simulation(N=N, seed=42, strategy="Aggressive", verbose=True)

    # ── STEP 3: STRATEGY COMPARISON ──────────────────────────────────────────
    section("STEP 3 / 7 — STRATEGY COMPARISON  (Aggressive / Moderate / Delay)")
    from strategy_comparison import run_strategy_comparison

    strategy_results = run_strategy_comparison(N=N, seed=42)

    # ── STEP 4: SENSITIVITY ANALYSIS (optional — requires SALib) ─────────────
    section("STEP 4 / 7 — SOBOL SENSITIVITY ANALYSIS")
    sobol_results = None
    try:
        from sensitivity import run_sobol_analysis
        sobol_n_base = 256 if args.quick else 1024
        sobol_results = run_sobol_analysis(n_base=sobol_n_base, seed=42)
    except ImportError:
        print("""
  SALib not installed — skipping Sobol analysis.
  To enable:  pip install SALib
  Then re-run: python main.py
  
  All other steps will still run normally.
""")

    # ── STEP 5: CANNIBALIZATION DECOMPOSITION ─────────────────────────────────
    section("STEP 5 / 7 — CANNIBALIZATION DECOMPOSITION")
    from cannibalization_decomposition import (
        analyze_swing_scenarios, decompose_by_segment
    )

    swing = analyze_swing_scenarios(mc_results)
    decomp = decompose_by_segment()

    # ── STEP 6: HISTORICAL BACKTEST ───────────────────────────────────────────
    section("STEP 6 / 7 — HISTORICAL BACKTEST  (2016–2024)")
    from historical_backtest import run_backtest, analyze_backtest_pattern

    backtest = run_backtest()
    analyze_backtest_pattern(backtest)

    # ── STEP 7: CHARTS ────────────────────────────────────────────────────────
    if not args.skip_plots:
        section("STEP 7 / 7 — GENERATING CHARTS")
        try:
            from visualize import plot_npv_distribution, plot_cumulative_probability
            print("Generating NPV distribution plots...")
            plot_npv_distribution(mc_results)
            plot_cumulative_probability(mc_results)
        except Exception as e:
            print(f"  Warning: NPV distribution charts failed: {e}")

        try:
            from cannibalization_decomposition import (
                plot_segment_decomposition, plot_waterfall
            )
            print("Generating cannibalization charts...")
            plot_segment_decomposition(decomp)
            plot_waterfall()
        except Exception as e:
            print(f"  Warning: cannibalization charts failed: {e}")

        if sobol_results is not None:
            try:
                from sensitivity import plot_tornado_sobol
                print("Generating Sobol tornado chart...")
                plot_tornado_sobol(sobol_results["results_table"])
            except Exception as e:
                print(f"  Warning: tornado chart failed: {e}")

        try:
            from strategy_comparison import (
                plot_strategy_distributions, plot_risk_return
            )
            print("Generating strategy comparison charts...")
            plot_strategy_distributions(strategy_results["all_results"])
            plot_risk_return(strategy_results["all_results"])
        except Exception as e:
            print(f"  Warning: strategy comparison charts failed: {e}")

        try:
            from break_even_frontier import build_breakeven_grid, plot_breakeven_frontier
            n_pts = 40 if args.quick else 60
            print(f"Generating break-even frontier ({n_pts}×{n_pts} grid)...")
            grid = build_breakeven_grid(n_points=n_pts)
            plot_breakeven_frontier(grid, mc_results=mc_results)
        except Exception as e:
            print(f"  Warning: break-even frontier failed: {e}")

        try:
            from historical_backtest import plot_backtest
            print("Generating historical backtest chart...")
            plot_backtest(backtest)
        except Exception as e:
            print(f"  Warning: backtest chart failed: {e}")

    else:
        section("STEP 7 / 7 — SKIPPED  (--skip-plots)")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    s = mc_results["summary"]

    section("PIPELINE COMPLETE")
    print(f"""
  Runtime: {elapsed:.1f}s

  ┌─────────────────────────────────────────────────────────┐
  │  BASE CASE NPV  (8yr + terminal value):  ₹{base_result['npv_with_tv']:>10,.0f} Cr  │
  │  Monte Carlo mean NPV  (N={N:,}):        ₹{s['mean_npv']:>10,.0f} Cr  │
  │  Monte Carlo median NPV:                 ₹{s['median_npv']:>10,.0f} Cr  │
  │  P(value destruction):                     {s['p_value_destruction']:>10.1%}     │
  │  P(cannibalization alone flips sign):      {s['p_cannibal_swing']:>10.1%}     │
  └─────────────────────────────────────────────────────────┘

  RECOMMENDATION: Launch aggressively — conditional on risk capacity.
  Mean NPV is positive; median is mildly negative. This is a close call,
  not an obvious one. See Jio_Decision_Memo.docx for the full analysis.

  Charts → outputs/   |   Run tests → python test_npv_engine.py
""")


if __name__ == "__main__":
    main()
