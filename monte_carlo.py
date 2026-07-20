"""
monte_carlo.py
========================
Correlated Monte Carlo simulation engine.

Architecture
------------
Step 1  Generate N×9 matrix of independent Uniform(0,1) draws.
Step 2  Convert to N×9 standard normals via inverse CDF (probit transform).
Step 3  Apply Cholesky decomposition of the correlation matrix to induce
        correlations between variables. This is the critical step that
        separates this simulation from naive independent draws.
Step 4  Convert correlated normals back to target distributions via the
        inverse CDF of each marginal distribution (probability integral
        transform). This is mathematically known as the Gaussian copula.
Step 5  Pass each row of draws to compute_npv() and store results.
Step 6  Aggregate: NPV distribution, P10/P50/P90, P(destruction),
        cannibalization swing count, strategy comparison.

Why Gaussian Copula (not direct multivariate normal)?
Because our marginals are Beta and Lognormal — not Normal.
The copula approach lets us specify arbitrary marginals with any correlation
structure. This is how quantitative finance practitioners actually do it.

How to defend this in interviews
---------------------------------
"I model correlation using the Gaussian copula: I generate correlated
standard normals using Cholesky decomposition of the correlation matrix,
then transform each marginal to its target distribution via the inverse CDF.
This preserves the full marginal distributions while introducing the
economic correlations I specified. It's the same approach used in
credit portfolio models and multi-factor risk systems."
"""

import numpy as np
from scipy import stats
from scipy.linalg import cholesky
import time
from assumptions import (
    DISTS, CORR_MATRIX, VARIABLE_NAMES, STRATEGIES, CAPEX_BASE_CR
)
from npv_engine import compute_npv, break_even_analysis, validate_base_case


def run_simulation(
    N:                       int   = 10_000,
    seed:                    int   = 42,
    include_terminal_value:  bool  = True,
    strategy:                str   = "Aggressive",
    verbose:                 bool  = True,
) -> dict:
    """
    Run N-scenario Monte Carlo simulation with Cholesky-correlated draws.

    Parameters
    ----------
    N       : number of scenarios (10,000 for production)
    seed    : random seed for reproducibility
    include_terminal_value : pass to compute_npv
    strategy : "Aggressive", "Moderate", or "Delay"
    verbose  : print progress and summary

    Returns
    -------
    dict with all results arrays and summary statistics
    """

    np.random.seed(seed)
    t0 = time.time()

    # ── STEP 1: CHOLESKY DECOMPOSITION ───────────────────────────────────────
    # L is lower triangular: CORR = L @ L.T
    # Verified positive-definite in assumptions.py at import time.
    L = cholesky(CORR_MATRIX, lower=True)

    # ── STEP 2: GENERATE INDEPENDENT STANDARD NORMALS ─────────────────────────
    # Shape: (N_variables, N_scenarios)
    n_vars = len(VARIABLE_NAMES)
    Z_independent = np.random.standard_normal((n_vars, N))

    # ── STEP 3: INDUCE CORRELATIONS ───────────────────────────────────────────
    # Z_correlated = L @ Z_independent
    # Each row of Z_correlated is now a correlated standard normal vector.
    Z_correlated = L @ Z_independent   # shape: (n_vars, N)

    # ── STEP 4: TRANSFORM TO TARGET DISTRIBUTIONS ─────────────────────────────
    # Gaussian copula: U = Phi(Z_correlated), then X = F_inv(U)
    # Phi = standard normal CDF; F_inv = inverse CDF of target marginal.
    #
    # Variable ordering (must match CORR_MATRIX rows):
    # 0: tam_growth           Normal   → stats.norm CDF/PPF
    # 1: share_up_y1          Beta     → stats.beta CDF/PPF
    # 2: can_up               Beta
    # 3: can_pp               Beta
    # 4: can_rural            Beta
    # 5: jio_arpu             Normal
    # 6: capex_mult           Lognormal
    # 7: wacc                 Normal
    # 8: arpu_comp            Normal

    dist_list = [
        DISTS["tam_growth"],
        DISTS["share_urban_prepaid_y1"],
        DISTS["cannibal_urban_prepaid"],
        DISTS["cannibal_urban_postpaid"],
        DISTS["cannibal_rural"],
        DISTS["jio_arpu_y2plus"],
        DISTS["capex_overrun_mult"],
        DISTS["wacc"],
        DISTS["arpu_compression"],
    ]

    # Convert correlated normals → uniform via standard normal CDF
    U = stats.norm.cdf(Z_correlated)   # shape: (n_vars, N), values in (0,1)

    # Clip to avoid numerical issues at boundaries
    U = np.clip(U, 1e-8, 1 - 1e-8)

    # Transform each row to target marginal distribution
    draws = np.zeros_like(U)
    for i, dist in enumerate(dist_list):
        draws[i, :] = dist.ppf(U[i, :])

    # draws shape: (9, N) — each column is one scenario's input vector

    # ── STEP 5: APPLY STRATEGY PARAMETERS ────────────────────────────────────
    strat_params = STRATEGIES[strategy]
    capex_strat_mult = strat_params["capex_multiplier"]
    strategy_share_mult = strat_params["share_scale"]

    # NOTE: The stochastic share_urban_prepaid_y1 draw
    # (draws[1,:], Beta(2,11) distribution) was being generated but never
    # passed into compute_npv() — only the fixed strategy multiplier was
    # used. This meant Year-1 share was IDENTICAL across all 10,000
    # scenarios (always 7.89% blended), which is why competitive response
    # This meant share uncertainty was disabled and competitive response
    # Capture" showed up as the #2 Sobol driver despite supposedly being
    # stochastic — that ranking came entirely from the deterministic
    # NPV-engine sweep in sensitivity.py, not from this MC engine.
    #
    # Fix: convert the stochastic Beta draw (representing the urban-prepaid
    # Y1 share rate, calibrated mean 15.4%) into a SCALE FACTOR relative to
    # the base-case table value used in SHARE_BY_YEAR (12% for Y1 urban
    # prepaid, in assumptions.py JIO_SHARE_MIX). This scale factor is then
    # combined multiplicatively with the strategy's own scale (Aggressive=
    # 1.0, Moderate=0.65, Delay=0.40), so both sources of share variation
    # apply consistently across all three segments and all years.
    BASE_TABLE_SHARE_UP_Y1 = 0.12   # from assumptions.JIO_SHARE_MIX["urban_prepaid"]
    stochastic_share_scale = draws[1, :] / BASE_TABLE_SHARE_UP_Y1
    combined_share_scale   = stochastic_share_scale * strategy_share_mult

    # ── STEP 6: RUN NPV FOR EACH SCENARIO ─────────────────────────────────────
    results = {
        "npv":                    np.zeros(N),
        "npv_with_tv":            np.zeros(N),
        "npv_ex_all_losses":      np.zeros(N),
        "prob_value_destruction": np.zeros(N, dtype=int),
        "cannibalization_swing":  np.zeros(N, dtype=bool),
        "pv_cannibalization":     np.zeros(N),
        "pv_comp_response":       np.zeros(N),
        "capex_pv":               np.zeros(N),
        "competitive_response":   np.zeros(N, dtype=bool),
        # Store key inputs for post-hoc sensitivity analysis
        "input_tam_growth":       draws[0, :],
        "input_share_up":         draws[1, :],
        "input_share_scale_applied": combined_share_scale,
        "input_cannibal_up":      draws[2, :],
        "input_cannibal_pp":      draws[3, :],
        "input_cannibal_rural":   draws[4, :],
        "input_jio_arpu":         draws[5, :],
        "input_capex_mult":       draws[6, :],
        "input_wacc":             draws[7, :],
        "input_arpu_comp":        draws[8, :],
    }

    for i in range(N):
        r = compute_npv(
            tam_growth              = float(draws[0, i]),
            share_scale             = float(combined_share_scale[i]),
            cannibal_urban_prepaid  = float(draws[2, i]),
            cannibal_urban_postpaid = float(draws[3, i]),
            cannibal_rural          = float(draws[4, i]),
            jio_arpu_y2plus         = float(draws[5, i]),
            capex_overrun_mult      = float(draws[6, i]) * capex_strat_mult,
            wacc                    = float(draws[7, i]),
            arpu_compression        = float(draws[8, i]),
            include_terminal_value  = include_terminal_value,
            verbose                 = False,
        )
        results["npv"][i]                    = r["npv"]
        results["npv_with_tv"][i]            = r["npv_with_tv"]
        results["npv_ex_all_losses"][i]      = r["npv_ex_all_losses"]
        results["prob_value_destruction"][i] = r["prob_value_destruction"]
        results["cannibalization_swing"][i]  = r["cannibalization_swing"]
        results["pv_cannibalization"][i]     = r["pv_cannibalization"]
        results["pv_comp_response"][i]       = r["pv_comp_response"]
        results["capex_pv"][i]               = r["capex_pv"]
        results["competitive_response"][i]   = r["competitive_response"]

    elapsed = time.time() - t0

    # ── STEP 7: AGGREGATE STATISTICS ──────────────────────────────────────────
    npv_arr = results["npv_with_tv"]

    summary = {
        "strategy":               strategy,
        "N":                      N,
        "elapsed_sec":            round(elapsed, 2),
        "mean_npv":               float(np.mean(npv_arr)),
        "median_npv":             float(np.median(npv_arr)),
        "std_npv":                float(np.std(npv_arr)),
        "p10_npv":                float(np.percentile(npv_arr, 10)),
        "p25_npv":                float(np.percentile(npv_arr, 25)),
        "p75_npv":                float(np.percentile(npv_arr, 75)),
        "p90_npv":                float(np.percentile(npv_arr, 90)),
        "p_value_destruction":    float(np.mean(results["prob_value_destruction"])),
        "p_cannibal_swing":       float(np.mean(results["cannibalization_swing"])),
        "p_comp_response":        float(np.mean(results["competitive_response"])),
        "mean_pv_cannibal":       float(np.mean(results["pv_cannibalization"])),
        "mean_pv_comp":           float(np.mean(results["pv_comp_response"])),
    }

    results["summary"] = summary

    if verbose:
        _print_summary(summary)

    return results


def _print_summary(s: dict):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  STRATEGY:  {s['strategy']}   |   N = {s['N']:,}   |   {s['elapsed_sec']}s")
    print(sep)
    print(f"  Mean NPV:           ₹{s['mean_npv']:>12,.0f} Cr")
    print(f"  Median NPV (P50):   ₹{s['median_npv']:>12,.0f} Cr")
    print(f"  Std deviation:      ₹{s['std_npv']:>12,.0f} Cr")
    print(f"  P10 (downside):     ₹{s['p10_npv']:>12,.0f} Cr")
    print(f"  P25:                ₹{s['p25_npv']:>12,.0f} Cr")
    print(f"  P75:                ₹{s['p75_npv']:>12,.0f} Cr")
    print(f"  P90 (upside):       ₹{s['p90_npv']:>12,.0f} Cr")
    print(sep)
    print(f"  P(value destruction): {s['p_value_destruction']:.1%}")
    print(f"  P(cannibal swing):    {s['p_cannibal_swing']:.1%}  "
          f"← scenarios where cannibalization alone flips NPV")
    print(f"  P(comp response):     {s['p_comp_response']:.1%}  "
          f"← scenarios where Jio share > 8% triggers incumbent response")
    print(f"  Mean PV cannibal cost:  ₹{s['mean_pv_cannibal']:>10,.0f} Cr")
    print(f"  Mean PV comp cost:      ₹{s['mean_pv_comp']:>10,.0f} Cr")


def convergence_test(
    n_values: list = None,
    seed:     int  = 42,
) -> dict:
    """
    Test convergence of mean NPV and P10/P90 as N increases.
    Run this once to defend your choice of N=10,000 in interviews.
    """
    if n_values is None:
        n_values = [100, 500, 1_000, 3_000, 5_000, 10_000]

    print("\nCONVERGENCE TEST")
    print(f"{'N':>8} {'Mean NPV':>14} {'P10':>14} {'P90':>14} {'Std Err Mean':>14}")
    print("─" * 68)

    conv_results = {}
    for n in n_values:
        res = run_simulation(N=n, seed=seed, verbose=False)
        arr = res["npv_with_tv"]
        mean, p10, p90 = np.mean(arr), np.percentile(arr, 10), np.percentile(arr, 90)
        std_err = np.std(arr) / np.sqrt(n)
        conv_results[n] = {"mean": mean, "p10": p10, "p90": p90, "std_err": std_err}
        print(f"{n:>8,} {mean:>14,.0f} {p10:>14,.0f} {p90:>14,.0f} {std_err:>14,.0f}")

    return conv_results


def run_all_strategies(
    N:    int  = 10_000,
    seed: int  = 42,
) -> dict:
    """
    Run simulation for all three strategies and return comparative summary.
    """
    print("\n" + "=" * 55)
    print("STRATEGY COMPARISON")
    print("=" * 55)

    all_results = {}
    for strat in ["Aggressive", "Moderate", "Delay"]:
        print(f"\nRunning: {strat}...")
        all_results[strat] = run_simulation(N=N, seed=seed,
                                            strategy=strat, verbose=True)

    # Print comparison table
    print("\n" + "=" * 55)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 55)
    print(f"{'Metric':<28} {'Aggressive':>16} {'Moderate':>16} {'Delay':>16}")
    print("─" * 78)

    metrics = [
        ("Mean NPV (₹ Cr)",         "mean_npv",             "₹{:,.0f}"),
        ("P10 NPV (₹ Cr)",          "p10_npv",              "₹{:,.0f}"),
        ("P90 NPV (₹ Cr)",          "p90_npv",              "₹{:,.0f}"),
        ("P(value destruction)",    "p_value_destruction",  "{:.1%}"),
        ("P(cannibal swing)",       "p_cannibal_swing",     "{:.1%}"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:<26}"
        for strat in ["Aggressive", "Moderate", "Delay"]:
            v = all_results[strat]["summary"][key]
            row += f"  {fmt.format(v):>16}"
        print(row)

    return all_results


def sensitivity_rank(
    results: dict,
    top_n:   int = 6,
) -> dict:
    """
    Rank input variables by their Pearson correlation with NPV.
    This is a fast proxy for Sobol first-order sensitivity index.
    (True Sobol indices require SALib — see sensitivity.py.)
    """
    npv_arr = results["npv_with_tv"]

    input_keys = [
        ("input_tam_growth",    "TAM Growth Rate"),
        ("input_share_up",      "Share Capture — Urban Prepaid"),
        ("input_cannibal_up",   "Cannibalization — Urban Prepaid"),
        ("input_cannibal_pp",   "Cannibalization — Urban Postpaid"),
        ("input_cannibal_rural","Cannibalization — Rural"),
        ("input_jio_arpu",      "Jio ARPU Y2+"),
        ("input_capex_mult",    "Capex Overrun Multiplier"),
        ("input_wacc",          "WACC"),
        ("input_arpu_comp",     "Incumbent ARPU Compression"),
    ]

    corrs = {}
    for key, label in input_keys:
        if key in results:
            r = np.corrcoef(results[key], npv_arr)[0, 1]
            corrs[label] = r

    ranked = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nPRELIMINARY SENSITIVITY (Pearson r with NPV) — top {top_n}")
    print("─" * 55)
    print(f"  {'Variable':<35} {'r':>8}  {'|r|':>8}")
    for label, r in ranked[:top_n]:
        bar = "█" * int(abs(r) * 30)
        print(f"  {label:<35} {r:>+8.4f}  {abs(r):>8.4f}  {bar}")
    print("  (Full Sobol indices via SALib — run sensitivity.py)")

    return dict(ranked)


if __name__ == "__main__":
    print("=" * 55)
    print("MONTE CARLO ENGINE — HOUR 3 TEST RUN")
    print("=" * 55)

    # 1. Validate base case first
    print("\nStep 0: Validate deterministic base case")
    validate_base_case(verbose=True)

    # 2. Run single strategy simulation
    print("\nStep 1: Run N=10,000 Aggressive strategy")
    res = run_simulation(N=10_000, seed=42, strategy="Aggressive", verbose=True)

    # 3. Preliminary sensitivity
    print("\nStep 2: Preliminary sensitivity ranking")
    sensitivity_rank(res)

    # 4. Quick convergence check
    print("\nStep 3: Convergence test")
    convergence_test([500, 1_000, 3_000, 5_000, 10_000])

    # 5. All three strategies
    print("\nStep 4: All strategies comparison")
    run_all_strategies(N=5_000, seed=42)
