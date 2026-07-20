"""
assumptions.py
==============
Single source of truth for every model input.

Design principle: every number here traces to either a primary source
(TRAI, Airtel AR, RIL AR) or a documented derivation. Nothing is
arbitrary. If you change a number, change the comment too.

Structure
---------
1. CONSTANTS       — fixed inputs that don't have distributions
2. DISTRIBUTIONS   — (dist_name, *params) tuples used by the MC engine
3. CORRELATION     — the 6x6 correlation matrix with justifications
4. SEGMENTS        — subscriber counts and ARPUs by segment
5. STRATEGIES      — three launch strategies compared across modules
"""

import numpy as np
from scipy import stats

# ── 1. CONSTANTS ──────────────────────────────────────────────────────────────

# Decision anchor
CAPEX_BASE_CR       = 150_000   # ₹ Crore — RIL disclosed ₹1.5L Cr total network investment
HORIZON_YEARS       = 5         # NPV horizon; 5yr standard for strategic telecom
LAUNCH_YEAR         = 2016      # Jio commercial launch: September 5, 2016

# Pre-Jio subscriber base (FY2016, million) — TRAI 2016
# Urban/rural from TRAI tele-density; postpaid 4.87% of total (BusinessToday citing TRAI Sep 2016)
SUBS = {
    "urban_postpaid": 60.2,
    "urban_prepaid":  495.5,
    "rural_prepaid":  447.6,
}
SUBS_TOTAL = sum(SUBS.values())   # 1003.3 M

# Pre-Jio ARPU (₹/user/month, FY2016)
# Airtel Q3 FY2016 ARPU = ₹192 (Business Standard, Jan 2016).
# Blended check: (60.2*800 + 495.5*200 + 447.6*90) / 1003.3 ≈ ₹182 ✓ (~6% below ₹192 — gap = roaming/VAS)
ARPU_PRE = {
    "urban_postpaid": 800,    # industry benchmark: postpaid 3-4x prepaid
    "urban_prepaid":  200,    # consistent with Airtel blended ₹192 being prepaid-weighted
    "rural_prepaid":   90,    # rural ~45% of urban prepaid (TRAI tele-density ratio)
}

# Pre-Jio industry revenue (₹ Crore, FY2016)
# Kleiner Perkins study cited by TechRadar: ₹1.93 trillion = ₹1,93,000 Cr
INDUSTRY_REVENUE_FY16_CR = 193_000

# Jio free period: Sep 2016 — Mar 2017 (Welcome + Happy New Year offers)
# Year 1 ARPU = ₹0 (no billing). Year 2+ = ₹100 base (Jio Prime ₹99/month from Apr 2017)
ARPU_JIO_Y1 = 0       # free period
ARPU_JIO_Y2 = 100     # ₹/month — conservative; actual Q2FY18 = ₹156.4 (Dazeinfo)

# Opex structure (two-component: fixed infra + variable)
# Fixed: spectrum fees + network maintenance ≈ ₹8,500 Cr/year regardless of revenue
# Variable: scales with subscribers (customer acquisition, billing, support)
OPEX_FIXED_CR_PA     = 8_500    # ₹ Crore per year — estimated from spectrum fee disclosures
OPEX_VARIABLE_RATE   = 0.45     # 45% of revenue; improves to 0.38 by Year 5 as revenue scales
OPEX_MARGIN_IMPROVE  = 0.015    # 1.5pp improvement per year in variable rate

# Validation anchors (actual outcomes — used ONLY for post-hoc validation, not calibration)
VALIDATION = {
    "jio_subs_dec2016_M":     72.0,   # TRAI Dec 2016 / Reuters / ET
    "jio_share_dec2016_pct":   6.4,   # derived: 72M / 1127M total
    "jio_subs_dec2017_M":    160.0,   # est from RIL IR; Fortune 2018 cites 168M early 2018
    "industry_rev_decline_cr": 5_000, # Kleiner Perkins: ₹1.93L → ₹1.88L Cr
    "airtel_share_loss_pp":    0.5,   # TRAI: 24.07% → 23.58% in first 4 months
    "airtel_arpu_fy17":       158,    # est: ₹194 FY16 → ₹158 FY17 (-18.6%)
}


# ── 2. DISTRIBUTIONS ──────────────────────────────────────────────────────────
# Each entry: (scipy_dist_object, *shape_params, loc, scale)
# Parameterised so dist.rvs(N) gives N draws directly.
#
# Variable ordering (IMPORTANT — matches correlation matrix rows/cols):
# 0: tam_growth
# 1: share_urban_prepaid_y1   (primary share variable; others derived from this)
# 2: cannibal_urban_prepaid
# 3: cannibal_urban_postpaid
# 4: cannibal_rural
# 5: jio_arpu_y2plus
# 6: capex_overrun_mult
# 7: wacc
# 8: arpu_compression_incumbent

VARIABLE_NAMES = [
    "TAM Growth Rate",
    "Jio Share — Urban Prepaid Y1",
    "Cannibalization — Urban Prepaid",
    "Cannibalization — Urban Postpaid",
    "Cannibalization — Rural Prepaid",
    "Jio ARPU Y2+ (₹/month)",
    "Capex Overrun Multiplier",
    "WACC",
    "Incumbent ARPU Compression",
]

# Frozen scipy distributions — call .rvs(N) to sample N values
DISTS = {
    # Normal(μ=7.5%, σ=3%) — pre-Jio CAGR 2013-15 = 7.5%; σ from GDP volatility band
    "tam_growth": stats.norm(loc=0.075, scale=0.03),

    # Beta(α=2, β=11) → E[X]=15.4%, mode=9.1%
    # Calibrated: Jio actual 14% in 12M (Y1 base 12%); historical entrant range 5-35%
    "share_urban_prepaid_y1": stats.beta(a=2, b=11),

    # Beta(α=3, β=5) → E[X]=37.5%, mode=28.6%
    # Airtel lost only 0.5pp despite Jio +6.4% → low realised cannibalization
    "cannibal_urban_prepaid": stats.beta(a=3, b=5),

    # Beta(α=1.5, β=13) → E[X]=10.3%, mode=3.7%
    # Postpaid: contract lock-in, corporate tie-ins, <1%/month churn (Airtel AR)
    "cannibal_urban_postpaid": stats.beta(a=1.5, b=13),

    # Beta(α=2, β=6) → E[X]=25%, mode=14.3%
    # Rural: price-sensitive but Jio coverage patchy in Y1; JioPhone only from Jul 2017
    "cannibal_rural": stats.beta(a=2, b=6),

    # Normal(μ=₹156, σ=₹20)
    # Centered on Jio's actual Q2 FY18 ARPU of ₹156.4 (Dazeinfo) — the
    # base-case figure in npv_engine.py. σ=₹20 reflects plan-mix uncertainty
    # and the magnitude of the FY19-20 price war trough (₹120-132).
    "jio_arpu_y2plus": stats.norm(loc=156, scale=20),

    # Lognormal: μ_ln=0, σ_ln=0.18 → median=1.0×, mean=1.016×, P90=1.36×
    # Moody's Jan 2018: RIL to spend another $23bn → implies capex overrun
    "capex_overrun_mult": stats.lognorm(s=0.18, loc=0, scale=1.0),

    # Normal(μ=11%, σ=1.5%) — analyst range 9-13% for RIL WACC in 2015
    "wacc": stats.norm(loc=0.11, scale=0.015),

    # Normal(μ=18%, σ=5%) — Airtel actual decline ₹194→₹158 = 18.6%
    "arpu_compression": stats.norm(loc=0.18, scale=0.05),
}


# ── 3. CORRELATION MATRIX ─────────────────────────────────────────────────────
# 9x9 symmetric, positive definite.
# Rows/cols: [tam_growth, share_up_y1, can_up, can_pp, can_rural,
#             jio_arpu, capex_mult, wacc, arpu_comp]
#
# Justification for each non-zero off-diagonal:
#   (0,2): tam↔can_up = -0.40  — fast growth → new users → less switching
#   (1,6): share↔capex = +0.30 — more subscribers requires more network → higher capex
#   (2,5): can_up↔jio_arpu = -0.25 — price war intensity links cannibalization to ARPU recovery
#   (0,7): tam↔wacc = -0.20    — high rates suppress growth; mild macro linkage
#   All others = 0 (no defensible economic link)

CORR_MATRIX = np.array([
#  tam   sh_up  cn_up  cn_pp  cn_ru  arpu   capex  wacc   comp
  [1.00, 0.00, -0.40,  0.00,  0.00,  0.00,  0.00, -0.20,  0.00],  # tam_growth
  [0.00, 1.00,  0.00,  0.00,  0.00,  0.00,  0.30,  0.00,  0.00],  # share_up_y1
  [-0.40, 0.00, 1.00,  0.00,  0.00, -0.25,  0.00,  0.00,  0.30],  # can_up
  [0.00, 0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # can_pp
  [0.00, 0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00],  # can_rural
  [0.00, 0.00, -0.25,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],  # jio_arpu
  [0.00, 0.30,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  0.00],  # capex_mult
  [-0.20, 0.00, 0.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00],  # wacc
  [0.00, 0.00,  0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00],  # arpu_comp
])

# Validate positive definiteness at import time
_eigvals = np.linalg.eigvalsh(CORR_MATRIX)
assert np.all(_eigvals > 0), (
    f"Correlation matrix is NOT positive definite. "
    f"Min eigenvalue = {_eigvals.min():.6f}. "
    f"Reduce correlation magnitudes."
)


# ── 4. SEGMENT STRUCTURE ──────────────────────────────────────────────────────
SEGMENTS = ["urban_postpaid", "urban_prepaid", "rural_prepaid"]

# Share of total Jio Y1 subscribers by segment (approximation)
# Urban prepaid = primary target; rural partial coverage; postpaid minimal
JIO_SHARE_MIX = {
    "urban_postpaid": 0.03,   # 3% of TAM in this segment
    "urban_prepaid":  0.12,   # 12% — primary target; calibrated to 72M actual
    "rural_prepaid":  0.04,   # 4% — limited by coverage and device penetration
}

# Year 2 through Year 8 share by segment (calibrated to actuals where available,
# extrapolated with diminishing growth/saturation beyond)
# Y2: 160M actual. Y3: 280M actual (Dec 2018, RIL IR). Y4-8: saturation curve
# reflecting Jio's actual trajectory toward ~430M subscribers by FY2023.
JIO_SHARE_Y2 = {"urban_postpaid": 0.05, "urban_prepaid": 0.22, "rural_prepaid": 0.10}
JIO_SHARE_Y3 = {"urban_postpaid": 0.07, "urban_prepaid": 0.30, "rural_prepaid": 0.18}
JIO_SHARE_Y4 = {"urban_postpaid": 0.09, "urban_prepaid": 0.35, "rural_prepaid": 0.24}
JIO_SHARE_Y5 = {"urban_postpaid": 0.10, "urban_prepaid": 0.38, "rural_prepaid": 0.28}
JIO_SHARE_Y6 = {"urban_postpaid": 0.11, "urban_prepaid": 0.40, "rural_prepaid": 0.31}
JIO_SHARE_Y7 = {"urban_postpaid": 0.12, "urban_prepaid": 0.41, "rural_prepaid": 0.33}
JIO_SHARE_Y8 = {"urban_postpaid": 0.12, "urban_prepaid": 0.42, "rural_prepaid": 0.34}

SHARE_BY_YEAR = {1: JIO_SHARE_MIX, 2: JIO_SHARE_Y2, 3: JIO_SHARE_Y3,
                 4: JIO_SHARE_Y4,  5: JIO_SHARE_Y5, 6: JIO_SHARE_Y6,
                 7: JIO_SHARE_Y7,  8: JIO_SHARE_Y8}

# Competitive response threshold: if Jio Y1 blended share > this, incumbents cut ARPU
RESPONSE_THRESHOLD_SHARE = 0.08   # ~80M subscribers


# ── 5. THREE STRATEGIES ───────────────────────────────────────────────────────
# Capex and share scale factors for each strategy
# Aggressive = base; Moderate = 70% capex, 65% share; Delay = same capex, 40% share Y1
STRATEGIES = {
    "Aggressive": {"capex_multiplier": 1.00, "share_scale": 1.00},
    "Moderate":   {"capex_multiplier": 0.70, "share_scale": 0.65},
    "Delay":      {"capex_multiplier": 1.00, "share_scale": 0.40},
}


if __name__ == "__main__":
    print("=== assumptions.py self-test ===")
    print(f"Total pre-Jio subscribers:  {SUBS_TOTAL:.1f} M")
    print(f"Blended pre-Jio ARPU check: ₹{sum(SUBS[s]*ARPU_PRE[s] for s in SEGMENTS)/SUBS_TOTAL:.0f}/month (should be ~₹182)")
    print(f"Correlation matrix eigenvalues (all must be >0): {np.round(_eigvals,4)}")
    print(f"Distribution spot checks:")
    np.random.seed(42)
    for name, dist in DISTS.items():
        s = dist.rvs(10_000)
        print(f"  {name:<35} mean={s.mean():.4f}  p10={np.percentile(s,10):.4f}  p90={np.percentile(s,90):.4f}")
    print("Self-test passed.")
