# Stochastic Market Entry & Value Cannibalization Forecaster

**A decision-support model evaluating whether Reliance should have launched Jio aggressively in 2016 — built as a strategy/finance portfolio project, not a prediction engine.**

This is not a model that forecasts subscriber growth. It is a model that answers a decision question: *given everything Reliance could reasonably have known in 2015, should they have committed ₹1.5 lakh crore to an aggressive telecom launch, knowing it would cannibalize the existing industry's revenue?*

---

## The one-paragraph pitch

Most market-entry case studies stop at a single NPV number from a deterministic DCF. This project instead treats every key assumption — market share capture, cannibalization rate, ARPU realisation, capex overrun, discount rate — as a probability distribution, runs 10,000 Cholesky-correlated Monte Carlo scenarios, and produces a full risk picture: the probability the investment destroys value, which single assumption matters most (Sobol sensitivity analysis), and where exactly the break-even boundary sits in two-dimensional assumption space. The model is then validated year-by-year against eight years of actual Jio performance data, and it is honest about exactly where it succeeds and where it fails.

## Headline finding

| Metric | Value |
|---|---|
| Base-case NPV (8-year horizon + terminal value) | **+₹51,085 Cr** |
| Monte Carlo mean NPV (N=10,000) | **+₹69,808 Cr** |
| Monte Carlo median NPV | **−₹11,115 Cr** |
| P(value destruction) | **51.9%** |
| Single largest driver of NPV uncertainty (Sobol) | **Jio's long-run ARPU**, not capex |

The mean and median tell different stories — mean is pulled positive by a fat right tail (genuine upside optionality), while the median scenario is mildly negative. **This was a close call, not an obvious one**, which is the model's central finding and the reason a single deterministic NPV number would have been misleading either way.

---

## Repository structure

```
assumptions.py                      All input distributions, correlation matrix, segment data — single source of truth
npv_engine.py                       Core deterministic NPV calculation (compute_npv) — the function everything else calls
monte_carlo.py                      Cholesky-correlated Monte Carlo engine (Gaussian copula, N=10,000 scenarios)
test_npv_engine.py                  14 unit tests on the NPV engine's core formulas

visualize.py                        Convergence testing + NPV distribution / CDF plots
cannibalization_decomposition.py    Segment-level cannibalization cost breakdown + NPV waterfall
sensitivity.py                      Sobol sensitivity analysis (SALib) — what actually drives the outcome
break_even_frontier.py              2D break-even contour: share capture × cannibalization rate

strategy_comparison.py              Aggressive / Moderate / Delay strategy comparison
risk_adjusted_selection.py          Three decision lenses: certainty-equivalent utility, VaR constraints, stochastic dominance
competitive_response_stresstest.py  Stress-tests the Stage-2 incumbent-response mechanism

historical_backtest.py              Year-by-year backtest against actual 2016-2024 Jio performance
correlation_robustness.py           Tests how much the correlation structure itself actually matters
validation_tightening.py            Documents calibration history + circularity check on validation claims

main.py                             Single entrypoint — runs the full pipeline end to end
master_interview_prep.py            Night-before reference: every number, finding, and disclosed limitation in one script
requirements.txt                    Python dependencies
canonical_numbers.json               Single source of truth — one canonical run's numbers, audited against every document
outputs/                            Generated charts (PNG) and the executive decision memo (DOCX)
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This runs the full pipeline: validates the base case, runs the 10,000-scenario Monte Carlo, generates every chart, runs the Sobol sensitivity analysis, and prints the final recommendation. Takes under two minutes on a laptop.

To run any individual module's analysis and see its own detailed output:
```bash
python npv_engine.py              # deterministic base case, verbose
python monte_carlo.py             # Monte Carlo + convergence + strategy comparison
python sensitivity.py             # Sobol tornado chart
python historical_backtest.py     # year-by-year validation
python test_npv_engine.py         # unit test suite
```

---

## Methodology in brief

**Decision framing.** Three strategic postures are compared: Aggressive (full pan-India 4G launch, free-data acquisition — what Reliance actually did), Moderate (lower capex, slower rollout), and Delay (wait and gather information). The model evaluates risk-adjusted NPV for each, not just expected value.

**Uncertainty modelling.** Nine inputs are modelled as probability distributions, not point estimates — Beta for bounded variables like market share and cannibalization rate (right-skewed, calibrated by method of moments to historical telecom-entry cases), Normal for symmetric estimation uncertainty (WACC, ARPU), Lognormal for capex overrun (asymmetric — projects rarely come in under budget).

**Correlation structure.** Four economically-justified correlations are built into a Cholesky-decomposed covariance matrix: TAM growth and cannibalization rate are negatively correlated (ρ=−0.40 — a fast-growing market has new users to absorb, reducing the need to steal share), market share and capex overrun are positively correlated (ρ=+0.30 — serving more subscribers requires more infrastructure spend), and two smaller links round out the structure. **Day 4's robustness testing found this correlation structure has a smaller effect on the headline NPV distribution than initially assumed** (see `correlation_robustness.py`) — disclosed honestly rather than overclaimed.

**The cannibalization mechanic** is the project's core novelty: cannibalization loss is computed segment-by-segment (urban postpaid, urban prepaid, rural prepaid) rather than as one blended number, because switching propensity differs fundamentally by contract type. 85% of total cannibalization cost is concentrated in urban prepaid alone.

**Competitive response** is modelled as a second stage: if Jio's Year-1 share exceeds an 8% threshold, incumbents are triggered to cut ARPU on their *retained* subscribers in subsequent years — a second-order value-destruction channel most market-entry models miss entirely. This creates a real, disclosed discontinuity in the NPV function at the threshold (~₹2,56,000 Cr jump), which is flagged as a known modelling simplification rather than smoothed away.

**Terminal value.** A 5-year telecom-revenue-only DCF understates a 15-20 year infrastructure asset by construction. The model uses an 8-year explicit horizon plus a Gordon Growth terminal value on Year-8 EBITDA — consistent with how Jio's actual EBITDA only turned solidly positive around Year 4.

---

## Validation — what actually checks out, and what doesn't

Three end-point validation checks against independent post-2016 data:

| Metric | Model | Actual | Error | Genuinely out-of-sample? |
|---|---|---|---|---|
| Jio Year-1 subscribers | 79.2M | 72.0M | +10.0% | Partially — the share table was tuned toward this |
| Industry revenue decline (Y1) | ₹4,997 Cr | ₹5,000 Cr | −0.1% | Yes |
| Airtel market share loss | 0.55pp | 0.50pp | +9.2% | Yes |

**The year-by-year backtest (2016-2024) tells a more complete and more honest story** than these three endpoints alone: the model tracks subscriber growth accurately through Year 2, then increasingly *under-predicts* from Year 3 onward (reaching −18% to −26% by Year 4-8) because it didn't fully anticipate how Vodafone-Idea's financial collapse would accelerate Jio's gains beyond organic growth. The ARPU trajectory miss is even more structural: the model assumes flat 3%/year growth, but actual ARPU followed a sharp V-shape — falling from ₹156 to a ₹128 trough in 2019-20 during the price war, then recovering past ₹175 after the December 2019 and November 2021 industry-wide tariff hikes. A static growth-rate assumption cannot capture a regulatory/competitive-cycle-driven V-shape; that would require a regime-switching model.

## Known limitations (disclosed, not hidden)

- **Competitive response is a binary on/off trigger**, not a smooth probability function — creates a real discontinuity in the NPV surface, confirmed and quantified by both `competitive_response_stresstest.py` and the unit test suite.
- **The ARPU growth assumption is flat**, missing the actual price-war V-shape documented in the backtest.
- **Cannibalization uses segment-average ARPU** as a proxy for switcher ARPU; actual switchers were likely below-average, which makes the model's cannibalization cost a conservative (upper-bound) estimate — a deliberate bias, not an oversight.
- **The 18% incumbent ARPU-compression parameter is calibrated to a single historical data point** (Airtel's actual decline), not an independently estimated structural relationship.
- **Tests were written on Day 4, after the model was substantially built**, not test-first. This is disclosed because it's the honest development history — and the test suite caught a real apparent bug (a non-monotonicity in NPV vs. share capture) that turned out to be the disclosed competitive-response discontinuity, not a new error.

## Bugs found and fixed during development (disclosed for transparency)

1. **Unit conversion error** (Day 2): revenue formula used `÷100` instead of the correct `×1.2` to convert ₹/month × million-subscribers into ₹ Crore — silently understated every revenue figure by 10x for two days. Caught by sanity-checking against Jio's actual Q4 FY18 quarterly revenue disclosure.
2. **Share-variable wiring bug** (Day 3): the stochastic market-share draw was generated and stored for sensitivity analysis but never actually passed into the NPV calculation — every Monte Carlo scenario silently used a fixed deterministic share value. This meant competitive response never activated across 10,000 "scenarios." Fixed by properly threading the stochastic draw through as a share-scale multiplier.

---

## Tech stack

Python 3 · NumPy · SciPy (distributions, Cholesky decomposition, root-finding) · SALib (Sobol sensitivity analysis) · Matplotlib · python-docx (executive memo generation)

## Data sources

TRAI Telecom Subscription Data Reports (2013-2024) · Reliance Industries Annual Reports · Bharti Airtel Annual Reports and quarterly results · Kleiner Perkins industry studies · Bernstein and Credit Suisse analyst research · Multiple financial news sources (Business Standard, Economic Times, Dazeinfo, Fierce Network, Inc42) — full citations in `assumptions.py` source comments and `historical_backtest.py`.
