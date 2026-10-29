# ZNE for a sampling algorithm: theory behind the scalar estimator, the gate, the grid and the pointwise failure

Position against Liu & Cai, arXiv:2502.11285 ("Quantum error mitigation for
sampling algorithms"). Every number below was recomputed today from
`paper/data/hw_*` and the code in `src/dgssp/mitigation/zne.py`.

## 0. Liu & Cai in one paragraph

Each bitstring probability is an expectation value, p(z) = Tr(Π_z ρ). Any
*linear* QEM method writes the mitigated state as ρ_em = Σ_i α_i ρ_i
(fixed coefficients, chosen before the experiment: PEC quasi-probabilities,
Richardson weights for fixed scales, ...). So p_em(z) = Σ_i α_i p_i(z) with the
same α_i for every z. Consequences: Σ_z p_em(z) = Σ_i α_i = 1 automatically;
the cost is known in advance, Var ≈ A²/N with A = Σ|α_i|; negatives come only
from shot noise (clip or keep); a signed-sample sampler draws from p_em; and
TV ≤ 2^{n/2−1}√TSE, worst for near-uniform distributions. They also test
hypotheses on single strings (accept p_em(z) > p_th ≈ 1.65σ̂) and mitigate
marginals digit by digit. Simulation only (4–6 qubit QPE); nonlinear
(exponential) ZNE is not treated.

## 1. The model that makes the exponential exact

If the noise on the measured register acts as a depolarising channel whose
strength scales with λ (what gate twirling pushes towards):

    ρ(λ) = F(λ) ρ_0 + (1 − F(λ)) I/2^n,      F(λ) = e^{−cλ}

then for any projector Π with Tr Π = m:

    P_Π(λ) = P_u + (P_Π(0) − P_u) e^{−cλ},   P_u = m/2^n.

For Π = Π_sol, m = M and P_u = M/2^n (random guessing). Hence

    P(0) = P_u + (P(1) − P_u) e^{c}     and     signal kept = e^{−c}.

Zero-noise extrapolation is "divide the observed excess over random guessing
by the survival F(1) = e^{−c}". If F were known, this would be a linear,
Liu–Cai-type correction, p_em = [p − (1−F)u]/F, with cost 1/F². **What our
estimator adds is that c is not known; it is learned from the folded points.**

Evidence the model holds here: one α fits λ = 1, 3, 5 (3.18, 3.21, ≥ 2.1);
every n ≥ 3 λ = 5 point sits at P_u (|z| ≤ 2.3); the gate-count survival
reproduces the full Aer model's kept signal (0.786 vs 0.792 at n = 2).
Limits: relaxation is non-unital (day-2 P-n3-M2-17 skews to 000/001); deep
circuits keep more signal than one α predicts.

## 2. Error propagation (delta method)

Two scales λ1 < λ2, excesses D_j = P(λ_j) − P_u with standard errors σ_j,
κ = λ1/(λ2 − λ1):

    P(0) − P_u = D1 (D1/D2)^κ
    rel. error of the excess = sqrt( (1+κ)²/s1² + κ²/s2² ),   s_j = D_j/σ_j.

It diverges as s2 → 0: the second point must still carry signal. The gate
(every point ≥ 2σ above P_u) enforces s_j ≥ 2.

Check against hardware (weighted 3-point fit, linearised covariance vs the
committed bootstrap):

| record | grid, shots | c | SE (formula) | SE (bootstrap) |
|---|---|---|---|---|
| P-n2-M1-11, job 1 | {1,3,5}, 4096 | 0.70 | 0.035 | 0.035 |
| P-n2-M1-11, job 2 | {1,1.25,1.5}, 12 288 | 0.61 | 0.032 | 0.031 |
| P-n3-M1-17_r1, job 2 | same | 1.08 | 0.053 | 0.050 |
| P-n3-M1-17, job 2 | same | 1.73 | 0.107 | 0.109 |

Price of learning c instead of knowing it (SE ratio to e^{c}σ1): 2.3–4.9.

## 3. The grid rule

With λ1 = 1, equal shots, D2 = D1 e^{−cΔ} (Δ = λ2 − 1):

    rel. error ∝ sqrt( (1 + 1/Δ)² + e^{2cΔ}/Δ² )

For large c this is minimised at **Δ ≈ 1/c**: put the second point where the
excess has fallen by a factor e. The minimum absolute error is
e^{c} σ1 sqrt((1+c)² + e²c²), so shots for a target error ε scale as
e^{2c} [(1+c)² + e² c²] / ε². The e^{2c} is the same exponential cost as any
unbiased method (Liu–Cai's A²); learning c adds only the polynomial factor
≈ 8.4 c².

On our data: n = 3 (kept 0.16, c ≈ 1.8) → Δ ≈ 0.55, i.e. λ2 ≈ 1.5, which is
the gentle grid; n = 2 (c ≈ 0.7) → Δ ≈ 1.4. Predicted for P-n3-M1-17:

| grid | shots | SE of P(0) | z of the λ points |
|---|---|---|---|
| {1,3,5} | 4096 | 0.64 | 19, 0.6, 0 (measured λ = 3: z = 0.7) |
| {1,3,5} | 12 288 | 0.37 | 33, 1.1, 0 |
| {1,1.25,1.5} | 12 288 | 0.12 (measured 0.11) | 33, 23, 15 |

So "more shots would not rescue the odd grid" is a computed statement.

## 4. Why unidentified fits explode (and what the raw numbers are)

With only λ1 informative, χ² is flat in c and its infimum is at c → ∞; the
estimator P_u + D1 e^{c} diverges and stops at the rate cap c_max = 10
(`EXP_RATE_MAX`). Every large raw value on hardware is exactly this:

| record | raw | P_u + (P(1) − P_u) e^{10} |
|---|---|---|
| P-n3-M2-17 | 2275.0 | 2275.0 |
| P-n4-M1-10_r2 | 742.2 | 742.2 |
| P-n3-M1-17_r3 | 457.2 | 457.2 |
| P-n4-M2-02 | 376.6 | 376.6 |

They are not bad estimates; they are the cap. The gate is the condition that
the χ² minimum in c is interior.

## 5. The pointwise failure, in the same language

Pointwise = §1 applied to every Π_z separately with asymptote 2^{−n} and its
own fitted c_z. Unlike Liu–Cai: the weights depend on the data, they differ
from string to string (so Σ_z p̂(z) ≠ 1), and the effective overhead
A_z = e^{ĉ_z} is random and as large as e^{10} ≈ 22 000. (Pointwise
*Richardson* with fixed scales *is* linear and is exactly inside their
framework.)

At M = 0 every string's excess is pure shot noise, so §4 applies string by
string. Hardware, P-n3-M0-06 (n = 3): 5 of 8 strings hit the cap, raw values
129, 48.5, −290, −37.5, −145 (others ≈ 0.13). Clipping keeps 000 and 001;
the simplex projection puts all mass on 000, so D_TV = 1 − 1/8 = 0.875. The
"mitigated answer" is the argmax of a λ = 1 fluctuation. Liu–Cai's TV bound
already flags near-uniform distributions as the worst case for any method;
here the large random overhead makes the worst case happen.

## 6. Use without knowing M

P_u = M/2^n needs M. In a real search, apply the scalar estimator to a
single candidate z (asymptote 2^{−n}, no M needed) or to a classically
verified candidate set (for SSP, checking a sum is cheap). The gate's
"informative point" test is then a one-sided test of "no signal at this noise
level", the same idea as Liu–Cai's p_em(z) > p_th, applied *before*
extrapolating because the overhead is not known in advance.

## 7. Verified at scale (2026-10-29)

Source for every number below: `paper/data/zne_theory.json`, written by
`paper/scripts/zne_theory_check.py`. Provenance is stamped at commit 9dc971c
plus the uncommitted Step 3 code. Keys are given as JSON paths. The functions
are in `src/dgssp/mitigation/zne.py` §11 ("Error propagation of the
fixed-asymptote exponential"); nothing existing was changed. SE_bootstrap =
(hi − lo)/3.92 of the interval stored in each record (B = 500, seed 0, as the
campaigns ran it).

### 7.1 Corrections to §2–§3 found on the way

- **The optimal step is not 1/c.** The minimiser of
  sqrt((1+1/Δ)² + e^{2cΔ}/Δ²) tends to 1.1089/c (`optimal_step.limit`, the
  root of u = 1 + e^{−2u}). The 1/c rule drops the first point's own error.
  In `optimal_step.table`, c·Δ* is 1.2766 at c = 0.5, 1.1331 at c = 4 and
  1.115 at c = 16. So 1/c is 13 % short at c = 4 and about 11 % short in the
  limit. For the hardware rates (`optimal_step.hardware`):

  | record | c | Δ* | 1/c |
  |---|---|---|---|
  | P-n2-M1-11 | 0.6976 | 1.7695 | 1.4334 |
  | P-n3-M1-17_r1, job 2 | 1.0807 | 1.1043 | 0.9254 |
  | P-n3-M1-17, job 2 | 1.725 | 0.6745 | 0.5797 |

  §3's "Δ ≈ 0.55" and "Δ ≈ 1.4" should read 0.67 and 1.77 with the fitted
  rates. `tests/test_zne_error.py` pins the 1.1089 limit (within 5 % at
  c = 4), not 1/c.
- **The delta method is not within 10 % everywhere.** It holds where the
  predicted relative SE (SE/signal) is small; see 7.2.
- **The stored `rate_at_bound` flag misses capped fits.** χ² is flat to
  machine precision near the cap. The golden-section search therefore stops
  about 1e−8 short of c_max, inside the flag's 1e−9 tolerance. Two of the four
  hardware cap records carry `rate_at_bound = False`
  (`hardware.cap_missed_by_stored_flag`: P-n3-M1-17_r3, P-n4-M1-10_r2). The
  flag is unchanged. The new diagnostics use `RATE_CAP_RTOL` = 1e−6.

### 7.2 Monte Carlo (`monte_carlo`, 2000 replicas, seed 0)

Model: P(λ) = 1/8 + 0.82 e^{−cλ}, with c ∈ {0.5, 1, 2}, grids {1,1.25,1.5},
{1,1.5,2} and {1,3,5}, and shots 4096 and 12 288.

- **Small relative SE: the formula holds.** On the 14 designs with predicted
  relative SE ≤ 0.2, `all.ratio_predicted` lies in [0.959, 1.023]. This
  includes c = 1 on {1,3,5}, where the λ = 5 point sits at expected z = 1.0
  (`z_min_expected`). A weak last point does not break the formula, as long
  as the fit is identified.
- **Larger relative SE: the formula is low.** c = 2 on {1,1.25,1.5} at 4096
  shots (relative SE 0.313) gives 0.890. c = 2 on {1,1.5,2} at 4096 (relative
  SE 0.272) gives 0.878. Conditional on the gate (`resolved.ratio_predicted`),
  the second rises to 0.997: the gate drops the replicas whose last point fell
  to random. On the first, every replica passes the gate and the ratio stays
  0.890; there the cause is curvature, not the gate.
- **Not identified.** c = 2 on {1,3,5}: formula/MC = 0.001, mean bias 890
  and 657 (the cap).
- **Bias (reported, not asserted).** It is positive and grows with relative
  SE. `all.bias_mean` ≤ 0.0060 on every design with relative SE ≤ 0.11. At
  relative SE 0.31 and 0.27 it is 0.0473 ± 0.0064 and 0.0486 ± 0.0057, while
  the medians are 0.0057 and 0.0013: the estimator is right-skewed, not
  shifted. On every identified design with relative SE ≤ 0.2, the formula
  interval covers the truth in 0.939–0.958 of replicas
  (`all.coverage_formula`).

### 7.3 Hardware (`hardware`)

- **Resolved fits, SE formula vs bootstrap** (`by_status.resolved`):

  | record | SE formula | SE bootstrap |
  |---|---|---|
  | P-n2-M1-11, job 1 | 0.0346 | 0.0346 |
  | P-n2-M1-11, job 2 | 0.0323 | 0.0308 |
  | P-n3-M1-17_r1, job 2 | 0.0531 | 0.0504 |
  | P-n3-M1-17, job 2 | 0.1072 | 0.1094 |

  The four §2 matches reproduce. `learning_cost` (SE over the known-F SE
  σ1 e^{c}) is 2.25–4.90.
- **Identified, not resolved** (`by_status.identified_not_resolved`):
  P-n3-M1-17_r1 job 1 gives 0.0568 / 0.0569. P-n4-M2-02 job 1 has its rate at
  the cap: the covariance is singular, so the formula is undefined, and the
  bootstrap SE is 153.6.
- **Cap identities** (`cap_identities`, |value − cap|/cap ≤ 1e−6), all job 1,
  relative differences ≤ 9.5e−8:

  | record | raw value = cap |
  |---|---|
  | P-n3-M2-17 | 2274.96 |
  | P-n4-M1-10_r2 | 742.165 |
  | P-n3-M1-17_r3 | 457.217 |
  | P-n4-M2-02 | 376.554 |

  `every_large_value_is_the_cap` = true: these are the only four hardware
  values with |value| > 1.
- **Recomputation.** Recomputing from the counts reproduces the stored
  observations and SEs exactly (`recomputed_equals_stored`: all differences
  0).

### 7.4 Simulation, SE_formula / SE_bootstrap on resolved fits (`simulation_se`)

| grid | resolved fits | median | 5–95 % | coverage, ±1.96 SE_formula | coverage, bootstrap CI |
|---|---|---|---|---|---|
| odd | 41 | 0.989 | 0.452–1.061 | 0.366 | 0.390 |
| odd7 | 13 | 1.012 | 0.930–1.057 | 0.462 | 0.462 |
| planned | 186 | 0.983 | 0.603–1.085 | 0.559 | 0.640 |
| pooled | 240 | 0.987 | 0.589–1.082 | 0.521 | 0.588 |

The acceptance condition (median in [0.85, 1.15]) holds on every grid. Where
the ratio breaks (`pooled.*`; median, with the 5th percentile where given):

- **Relative SE** (`by_relative_se`). This is the same threshold as the Monte
  Carlo.

  | relative SE | median | p5 |
  |---|---|---|
  | < 0.05 | 1.010 | 0.945 |
  | 0.05–0.1 | 1.002 | |
  | 0.1–0.2 | 0.983 | 0.763 |
  | 0.2–0.3 | 0.948 | 0.502 |
  | ≥ 0.3 (n = 33) | 0.732 | 0.301 |

- **Few scales** (`by_n_scales`). 2-scale planned grids: 0.734 (p5 0.263,
  n = 18). 3 scales: 0.993. 4 scales: 1.012.
- **Near the gate** (`by_z_min`). Smallest observed z in [2, 3): 0.947, p5
  0.339. The worst case (`worst[0]`) is P-n6-M3p-19 on the odd grid
  (z = 7.0, 2.06, 2.13): bootstrap SE 128.8 against formula 0.0097. Bootstrap
  replicas fall below the gate and hit the cap, which the linearisation cannot
  see.
- **Large c** (`by_rate`). c ≥ 2.5: 0.612 (n = 4). 1.5–2.5: 0.927. Below
  1.5: ≥ 0.989.
- **By n** (`by_n`). n = 2: 1.013. n = 3: 1.007. n = 4: 0.935. n = 5: 0.944.
  n = 6: 0.842.
- **Outside the gate the formula means nothing.** On identified-not-resolved
  fits the median ratio is 0.881 (odd), 0.950 (odd7) and 0.379 (planned).
  The ranges span 8e−5 to 4e7 (`<grid>.identified_not_resolved`).

**Coverage fails for both intervals, and the cause is bias, not SE.**

- The two coverages are nearly equal on every grid, so the SE is right.
- The median standardised error (value − ideal)/SE is −2.65 (odd), −2.56
  (odd7) and −1.64 (planned). The median bias is −0.063, −0.021 and −0.094
  (`<grid>.overall`).
- So under the fake_marrakesh model the extrapolation lands about two of its
  own SEs below the ideal. The model P(λ) = P_u + b e^{−cλ}, with c learned
  from the folds, under-corrects.
- Hypothesis, not tested here: the folds scale gate noise but not readout or
  idling, so a c learned from λ ≥ 1 understates the decay from 0 to 1.
- This belongs next to the hardware coverage claim. "Every interval contains
  the ideal" on 3 records is not what 240 simulated fits show.

One unidentified odd-grid record (P-n5-M3p-09, SE ~5e9) recomputes 3.6 % away
from its stored SE (`odd.recomputed_equals_stored.max_rel_se_diff`). The cause
is a flat χ² near the cap: golden-section noise between platforms. Every other
record reproduces to ≤ 5e−8.

### 7.5 The grid rule (`grid_rule`)

**Setup.**

- c = −ln kept, from the pooled distinct λ = 1 counts of each instance's
  records. That covers 340 instances with signal: c median 2.37, 5–95 %
  0.69–5.93, and 19 instances with kept ≤ 0.
- The planned grids reproduce from `plan_scales` on 340 of 340
  (`planned_grid_reproduced`).
- 154 of the 340 signal instances got the fallback grid {1, 1.5, 2}, not a
  planned one (`planned_source_counts`). The "typical planned Δ = 0.5/1" in the
  request *is* that fallback. The most common genuine plan is {1, 1.5, 2.5}
  (`typical_planned_grid`).

Resolved fraction, pooled over odd, odd7 and planned (810 grid × instance
pairs, `per_grid.pooled`):

| c·Δλ (first step) | n | resolved |
|---|---|---|
| 0.25–0.5 | 12 | 1.00 |
| 0.5–1 | 167 | 0.99 |
| 1–2 | 127 | 0.35 |
| 2–4 | 229 | 0.06 |
| ≥ 4 | 275 | 0.01 |

| predicted SE / signal | n | resolved |
|---|---|---|
| < 0.05 | 63 | 0.90 |
| 0.05–0.1 | 115 | 0.59 |
| 0.1–0.2 | 90 | 0.51 |
| 0.2–0.5 | 97 | 0.36 |
| 0.5–1 | 51 | 0.39 |
| ≥ 1 | 394 | 0.04 |

How well each predicts "resolved" (`per_grid.pooled.predictors`):

| predictor | accuracy |
|---|---|
| predicted SE/signal ≤ 0.2 | 0.795 |
| c·Δλ in [0.5, 2] | 0.862 |
| both | 0.856 |
| **expected gate** (every point's expected excess ≥ 2σ: the `plan_scales` criterion at k = 2) | **0.942** |

When the expected gate predicts resolution, 0.957 of fits resolve; when it
does not, 0.063 do.

**Verdict: the data do not support the claim "a grid resolves when its
spacing is near 1/c and its predicted SE is small" as stated.**

1. **The first step must be at or below 1/c.** 0.99 of fits resolve for
   c·Δλ < 1, and 0.35 for c·Δλ in 1–2. The SE-optimal step (c·Δ* ≈ 1.16–1.23
   at the hardware rates) sits just past the point where the gate starts to
   fail.
2. **Small predicted SE is neither necessary nor sufficient.** On the planned
   grid, 0.296 of designs with predicted SE/signal > 0.2 resolve. On the odd
   grid only 0.422 of designs with ≤ 0.2 do
   (`per_grid.<grid>.predictors.small_se`).
3. **The expected gate predicts resolution best,** and that is what
   `plan_scales` already does.

So the rule is: plan for the gate first, then minimise SE among the grids that
pass it. That is `plan_scales_spacing`. Caveat: c comes from the same λ = 1
data the gate tests.

**`plan_scales_spacing` vs the campaign's `plan_scales`** (`spacing_vs_planned`).
Same a-priori survival, candidates, k = 3 and 3 points; both scored at the
measured c.

- On the 186 genuinely planned instances, the predicted SE ratio has median
  0.909 (5–95 % 0.807–1.002, max 1.29).
- With the measured c as planner input instead, the median is 0.896 (on 227
  instances).
- So the predicted gain is about 10 %. It is *not* tested: no data were taken
  on these grids.

### 7.6 Pointwise at M = 0 (`pointwise_m0`)

| grid | n | strings at cap, median | instances with ≥ 1 capped string | simplex output = one string | D_TV clip, any cap / none |
|---|---|---|---|---|---|
| odd | 116 | 0.219 | 0.862 | 0.759 | 0.869 (100) / 0.028 (16) |
| odd7 | 60 | 0.125 | 0.750 | 0.583 | 0.866 (45) / 0.028 (15) |
| planned | 116 | 0.203 | 0.879 | 0.759 | 0.868 (102) / 0.039 (14) |

On every record, "simplex output is one string" and "D_TV = 1 − 2⁻ⁿ" coincide
(`fraction_tv_simplex_is_max`). The recomputed D_TV matches the stored
pathology block to ≤ 7e−11.

**The rank correlation between cap fraction and D_TV after clip is weak
pooled, and of the wrong kind.**

- Spearman ρ, pooled: 0.13 (p = 0.16, odd), 0.47 (p = 1.4e−4, odd7), 0.15
  (p = 0.10, planned).
- By n (`by_n`): ρ = 0.86–0.90 at n = 2, and ≤ 0 at n ≥ 4 (n = 5: −0.39 odd,
  −0.56 planned).
- The relation is a step, not a gradient: one capped string is enough. D_TV
  after clip is about 0.87 whenever any string caps and about 0.03 when none
  does. Beyond that, the fraction of strings at the cap does not matter.

Hardware (`hardware.records`):

| record | strings capped | simplex output | D_TV (simplex) |
|---|---|---|---|
| P-n2-M0-06 | 0 of 4 | | 0.029 |
| P-n3-M0-06 | 5 of 8 | one string | 0.875 |
| P-n4-M0-09 | 7 of 16 | one string | 0.9375 |

§5's "5 of 8" reproduces.

### 7.7 Linearity (`linearity`)

**Pointwise Richardson with fixed scales sums to one before renormalisation on
every record,** so it sits inside the Liu–Cai framework. Max |Σ − 1|:

| records | max \|Σ − 1\| |
|---|---|
| odd, 463 | 0 |
| odd7, 197 | 0 |
| planned, 463 | 2.6e−15 |
| hardware, 19 | 2.0e−15 |

**Pointwise exponential does not:**

| records | median Σ_z raw(z) | 5–95 % | off by more than 1 |
|---|---|---|---|
| odd | −23.2 | [−1238, 368] | 0.88 |
| odd7 | 0.876 | [−1302, 763] | 0.77 |
| planned | 0.0086 | [−341, 151] | 0.74 |
| hardware | 0.981 | [−699, 490] | 0.79 |

The data-dependent per-string rates make the map nonlinear, and the cap makes
it unbounded.

### 7.8 Figure

`paper/figures/fig_zne_grid_theory.{pdf,png}`, drawn from
`zne_theory.json.figure` and `optimal_step.hardware`.

- (a) Two-point relative SE (in units of σ1/D1) against Δλ for the three
  resolved hardware rates, with each optimum marked. Ticks mark the odd
  {1,3,5}, gentle {1,1.25,1.5}, most common planned {1,1.5,2.5} and fallback
  {1,1.5,2} grids.
- (b) SE_formula against SE_bootstrap on every resolved simulated fit (by
  grid) and on the hardware fits, with the identity line.
