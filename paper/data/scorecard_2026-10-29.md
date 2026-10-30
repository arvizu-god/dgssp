# Scorecard — 2026-10-29

Written by `paper/scripts/score_preregistration.py` from the pre-registration `paper/hardware/preregistration_2026-10-29.json` (sha256 `cee4f23f9999e8143bbe7d04c15ffc89f199cca8e3ac6d3adc10fd2934880816`; recorded at submission: cee4f23f9999e8143bbe7d04c15ffc89f199cca8e3ac6d3adc10fd2934880816) and the records in `paper/data/hw_2026-10-29` (executor hardware).

| ID | Question | Tested | Verdict | Sentence |
|---|---|---|---|---|
| Q1 | Does alpha_hat hold on a third day? | yes | fitted not held; out_of_sample not held | On a third day, alpha_hat missed 1 of the 5 circuits it was fitted to by more than 0.05 at lambda = 1 (RMS 0.046, largest 0.085). On circuits it was not fitted to, alpha_hat missed 3 of 3 by more than 0.05 (RMS 0.212, largest 0.350). |
| Q2 | All errors x alpha_all, or CZ errors x alpha_2q? | yes | out_of_sample indistinguishable; fitted all better | [out_of_sample] The third day does not separate the two models (RMS 0.212 vs 0.214, margin 0.005). [fitted] Scaling every error fits the third day better than scaling the CZ errors alone (RMS 0.046 vs 0.060). |
| Q3 | Does the grid alone decide? | yes | confirmed | On the same circuit, layout, shots and job, the odd grid could not resolve the extrapolation (identified, not resolved) and the gentle grid did: the grid alone decided. |
| Q4 | Does the error formula predict hardware? | yes | Q4a held; Q4c held | Where its predicted relative error was at most 0.2, the propagated error matched the bootstrap on every resolved hardware fit (ratios P-n2-M1-11_L1/gentle 1.02, P-n2-M1-11_L1/rule 1.01, P-n3-M1-17_r1_L1/gentle 1.04). The grid rule's predicted gain held: the gentle grid's error was 1.88 times the rule grid's, against 1.74 predicted. |
| Q5 | Predicted ZNE status | reported | reported | The pre-registered status matched on 4 of 6 fits; the predictions expected 6.0. |
| Q6 | Is alpha_r a property of the circuit? | yes | not reproducible | A circuit's alpha did not reproduce on a third day (chi2/dof 8.04 after a common shift of -0.59 (vs 2026-10-25), -0.57 (vs 2026-10-28)): the scatter between circuits is not a fixed property of each circuit. |
| Q7 | P-n3-M2-17: layout or day? | yes | L2=random,L1=fine | a persistent defect of the day-2 layout that the calibration does not show |
| Q8 | Does gentle-grid ZNE reach n = 4? | yes | does not reach | At n = 4, r = 1 the gentle grid did not resolve the extrapolation (identified, not resolved); the unmitigated value is 0.075. |
| Q9 | Does the P-n4-M1-10_r2 residual reproduce? | reported | reported | The day-1 residual of P-n4-M1-10_r2 at lambda = 1 was +0.022 (z +3.5); on the third day it was -0.008 (z -3.8). |

## Named grids, each fitted on the same counts

| Record | Grid | Status (pred.) | Reported [CI] | SE_boot | SE_formula | SE_pred |
|---|---|---|---|---|---|---|
| P-n2-M1-11_L1 | gentle | resolved (resolved) | 0.940 [0.871, 1.000] | 0.038 | 0.037 | 0.055 |
| P-n2-M1-11_L1 | rule | resolved (resolved) | 0.934 [0.895, 0.975] | 0.020 | 0.020 | 0.032 |
| P-n3-M1-17_L1 | odd | identified, not resolved (not identifiable) | 0.245 [-, -] | 0.046 | 0.039 | - |
| P-n3-M1-17_L1 | gentle | resolved (resolved) | 0.659 [0.521, 0.839] | 0.081 | 0.081 | - |
| P-n3-M1-17_r1_L1 | gentle | resolved (resolved) | 0.692 [0.590, 0.823] | 0.060 | 0.057 | 0.076 |
| P-n4-M1-10_r1_L1 | gentle | identified, not resolved (resolved) | 0.075 [-, -] | 4.578 | 0.522 | - |

## lambda = 1 against the prediction (Q1, Q2)

| Record | Set | Observed | Pred. all | Dev. all |
|---|---|---|---|---|
| P-n2-M1-11_L1 | fitted | 0.585 | 0.500 | +0.085 |
| P-n3-M1-17_L1 | fitted | 0.245 | 0.195 | +0.050 |
| P-n3-M1-17_r1_L1 | fitted | 0.307 | 0.287 | +0.020 |
| P-n4-M1-10_r1_L1 | fitted | 0.075 | 0.096 | -0.021 |
| P-n4-M1-10_r2_L1 | fitted | 0.065 | 0.073 | -0.008 |
| P-n2-M1-11_L2 | out_of_sample | 0.260 | 0.610 | -0.350 |
| P-n3-M1-17_L2 | out_of_sample | 0.290 | 0.222 | +0.068 |
| P-n3-M1-17_r1_L2 | out_of_sample | 0.358 | 0.269 | +0.089 |

## alpha_r (Q6)

| Record | Ref. day | alpha_r ref | alpha_r day 3 | Difference | Included |
|---|---|---|---|---|---|
| P-n2-M1-11_L1 | 2026-10-25 | 3.02±0.08 | 2.35±0.04 | -0.67 | yes |
| P-n3-M1-17_L1 | 2026-10-25 | 3.10±0.11 | 2.40±0.06 | -0.69 | yes |
| P-n3-M1-17_r1_L1 | 2026-10-25 | 3.76±0.22 | 2.91±0.08 | -0.85 | yes |
| P-n4-M1-10_r1_L1 | 2026-10-25 | 2.47±0.15 | 4.57±0.44 | +2.10 | yes |
| P-n4-M1-10_r2_L1 | 2026-10-25 | 2.18±0.20 | 4.86±1.33 | +2.68 | day 3: at random (lambda = 1 less than 2 sigma above P_u) |
| P-n3-M2-17_L1 | 2026-10-25 | 3.43±0.13 | 3.07±0.08 | -0.36 | yes |
| P-n3-M2-17_L2 | 2026-10-28 | 8.00±0.21 | 7.27±0.65 | -0.72 | reference: alpha_r at the grid edge; reference: day-2 out-of-sample failure (at random on every circuit); day 3: at random (lambda = 1 less than 2 sigma above P_u) |
| P-n2-M1-11_L2 | 2026-10-28 | 3.07±0.05 | 8.00±0.00 | +4.93 | day 3: alpha_r at the grid edge |
| P-n3-M1-17_L2 | 2026-10-28 | 2.98±0.07 | 2.34±0.05 | -0.63 | yes |
| P-n3-M1-17_r1_L2 | 2026-10-28 | 2.65±0.06 | 2.12±0.04 | -0.53 | yes |

Q7 cell `L2=random,L1=fine`: L1 fine (P 0.351 vs pred. 0.338, 23.4 sigma above random), L2 random (P 0.253 vs pred. 0.339, 1.2 sigma above random)

## Reported, not tested: layer fidelity

Days 1-2: 2026-10-25 EPLG 0.00613, EPLG / median CZ 2.35; 2026-10-28 EPLG 0.00555, EPLG / median CZ 2.19. Day 3: EPLG 0.00513, EPLG / median CZ 2.11.

## Deviations

None.
