# Pre-registration — job3, plan date 2026-10-29

Written by `paper/scripts/preregister.py`; do not edit by hand. The JSON beside this file is the record; its sha256 is printed by the script and goes on the checklist.

- manifest `paper/hardware/manifest_2026-10-29.json`, sha256 `8fcfe9bb675656a7869793915b32a0cc2ead1fcc447d2f5e5248758a7e06691e`
- design `paper/hardware/design_job3.json`, sha256 `ba3a9ceb110f009b7533f40c36bd60bac535aac27a4837b2d662e4e93258f604`
- calibration `paper/hardware/calibration_2026-10-29.json`, stamp 2026-10-29T21:44:40-06:00
- code commit `34a1d964dc080c780f26dab3ed24b313e6f852fa` (dirty: False)
- REPORT_BOUNDS = clip, RESOLVED_RULE = v2, 12288 shots per PUB
- alphas: all = 3.176 (all errors), 2q = 4.055 (2q errors); Q4b band alpha_r = 2.399, 3.512 (quantiles [0.16, 0.84] of 12 records)

## Questions, rules and what is predicted

| ID | Question | Tested | Rule | Prediction |
|---|---|---|---|---|
| Q1 | Does alpha_hat hold on a third day? | yes | per set: 'held' iff every record's \|P_hw - P_pred\| <= threshold_abs; RMS and max reported | fitted: P-n2-M1-11_L1 0.500, P-n3-M1-17_L1 0.195, P-n3-M1-17_r1_L1 0.287, P-n4-M1-10_r1_L1 0.096, P-n4-M1-10_r2_L1 0.073; out_of_sample: P-n2-M1-11_L2 0.610, P-n3-M1-17_L2 0.222, P-n3-M1-17_r1_L2 0.269 |
| Q2 | All errors x alpha_all, or CZ errors x alpha_2q? | yes | per set: 'not decidable by design' if the predicted RMS of (P_all - P_2q) < decidable_min_predicted_rms_difference; else '<model> better' iff its RMS is lower by >= margin; else 'indistinguishable' | out_of_sample: predicted RMS(P_all - P_2q) 0.021 -> decidable; fitted: predicted RMS(P_all - P_2q) 0.015 -> decidable |
| Q3 | Does the grid alone decide? | yes | 'confirmed' iff odd status in odd_statuses and gentle status == gentle_status; else 'not confirmed' | odd not identifiable, gentle resolved; P(confirmed) = 1.00 |
| Q4 | Does the error formula predict hardware? | yes | Q4a: 'held' iff every eligible fit's ratio is inside window; 'not tested' if no fit is eligible; Q4b: the prospective prediction: SE_boot / SE_pred(alpha_all), with the SE_pred band at the alpha_r quantiles (alpha_r_band); Q4c: 'held' iff \|observed / predicted - 1\| <= tolerance_rel | Q4a eligible: P-n2-M1-11_L1/gentle, P-n2-M1-11_L1/rule, P-n3-M1-17_r1_L1/gentle; Q4c ratio 1.74 |
| Q5 | Predicted ZNE status | reported | reported: how many fits got the pre-registered status (the gate at the expected values), against the number expected if the predictions are right (the sum of each pre-registered status's bootstrap probability) | 6 fits; expected matches 6.0 |
| Q6 | Is alpha_r a property of the circuit? | yes | 'reproducible per circuit' iff chi2/dof of the differences, after removing one weighted-mean shift per reference day (dof = n - number of reference days), <= max_chi2_red_differences AND the day-3 between-circuit chi2/dof about its weighted mean > min_chi2_red_between_day3; 'not reproducible' if the first fails; 'no scatter to reproduce' if only the second fails; Spearman reported, not tested | P-n2-M1-11_L1 3.02±0.08, P-n3-M1-17_L1 3.10±0.11, P-n3-M1-17_r1_L1 3.76±0.22, P-n4-M1-10_r1_L1 2.47±0.15, P-n4-M1-10_r2_L1 2.18±0.20, P-n3-M2-17_L1 3.43±0.13, P-n3-M2-17_L2 8.00±0.21 (excl.), P-n2-M1-11_L2 3.07±0.05, P-n3-M1-17_L2 2.98±0.07, P-n3-M1-17_r1_L2 2.65±0.06 |
| Q7 | P-n3-M2-17: layout or day? | yes | random: P_hw < P_u + random_k_sigma * sigma(P_hw); fine: not random and P_hw >= P_pred - fine_threshold; degraded: not random and P_hw < P_pred - fine_threshold | L1: P_pred 0.338, P_u 0.250, decidable; L2: P_pred 0.339, P_u 0.250, decidable |
| Q8 | Does gentle-grid ZNE reach n = 4? | yes | 'reaches' iff the gentle fit is resolved; 'does not reach' otherwise (the lambda = 1 value is reported); 'not tested' if the record's folds were trimmed. Whether the interval contains the ideal is reported, not tested | resolved, P(resolved) = 1.00 |
| Q9 | Does the P-n4-M1-10_r2 residual reproduce? | reported | lambda = 1 residual P_hw - P_pred(alpha 'all'), day 3 (plan calibration) beside day 1 (execution calibration, in-sample) | day 1 residual +0.022 (z +3.5); day 3 P_pred 0.073 |

## Per record x grid (alpha_all, plan calibration)

| Record | Grid | lambda | P(lambda) predicted | Status | P(status) | Reported (clip) | SE_pred | SE band (alpha_r) | 2q status |
|---|---|---|---|---|---|---|---|---|---|
| P-n2-M1-11_L1 | gentle | 1, 1.25, 1.5 | 0.498, 0.442, 0.397 | resolved | 1.00 | 0.944 [0.837, 1.000] | 0.055 | 0.043–0.066 | resolved |
| P-n2-M1-11_L1 | rule | 1, 2, 2.5 | 0.498, 0.341, 0.306 | resolved | 1.00 | 0.915 [0.853, 0.977] | 0.032 | 0.024–0.035 | resolved |
| P-n3-M1-17_L1 | odd | 1, 3, 5 | 0.195, 0.123, 0.118 | not identifiable | 1.00 | 0.195 (lambda = 1) | n/a (4834415.54) | n/a (1.05)–n/a (-) | not identifiable |
| P-n3-M1-17_L1 | gentle | 1, 1.25, 1.5 | 0.195, 0.173, 0.146 | resolved | 1.00 | 0.705 [0.400, 1.000] | n/a (0.27) | 0.072–n/a (0.35) | resolved |
| P-n3-M1-17_r1_L1 | gentle | 1, 1.25, 1.5 | 0.287, 0.240, 0.205 | resolved | 1.00 | 0.784 [0.636, 0.933] | 0.076 | 0.048–0.072 | resolved |
| P-n4-M1-10_r1_L1 | gentle | 1, 1.25, 1.5 | 0.096, 0.084, 0.076 | resolved | 1.00 | 0.290 [0.108, 0.473] | n/a (0.41) | n/a (0.22)–n/a (0.56) | resolved |

## lambda = 1 per record (drift copies pooled)

| Record | Set | P_pred (all) | P_pred (2q) | P_u | Ideal | Reference alpha_r |
|---|---|---|---|---|---|---|
| P-n2-M1-11_L1 | fitted | 0.500 | 0.474 | 0.250 | 1.000 | 3.02 (2026-10-25) |
| P-n3-M1-17_L1 | fitted | 0.195 | 0.188 | 0.125 | 0.945 | 3.10 (2026-10-25) |
| P-n3-M1-17_r1_L1 | fitted | 0.287 | 0.267 | 0.125 | 0.781 | 3.76 (2026-10-25) |
| P-n4-M1-10_r1_L1 | fitted | 0.096 | 0.100 | 0.062 | 0.473 | 2.47 (2026-10-25) |
| P-n4-M1-10_r2_L1 | fitted | 0.073 | 0.074 | 0.062 | 0.908 | 2.18 (2026-10-25) |
| P-n3-M2-17_L1 | layout_test | 0.338 | 0.327 | 0.250 | 1.000 | 3.43 (2026-10-25) |
| P-n3-M2-17_L2 | layout_test | 0.339 | 0.338 | 0.250 | 1.000 | 8.00 (2026-10-28) |
| P-n2-M1-11_L2 | out_of_sample | 0.610 | 0.621 | 0.250 | 1.000 | 3.07 (2026-10-28) |
| P-n3-M1-17_L2 | out_of_sample | 0.222 | 0.224 | 0.125 | 0.945 | 2.98 (2026-10-28) |
| P-n3-M1-17_r1_L2 | out_of_sample | 0.269 | 0.303 | 0.125 | 0.781 | 2.65 (2026-10-28) |

Q2 baselines (each model's lambda = 1 RMS on these circuits on their own day): all: fitted 0.021 (job 2026-10-25 (in-sample)), out_of_sample 0.027 (job 2026-10-28 (out of sample)); 2q: fitted 0.024 (job 2026-10-25 (in-sample)), out_of_sample 0.014 (job 2026-10-28 (out of sample))

## Outcome -> sentence

- **Q1 fitted.held**: On a third day, alpha_hat predicted every circuit it was fitted to within {threshold_abs} at lambda = 1 (RMS {rms:.3f}, largest {max_abs:.3f}).
- **Q1 fitted.not held**: On a third day, alpha_hat missed {n_out} of the {n} circuits it was fitted to by more than {threshold_abs} at lambda = 1 (RMS {rms:.3f}, largest {max_abs:.3f}).
- **Q1 out_of_sample.held**: On circuits it was not fitted to, from another day's transpilation, alpha_hat held within {threshold_abs} (RMS {rms:.3f}, largest {max_abs:.3f}).
- **Q1 out_of_sample.not held**: On circuits it was not fitted to, alpha_hat missed {n_out} of {n} by more than {threshold_abs} (RMS {rms:.3f}, largest {max_abs:.3f}).
- **Q2 all better**: Scaling every error fits the third day better than scaling the CZ errors alone (RMS {rms_all:.3f} vs {rms_2q:.3f}).
- **Q2 2q better**: Scaling the CZ errors alone fits the third day better than scaling every error (RMS {rms_2q:.3f} vs {rms_all:.3f}).
- **Q2 indistinguishable**: The third day does not separate the two models (RMS {rms_all:.3f} vs {rms_2q:.3f}, margin {margin}).
- **Q2 not decidable by design**: The two models predict these circuits too alike to be told apart (predicted RMS difference {predicted_rms_difference:.3f}).
- **Q3 confirmed**: On the same circuit, layout, shots and job, the odd grid could not resolve the extrapolation ({odd_status}) and the gentle grid did: the grid alone decided.
- **Q3 not confirmed**: On the same circuit and job the odd grid gave '{odd_status}' and the gentle grid '{gentle_status}': the grid alone did not decide as predicted.
- **Q4 Q4a.held**: Where its predicted relative error was at most {eligible_max_predicted_relative_se}, the propagated error matched the bootstrap on every resolved hardware fit (ratios {ratios}).
- **Q4 Q4a.not held**: Where its predicted relative error was at most {eligible_max_predicted_relative_se}, the propagated error missed the bootstrap on {n_out} of {n} resolved hardware fits (ratios {ratios}).
- **Q4 Q4a.not tested**: No resolved fit had a predicted relative error of at most {eligible_max_predicted_relative_se}; the formula was not tested.
- **Q4 Q4c.held**: The grid rule's predicted gain held: the gentle grid's error was {observed:.2f} times the rule grid's, against {predicted:.2f} predicted.
- **Q4 Q4c.not held**: The grid rule's predicted gain did not hold: the gentle grid's error was {observed:.2f} times the rule grid's, against {predicted:.2f} predicted.
- **Q4 Q4c.not tested**: The P-n2 grid comparison was not tested: {reason}.
- **Q5 reported**: The pre-registered status matched on {matches} of {n} fits; the predictions expected {expected:.1f}.
- **Q6 reproducible per circuit**: A circuit's alpha is a property of the circuit: on a third day it reproduced circuit by circuit (chi2/dof {chi2_red_differences:.2f} after a common shift of {shifts}), while circuits still differed from one another (chi2/dof {chi2_red_between:.2f}).
- **Q6 not reproducible**: A circuit's alpha did not reproduce on a third day (chi2/dof {chi2_red_differences:.2f} after a common shift of {shifts}): the scatter between circuits is not a fixed property of each circuit.
- **Q6 no scatter to reproduce**: On the third day the circuits no longer differed beyond shot noise (chi2/dof {chi2_red_between:.2f}), so there was no per-circuit scatter to reproduce.
- **Q6 not tested**: Not tested: {n} circuits remained after the exclusions, too few for the differences and the scatter.
- **Q7 L2=random,L1=fine**: a persistent defect of the day-2 layout that the calibration does not show
- **Q7 L2=fine,L1=fine**: a transient on 2026-10-28
- **Q7 L2=random,L1=random**: not a layout effect: both layouts were at random on the third day; open
- **Q7 L2=fine,L1=random**: the defect moved to the day-1 layout; open
- **Q7 L2=degraded,L1=fine**: a persistent, partial defect of the day-2 layout: still below its calibration, no longer at random
- **Q7 L2=degraded,L1=degraded**: both layouts below the prediction: this instance degraded on the third day, on either layout; open
- **Q7 L2=degraded,L1=random**: the day-1 layout at random and the day-2 layout degraded; open
- **Q7 L2=random,L1=degraded**: the day-2 layout still at random and the day-1 layout degraded; mostly a layout defect, open
- **Q7 L2=fine,L1=degraded**: the day-2 layout recovered and the day-1 layout degraded: a transient that moved; open
- **Q8 reaches**: The gentle grid resolves the extrapolation at n = 4, r = 1: {value:.3f} [{lo:.3f}, {hi:.3f}] against an ideal of {p_ideal:.3f}.
- **Q8 does not reach**: At n = 4, r = 1 the gentle grid did not resolve the extrapolation ({status}); the unmitigated value is {p1:.3f}.
- **Q8 not tested**: Not tested: the folds of {record} were trimmed.
- **Q9 reported**: The day-1 residual of P-n4-M1-10_r2 at lambda = 1 was {day1:+.3f} (z {day1_z:+.1f}); on the third day it was {day3:+.3f} (z {day3_z:+.1f}).
