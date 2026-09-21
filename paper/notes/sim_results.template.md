# Simulation results (WP5)

Every number the paper's text quotes from the simulation campaign, in one
place. **This file is generated**: `paper/scripts/aggregate.py` reads
`paper/notes/sim_results.template.md`, substitutes every `{{path.to.value}}`
placeholder against `paper/data/sim_summary.json`, and writes the result to
`paper/notes/sim_results.md`. A placeholder that resolves to `??` marks a
stage that has not been run yet, not a missing result. Edit the *template*;
the filled copy is overwritten on every aggregation.

Campaign: noise `{{noise}}`, grids `{{grids}}`, {{n_instances}} instances.
Records found per stage: ideal {{stages_present.ideal}}, noisy
{{stages_present.noisy}}, ZNE {{stages_present.zne}}, Mitiq
{{stages_present.mitiq}}.

Tables behind these numbers: `paper/data/sim_per_instance.csv`,
`sim_cells.csv`, `sim_pathology.csv`, `sim_models.csv`, and
`paper/tex/sim_table_<grid>.tex`. Figures: `fig_sim_psol`, `fig_sim_dtv`,
`fig_sim_grid`, `fig_sim_sweep`, `fig_sim_models`.

---

## 1. What the scalar estimator recovers

The signal-recovery fraction is
`(P_reported - P(lambda=1)) / (P_ideal - P(lambda=1))` per feasible instance,
then the median: 1.0 means the extrapolation recovered the whole gap between
the unmitigated observation and the noiseless value, 0.0 means it recovered
none of it. It is a *median*, because an unidentified exponential has no finite
mean (see §3).

| grid | instances | median recovery | N behind it |
|---|---|---|---|
| planned | {{per_grid.planned.n_instances}} | {{per_grid.planned.signal_recovery_median}} | {{per_grid.planned.signal_recovery_n}} |
| odd | {{per_grid.odd.n_instances}} | {{per_grid.odd.signal_recovery_median}} | {{per_grid.odd.signal_recovery_n}} |
| odd7 | {{per_grid.odd7.n_instances}} | {{per_grid.odd7.signal_recovery_median}} | {{per_grid.odd7.signal_recovery_n}} |

Per-cell means (or medians) of `P(solution)` for every estimator: the LaTeX
table `paper/tex/sim_table_<grid>.tex`, and `sim_cells.csv` for the full set
including `zne_scalar_value` (the raw extrapolation) beside
`zne_scalar_reported` (what the reporting rule quotes). **The two columns
differ exactly where a fit is unresolved or unidentified** — which is the WP4
result, so the paper quotes both.

TEXT TO WRITE: the range of recovery across `n`, and the `n` at which it
collapses.

## 2. Resolved and identifiable

| grid | resolved | identifiable | reported from extrapolation |
|---|---|---|---|
| planned | {{per_grid.planned.fraction_resolved}} | {{per_grid.planned.fraction_identifiable}} | {{per_grid.planned.fraction_reported_from_extrapolation}} |
| odd | {{per_grid.odd.fraction_resolved}} | {{per_grid.odd.fraction_identifiable}} | {{per_grid.odd.fraction_reported_from_extrapolation}} |
| odd7 | {{per_grid.odd7.fraction_resolved}} | {{per_grid.odd7.fraction_identifiable}} | {{per_grid.odd7.fraction_reported_from_extrapolation}} |

Against `n`, odd versus planned (this is the partial-folding result, and
`fig_sim_grid`):

| n | odd: resolved / identifiable | planned: resolved / identifiable |
|---|---|---|
| 2 | {{grid_comparison.2.odd.fraction_resolved}} / {{grid_comparison.2.odd.fraction_identifiable}} | {{grid_comparison.2.planned.fraction_resolved}} / {{grid_comparison.2.planned.fraction_identifiable}} |
| 3 | {{grid_comparison.3.odd.fraction_resolved}} / {{grid_comparison.3.odd.fraction_identifiable}} | {{grid_comparison.3.planned.fraction_resolved}} / {{grid_comparison.3.planned.fraction_identifiable}} |
| 4 | {{grid_comparison.4.odd.fraction_resolved}} / {{grid_comparison.4.odd.fraction_identifiable}} | {{grid_comparison.4.planned.fraction_resolved}} / {{grid_comparison.4.planned.fraction_identifiable}} |
| 5 | {{grid_comparison.5.odd.fraction_resolved}} / {{grid_comparison.5.odd.fraction_identifiable}} | {{grid_comparison.5.planned.fraction_resolved}} / {{grid_comparison.5.planned.fraction_identifiable}} |
| 6 | {{grid_comparison.6.odd.fraction_resolved}} / {{grid_comparison.6.odd.fraction_identifiable}} | {{grid_comparison.6.planned.fraction_resolved}} / {{grid_comparison.6.planned.fraction_identifiable}} |

TEXT TO WRITE: the `n` at which the odd grid stops identifying the exponential,
and that the planned grid still does, at the same shot count. This is the
claim partial folding exists to support.

## 3. The reporting rule

`reported_value` is the extrapolation when the fit is **resolved and
identified**, and the `lambda = 1` observation otherwise. The open WP4 question
— whether to relax that to "identified, with resolvability as a quality flag" —
needs no re-run: `sim_per_instance.csv` carries `resolved`, `identifiable`,
`n_informative_scales`, `zne_scalar_value` and `zne_scalar_reported` per
instance, so both rules can be evaluated from the archived table.

TEXT TO WRITE: the decision, with the bias of each rule from `sim_cells.csv`
(`bias_zne_scalar_value` against `bias_zne_scalar_reported`).

## 4. Distances, and the infeasible-instance pathology

Distance to the exact reference, per cell: `sim_cells.csv`, series `tv_noisy`,
`tv_pointwise_clip`, `tv_pointwise_simplex`; figure `fig_sim_dtv`.

The `M = 0` instances are the clean test: their exact index distribution is
*uniform*, so any distance the mitigation adds is pathology, not residual
error.

| grid | M=0 instances | median D_TV at lambda=1 | after clip | after simplex | clip worse | simplex worse |
|---|---|---|---|---|---|---|
| planned | {{per_grid.planned.pathology.n_m0}} | {{per_grid.planned.pathology.median_tv_lambda1}} | {{per_grid.planned.pathology.median_tv_clip}} | {{per_grid.planned.pathology.median_tv_simplex}} | {{per_grid.planned.pathology.fraction_clip_worse}} | {{per_grid.planned.pathology.fraction_simplex_worse}} |
| odd | {{per_grid.odd.pathology.n_m0}} | {{per_grid.odd.pathology.median_tv_lambda1}} | {{per_grid.odd.pathology.median_tv_clip}} | {{per_grid.odd.pathology.median_tv_simplex}} | {{per_grid.odd.pathology.fraction_clip_worse}} | {{per_grid.odd.pathology.fraction_simplex_worse}} |

TEXT TO WRITE: the factor by which pointwise ZNE inflates `D_TV` on an M=0
instance, that the simplex projection does not rescue it, and the mechanism
(noise amplification by extrapolation weights that grow as the scales close up).

## 5. Best extrapolation model

On the `odd7` grid (four points, Family P, `n <= 4`), median absolute error to
the same instance's ideal `P(solution)`, **identified fits only**, with the
exclusions counted: `sim_models.csv`, figure `fig_sim_models`.

TEXT TO WRITE: which model wins, by how much, and how many fits each one had
to exclude — a model that is accurate only where it is identified is not more
accurate, it is more selective.

## 6. Iteration sweep

`fig_sim_sweep`, from `paper/data/sim_sweep_<noise>/`: `P(solution)` against
`r` for both sweep instances, ideal / noisy / ZNE-scalar against the analytic
`sin^2((2r+1) theta)`.

TEXT TO WRITE: where the noisy curve stops tracking the analytic one, and
whether ZNE moves the peak back.

---

Generated {{provenance.timestamp_utc}} from commit
{{provenance.git.short_commit}}.
