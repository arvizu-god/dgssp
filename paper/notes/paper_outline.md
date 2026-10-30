# Paper A: contributions, novelty, and section-by-section outline (2026-10-28)

Built from `paperA_plan.md`, `resource_model.md`, `instance_family.md`,
`sim_results.md`, `hw_results.md`, `hw_review_wp7.md`, the committed data, and a
literature check done today (sources at the end). Every number is one already
checked against `paper/data/`.

---

## Part 1. Contributions

### 1.1 What we planned (WP0, frozen)

- **C1** Closed-form + measured resource scaling of the Draper-oracle Grover-SSP
  circuit on heavy-hex devices, including the offset + guard-bit signed encoding.
- **C2** Hardware execution with ISA-level-folded ZNE (scalar and pointwise),
  compared against Mitiq and against a calibration-derived noise model, on a
  frozen random instance family, with bootstrap CIs.
- **C3** Where the algorithm fails on hardware (iteration sweep, survival
  estimate) and the limits of ZNE on sampling distributions (the M = 0 pathology).

### 1.2 What we actually have

| Planned | Delivered? | What the data actually say | Change in emphasis |
|---|---|---|---|
| C1 resource model | **Yes, fully** | Closed forms exact on 144/144 circuits. Heavy-hex routing costs 1.25–2.08× (median 1.64). Signed items cost **zero** two-qubit gates (only 2b phase gates); the guard bit is the real cost (29–168 2Q). Asymptotically the ancilla-free diffuser Θ(n²) dominates the Θ(n log n) adder, though not inside the measured grid. | None. Add the hardware-limit band on Fig 4: only (n, b) = (2,3), (2,4), (3,3) fit at one iteration. |
| C2 hardware + ZNE + Mitiq + model | **Yes, but different from expected** | (a) The calibration model is optimistic; one factor α̂ = 3.18 on every error fits folded and unfolded circuits, and predicted a second day to within 0.05 on 3 of 4 records. **A pre-registered third day did not hold** (fitted circuits RMS 0.046; out of sample RMS 0.212), and the per-circuit α_r did not reproduce (χ²/dof 8.04). (b) Whether ZNE works is set by the fold grid: odd grid resolves n = 2 only; gentle grid resolves all M = 1 records up to n = 3. (c) Mitiq agrees wherever our fit is resolved. | The noise-model comparison became a **result** (α̂ + out-of-sample test), not a check. "Compared against Mitiq" became an implementation cross-check plus a demonstration of the gate. "Frozen random family" is true in simulation (463 instances); on hardware it is the **shallowest instance per cell**, i.e. best cases. Say so. |
| C3 failure map + ZNE limits | **Yes** | Signal kept: 0.47 (n = 2), 0.10–0.29 (n = 3), ≤ 0.021 (n = 4, detectable, useless). Fewer iterations than optimal win. The a-priori survival estimate put the limit ~2× too far. Pointwise ZNE: coin toss where unresolved, always harmful at M = 0. | Fewer-iterations-win is **known** (Tenev et al. 2026), so cite, don't claim. |
| (not in C1–C3) | **New** | The **reporting gate**: extrapolate only when every folded point is still ≥ 2σ above random and enough points are informative; otherwise report the λ = 1 value, flagged. Without it: 3.547 (Mitiq), 376.6 and 2275 (raw fit) on hardware. | Promote. It is the most transferable thing in the paper. |

### 1.3 Proposed contribution statement (rewrite of C1–C3)

The three contributions stay, but each now says what was found, not what was done:

1. **Resource model of the Draper-oracle Grover circuit for signed Subset Sum** (C1).
   Exact gate counts per iteration, checked against every built circuit; measured
   heavy-hex cost (routing ×1.64); signed items at zero two-qubit cost; and a map
   of which (n, b) fit under the measured hardware limit.
2. **A hardware failure map, and how far a scaled noise model carries over** (C2 +
   C3a). Where the search stops returning useful answers on ibm_fez. A scaled
   calibration model describes one day well, carries over only roughly, and misses
   failures the calibration doesn't show; we show this with a pre-registered test
   (every error × 3.18, fitted on day 1; close on day 2; not held on day 3,
   `hw_results.md` §9).
3. **Zero-noise extrapolation for a sampling algorithm: when it can be trusted**
   (C2 + C3b). The scalar estimator plus a resolution gate. The fold grid decides
   whether extrapolation is possible at all. Extrapolating the whole distribution
   point by point fails, worst on infeasible instances.

### 1.4 Is it novel enough?

Short answer: **yes for EPJ Quantum Technology or Quantum Information Processing
as a characterization + methodology paper; no for a high-impact journal.** No
single piece is a big new idea. The value is the combination, done carefully, on
a construction nobody has put on hardware with data.

Item by item, against what exists (search today, not exhaustive):

| Our claim | Closest prior work | Verdict |
|---|---|---|
| Hardware data for an arithmetic-oracle Grover SSP | Benoit, Schwartz, Cytron (arXiv:2410.01775): ripple-carry adder, positive items, gate counts only; say IBM hardware results were "disappointing" but report none. No hardware data for SSP/knapsack Grover with an adder oracle found. | **Novel** ("to our knowledge"). Weak on its own: n ≤ 4. |
| Signed SSP at zero 2Q cost; exact counts; heavy-hex factor | 2410.01775 (no signs, no QFT adder); Draper / Ruiz-Perez (generic QFT arithmetic); Lancellotti et al. 2026 (quantum walk SSP, fault-tolerant resource counts) | **Novel for this construction**, incremental in kind. Supporting contribution. |
| Grover dies before r_opt; limit at a few hundred 2Q | Tenev, Nedev, Vitanov 2026 (Heron r1–r3, 3–6 qubit phase oracles: peak below r_opt, 6 qubits at random); AbuGhanem, Sci Rep 2025 (3 qubits, Eagle) | **Known qualitatively.** Our addition: the limit expressed in 2Q gates for an arithmetic oracle, and tied to a resource model. |
| Calibration model is optimistic | AbuGhanem 2025 (noisy sim 77 % vs hardware 45 %); Bravo-Montes et al. EPJ QT 2024 (choose among noise models, no scaling factor, no out-of-sample test); Díaz Agreda et al. 2026 (digital twin, no factor, no out-of-sample) | Optimism is **known**. A single factor that fits folded + unfolded circuits and is **tested on a second day, then against a pre-registered third day where it did not hold,** is not done in these papers. The novelty is the test, including its failure. **Modest novelty.** |
| ZNE of bitstring distributions; pointwise pathology | Liu & Cai, arXiv:2502.11285: general framework for mitigating sampling (bitstring probability = projector expectation), clipping negatives, simulation only, PEC-focused, no analysis of near-uniform failure | Framework **exists**; our failure mode (nonlinear per-bitstring fit near uniform, D_TV → 1 − 2⁻ⁿ) and hardware demonstration are **new but narrow**. Must cite and position. |
| Reporting gate | Miranskyy et al. 2026 (bounded ZNE models; "coverage"; no refusal mechanism); Mitiq best-practices paper (arXiv:2307.05203) | **Modest novelty.** The gate is simple; its value is shown by the hardware contrast (3.547, 376.6, 2275). |
| Fold grid decides | Giurgica-Tiron et al. 2020 (partial folding); general advice to keep scale factors small | Known in spirit; **quantified** here (463 instances + same instances on hardware), and **confirmed pre-registered on one circuit in one job** (Q3). Modest. |

**What a referee will say:** "incremental, small n". The answer is the
framing: lead with the transferable lessons (how far a calibration model is
off and how far one factor carries over between days; when ZNE output can be trusted for a
sampling algorithm; why the fold grid matters), with SSP as the testbed.

**Open gap from WP0.** `related_work.md` and `contributions.md` were never
written into the repo. "Sankar et al., npj QI 2024 (multi-solver SSP benchmark)"
in the plan could not be found today; verify it exists before citing it.

---

## Part 2. Section-by-section outline

Target: ~7 000–8 000 words of main text, 10 figures max (better 8), 5 tables,
plus a supplement. Numbers in brackets are 95 % intervals.

### Title (candidates)

1. *Draper-oracle Grover search for Subset Sum on a heavy-hex processor:
   resource scaling, how far a scaled noise model carries over, and when zero-noise
   extrapolation can be trusted*
2. *How far does an arithmetic-oracle Grover search run on IBM Heron hardware?*
3. *Error-mitigated Grover search for signed Subset Sum on superconducting
   hardware: limits and lessons*

### Abstract (~200 words)

Problem (Grover with arithmetic oracle; NISQ). What we did (resource model;
463-instance simulation; three jobs on ibm_fez, the third pre-registered). Three findings with numbers:
signal kept 47 % / 10–29 % / ≤ 2 %; α̂ = 3.18, close on a second day, not held on a pre-registered third;
ZNE resolves n = 3 only on a gentle grid, and a gate keeps unsupported values
out (3.55 for a probability). One closing line: code and data open.

### 1 Introduction (~800 words)

What we explain:
- SSP in one paragraph: NP-complete, the decision version, why it is a natural
  Grover target (checking a candidate is an addition and a comparison).
- Grover with an arithmetic oracle: the oracle must compute Σ aᵢxᵢ in
  superposition, compare it with the target, and uncompute. Draper's adder does
  the addition in the Fourier basis with controlled phases, needing no carry
  ancillas.
- The gap: such circuits are designed and counted (2410.01775; quantum-walk
  SSP), but their behaviour on hardware is not reported. Grover on hardware is
  studied only with simple phase oracles.
- Why characterization matters: how many gates a device really affords; whether
  simulators predict it; whether ZNE helps an algorithm whose output is a
  distribution, not an expectation value.
- Contributions (the three of §1.3), each with its key number.
- No "advantage" or "speedup" anywhere.
- Paper map.

### 2 Background and related work (~700 words)

What we explain:
- **Grover-type SSP oracles**: 2410.01775 (ripple-carry, positive integers, gate
  counts); Gilliam et al. (Grover adaptive search, QFT-based arithmetic for
  constrained binary optimization); quantum-walk SSP (Lancellotti et al.,
  fault-tolerant counts). One positioning table: adder, signed?, hardware data?,
  max n, mitigation.
- **Grover on hardware**: Tenev et al. 2026; AbuGhanem 2025; older IBM / trapped-
  ion demos. Common findings: success falls fast with qubits; best r < r_opt.
- **ZNE**: Temme–Bravyi–Gambetta, Li–Benjamin, Kandala et al., digital folding
  (Giurgica-Tiron et al.), Mitiq (LaRose et al.), best practices (2307.05203),
  bounded models (Miranskyy et al. 2026).
- **Mitigating samplers**: Liu & Cai 2025. Their framework treats each bitstring
  probability as an observable; our pointwise estimator is a nonlinear instance
  of it, and our scalar estimator is the one-observable alternative.
- **Noise-model fidelity**: Bravo-Montes et al. 2024; Díaz Agreda et al. 2026;
  IBM's layered-gate benchmarks (EPLG) as the likely reason isolated gate errors
  understate circuit errors.
- Close with what nobody in the table did.

### 3 The algorithm and its encoding (~700 words, Fig 1)

What we explain:
- Registers: n index qubits (one per item), b sum qubits.
- One iteration = add (QFT, controlled phases from each index qubit, QFT†),
  mark (X on zero bits of the target, multi-controlled Z on b qubits), uncompute
  (QFT, subtract, QFT†), diffuse (multi-controlled Z on n qubits). Four QFT blocks
  per iteration, and why the middle pair cannot cancel (the oracle needs the sum
  in the computational basis).
- **Signed encoding**: offset = −Σ(negative items), so every reachable sum is
  non-negative; b = ⌈log₂(range)⌉ + 1, the +1 being the guard bit that prevents
  wrap-around (aliasing) in the modular adder. The offset is an uncontrolled phase
  addition. Key fact: the range, and therefore b, does not change when an item's
  sign flips.
- Iteration count r_opt(n, M) = ⌊π / (4 arcsin √(M/2ⁿ))⌋; M is known here from a
  classical solver (state it: this is a characterization, not a solver that
  learns M).
- Fig 1: circuit schematic (exists, `fig_circuit`).

### 4 Resource model (~900 words, Table 2, Fig 4) — C1

What we explain:
- Per-iteration closed forms: #CP = 2b(b−1) + 2n₊b; #P_offset = 2b if any item
  is negative, else 0; two multi-controlled Z of widths b and n; qubits n + b, no
  declared ancillas.
- The pinned ancilla-free reference for multi-controlled gates (g(w) =
  8w² − 32w − 36 for w ≥ 8) and why a reference is needed (the transpiler borrows
  idle qubits, so cost depends on context).
- #2Q(n, b, r) = r[4b(b−1) + 4n₊b + g(b) + g(n)]. Asymptotically Θ(n²) per
  iteration from the diffuser, not from the adder; in the measured window the
  adder still dominates.
- Measured: all 144 circuits match the formulas; transpiled ×1.25–2.08 (median
  1.64) on three heavy-hex twins; ancilla borrowing 0.78×; signed −15 to 0 2Q
  (median 0); guard bit 29–168 2Q.
- The survival estimate S = (1−ε₂)^{n₂}(1−ε₁)^{n₁}, presented as the a-priori
  hardware criterion. Its outcome is reported in §7, not here.
- Fig 4 with the hardware band: which cells fit (only (2,3), (2,4), (3,3) at one
  iteration).
- One sentence on the unexploited 2π-identity gates (removed by the transpiler)
  and one pointing to approximate rotations as future work.

### 5 Methods (~1 200 words, Fig 3, Table 3)

What we explain, in order of the pipeline (Fig 3, `fig_pipeline`):
1. **Instances.** Family P (n = 2–6, positive) and S (n = 3–5, one or two
   negated items), 463 instances in 44 cells, items bounded so b ∈ {3,4,5},
   targets drawn by multiplicity class M ∈ {0, 1, 2, 3+}. Fully seeded.
   Hardware subset: shallowest instance of each (n, M) cell + one edge instance,
   two sweep instances. State that these are best cases and that several are
   degenerate ([1,1,1] → 3).
2. **Metrics.** P(solution); signal kept (P − P_u)/(P_ideal − P_u); D_TV to the
   ideal distribution; multinomial bootstrap CIs (B = 2000, seed 0).
3. **Noise scaling.** Transpile once, fold the ISA circuit (gate folding, partial
   for non-integer λ), never re-route. Grids: odd {1,3,5}, odd7 {1,3,5,7},
   planned (per instance, mostly {1,1.5,2}), gentle {1,1.25,1.5} (hardware job 2).
4. **Scalar ZNE.** Exponential with asymptote fixed at P_uniform:
   P(λ) = P_u + A e^{−cλ}. Why the asymptote is fixed (full noise gives random
   guessing; checked: n ≥ 3 λ = 5 points at P_u within |z| ≤ 2.3).
5. **The reporting gate**, in plain terms: a scale is informative if it is ≥ 2
   standard errors above random on the ideal's side; the fit is identifiable if
   at least two scales are informative and resolved if all are; we report the
   extrapolation only when it is resolved and identifiable, otherwise the λ = 1
   value, flagged. State that rule v2 (one-sided) was adopted after job 1 and
   changed none of the 75 hardware fit blocks.
6. **Pointwise ZNE.** One fit per bitstring, then clip-and-renormalize or project
   onto the simplex.
7. **Mitiq cross-check.** Same counts, same asymptote, same bootstrap; only the
   fit code differs.
8. **Noise model and α.** Aer model from the calibration in force during each job;
   α multiplies every gate, readout and reset error (T1/T2, durations fixed);
   fitted to hardware P(solution) on a grid with bootstrap.
9. **Hardware protocol** (Table 3): ibm_fez, transpile once at opt level 3 over
   32 SABRE seeds, gate twirling (32 × 128), DD off, no readout mitigation, one
   job per experiment, dry run on the day's calibration, prediction before job 2,
   drift copies at start and end. QPU: 75 s + 51 s + 7 s probe = 133 s.

### 6 Simulation results (~1 100 words, Figs S-grid, S-dtv, S-sweep; Table)

What we explain (fake_marrakesh, 463 instances; this is the method study):
- **Grid matters most.** Resolved fraction on instances with signal, odd vs
  planned: n = 2: 55 / 100 %; n = 3: 28 / 98 %; n = 4: 6 / 72 %; n = 5: 1 / 12 %;
  n = 6: 3 / 20 %, at equal shots.
- **Recovery.** Median share of the gap closed (planned): 0.81, 0.89, 0.58 at
  n = 2, 3, 4; 0 at n = 5, 6. Odd grid: 0 everywhere. Do not quote the pooled 0.25.
- **The reporting rule.** Strict: median bias −0.44, 19 of 340 fits outside
  [0, 1] (max 4.17), reported clipped to [P_u, 1] since Step 5 (0 reported
  outside). Relaxed: −0.34, 26 fits outside (one at 298). Raw: −0.17, 71
  outside, mean bias +16.7. Why we keep strict: with clipping, range no longer
  separates the rules (`hw_results.md` §7, "Open (Step 5)").
- **Model choice** (odd7): exponential 0.113 median error on 75 fits; Richardson
  0.057 but on only 13; quadratic 0.32; linear 0.55. Richardson is more selective,
  not more accurate.
- **M = 0 pathology.** 116 infeasible instances: median D_TV to uniform 0.023 →
  0.86 (clip) / 0.94 (simplex), every instance worse; simplex collapses all mass
  to one outcome. Mechanism in one paragraph: dividing shot noise by the decay of
  a signal that is not there. Link to Liu & Cai: the amplification factor
  diverges when the decay is unresolved.
- **Mitiq.** Agrees where resolved (median difference 0.016 odd, 0.061 planned);
  where declined, median error 0.69–0.86 and 40–47 values outside [0, 1].
- **Sweep.** n = 3 noisy peak moves to r = 1; ZNE restores the shape (0.736, 0.922,
  0.412 vs 0.781, 0.945, 0.330). n = 4 falls monotonically; ZNE resolves only r = 1.
- Keep one figure for the grid result and one for M = 0; the rest to supplement.

### 7 Hardware results (~1 500 words, Table 4, Table 5, Figs 5, 7, 9, 10)

Drafted in `hw_review_wp7.md` §5. Order:
1. Unmitigated performance and the sweep.
2. Device vs calibration model; α̂ fitted on job 1; close on the second day; not held
   on the pre-registered third (scorecard first, `hw_results.md` §9); failures the
   calibration doesn't show (P-n3-M2-17_L2 on two days, P-n2-M1-11_L2 on day 3), with
   the qubit 90–92 pattern as a labelled post hoc hypothesis (no further job);
   survival at α = 1 placed the limit ~2× too far.
3. ZNE: odd vs gentle grid (promote Table S2 to Table 5).
4. What the gate prevents (Mitiq 3.547; raw 376.6 / 2275).
5. Pointwise on hardware: coin toss; M = 0 collapse.

### 8 Discussion (~1 000 words)

What we explain:
- **Why it fails.** The useful budget on this device is roughly 100–230 two-qubit
  gates. One Draper-Grover iteration at n = 3 already costs 112–212 after routing.
  The arithmetic, not the qubit count, is what consumes the budget at small n
  (diffuser takes over only at n ≈ 10+). Routing ×1.64 and α ≈ 3.2 together
  mean a simulator built from calibration overstates the reachable size by
  ~2× in gate count.
- **What α is and is not.** A phenomenological factor. Likely contributors:
  crosstalk and spectator errors in parallel layers (what layered benchmarks
  capture and isolated RB misses), leakage, residual coherent error. It is not
  constant: on a pre-registered third day the healthy circuits' α_r fell by about
  0.6, and circuits failed outright that the calibration rated as ordinary.
  Treat it as a description of one day that carries over only roughly.
- **What ZNE can do for a sampler.** Extrapolate a scalar (P(solution)), not the
  distribution. Choose the grid so folded points still carry signal (pre-register
  with the survival rule). Report only resolved fits. ZNE extended the reach from
  n = 2 to n = 3 here, not to n = 4.
- **Comparison with simple-oracle Grover on Heron** (Tenev et al.): consistent;
  the arithmetic oracle hits the same 2Q budget at fewer qubits.
- **Limitations, stated plainly:** n ≤ 4 and easiest instances; one device; three
  days; M known in advance; no readout mitigation; α includes readout; the
  gentle-grid extrapolation has a lever arm of 2 and model choice is not in the
  CI (defended by the λ-independence of α).
- **Future work (one paragraph, no results):** approximate rotations (K = 3: −22 %
  2Q at ≤ 6 % ideal loss, measured offline); dynamic-circuit counting; a second
  device; why α shifts between days.

### 9 Conclusion (~250 words)

Three sentences, one per contribution, each with its number. One sentence on
what a practitioner should take away (budget in 2Q gates; scale your simulator;
gate your extrapolations). Data and code availability (Zenodo DOI).

### Appendices / supplement

- A. Instance family table (`family_table.tex`).
- B. Resource-model derivations in full (from `resource_model.md` §§1–8).
- C. Calibration snapshots; α fit table (λ subsets, leave-one-out, χ²); per-PUB
  model deviations; residual pattern by sweep variant.
- D. Hardware per-PUB table for both jobs; Mitiq per record; pointwise D_TV per
  record.
- E. Drift tests and the null-results list (without the day-1/day-2 row).
- F. Rule v1 vs v2.
- G. P-n3-M2-17 diagnostics (marginals, distribution, layouts).
- H. Reproducibility: commit hashes, job IDs, QPU seconds, package lock.

### Figures: keep / move

| Fig | Content | Where |
|---|---|---|
| 1 | Circuit | main |
| 2 | Worked hardware example | supplement |
| 3 | Pipeline | main |
| 4 | Resources + hardware band | main |
| 5 | Signal kept vs 2Q (hardware, model, α̂) | main (key figure) |
| 6 | Folding curves by status | merge with 7 or supplement |
| 7 | ZNE vs Mitiq | main |
| S-grid | Odd vs planned resolved fractions | main (§6) |
| 9 | Sweep, sim + hardware | main |
| 10 | M = 0 pathology, sim + hardware | main |
| S1 | Model comparison | supplement |

---

## Sources (checked 2026-10-28)

- Benoit, Schwartz, Cytron, *Optimization of a Quantum Subset Sum Oracle*, arXiv:2410.01775
- Liu, Cai, *Quantum Error Mitigation for Sampling Algorithms*, arXiv:2502.11285
- Tenev, Nedev, Vitanov, *Improvement of performance of Grover's algorithm on three generations of Heron family IBM QPUs…*, arXiv:2604.23228
- AbuGhanem, *Characterizing Grover search algorithm on large-scale superconducting quantum computers*, Sci Rep 15, 1281 (2025)
- Bravo-Montes et al., *A methodology to select and adjust quantum noise models through emulators*, EPJ Quantum Technol. 11, 71 (2024)
- Díaz Agreda et al., *Experimental Implementation of the Quantum Volunteer's Dilemma on NISQ Hardware…*, arXiv:2605.30676
- Miranskyy et al., *Improving Zero-Noise Extrapolation via Physically Bounded Models*, arXiv:2604.24475
- Lancellotti et al., *Solving the Subset Sum Problem via Quantum Walk Search*, IEEE Trans. Comput. 75(1), 2026
- Best practices for digital ZNE, arXiv:2307.05203; Giurgica-Tiron et al., arXiv:2005.10921
