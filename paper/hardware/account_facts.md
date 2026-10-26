# IBM Quantum account facts — TEMPLATE

**Status: filled 2026-10-25.** Answer every question below from your own dashboard and
from one run of `python paper/scripts/describe_account.py`, then change this
line to `Status: filled YYYY-MM-DD`.

Nothing in the paper's hardware plan (WP5/WP6) may be written until this file
is filled. Every question below changes what a hardware run can look like.

Run first:

```
python paper/scripts/describe_account.py --json paper/hardware/account_snapshot.json
```

That prints plan, instances, backends and usage, and saves a machine-readable
snapshot next to this file. Then answer the questions from the printout plus
your dashboard at <https://quantum.cloud.ibm.com/>.

Anything marked **[verify]** is something the code deliberately does *not*
assume. Do not fill it from memory or from a blog post — confirm it against
the live account or current docs and record the date and URL you confirmed it
from.

---

## 1. Plan and allowance

| Question | Answer |
|---|---|
| Which plan is the account on (`Open`, `Pay-As-You-Go`, `Premium`)? | `Open`|
| Instance CRN(s) in use | 'crn:v1:bluemix:public:quantum-computing:us-east:a/fe3f24af7a3d48a3a8c2579584054360:21f8c0b1-86ea-463a-9c04-710182ac28b0::' |
| Monthly QPU-time allowance, exact figure and unit | 'usage_limit_seconds': 600 |
| Does the allowance reset on a fixed calendar day, or a rolling window? **[verify]** | Rolling Window |
| Allowance remaining right now | 'usage_remaining_seconds': 600 |
| Where on the dashboard is the remaining allowance shown (page + label)? | In the 'Instances' section, it shows the 'experiments' instance and its usage time used. |

**Notes on how usage is reported** (fill in what you actually observed):

- `service.usage()` — what keys did it return, and what do they mean? Its
  payload shape is set by the Platform API, not by Qiskit, so paste the literal
  dictionary here: 

  ```
  {'instance_id': 'crn:v1:bluemix:public:quantum-computing:us-east:a/fe3f24af7a3d48a3a8c2579584054360:21f8c0b1-86ea-463a-9c04-710182ac28b0::', 'plan_id': '850b21a7-71de-4e53-9441-1abdd202f35d', 'usage_consumed_seconds': 0, 'usage_consumed_ru': 0, 'usage_period': {'start_time': '2026-09-26T15:55:14.124Z', 'end_time': '2026-10-24T15:55:14.124Z'}, 'usage_limit_seconds': 600, 'usage_limit_ru': 10, 'usage_limit_reached': False, 'usage_remaining_seconds': 600}
  ```

- `job.usage()` — documented as "job usage in seconds". Confirm on a real job
  what that number counts (queue time? execution only?). Execution Only
- `job.metrics()` — documented to carry `timestamps` and a `usage` sub-dict.
  Paste one real job's metrics here once you have run one:

  ```
{'caller': 'qiskit_ibm_runtime~sampler.py', 'qiskit_version': 'qiskit_ibm_runtime-0.46.1,qiskit-2.3.1*,qiskit_aer-0.17.2*', 'timestamps': {'created': '2026-04-10T07:57:27.279994Z', 'finished': '2026-04-10T07:57:34.822349Z', 'running': '2026-04-10T07:57:29.099921Z'}, 'usage': {'qpu_charge_time_seconds': 3, 'value': 0.050001, 'unit': 'resource_units', 'details': [{'value': 0.050001, 'unit': 'resource_units', 'metric': {'id': 'caffe3ca-729c-5750-8777-df63bbf8bfd9', 'value': 3}}]}, 'circuits_execution_time_ns': 0}
  ```

- Does the dashboard's reported usage agree with the sum of `job.usage()` over
  your jobs? **[verify]** — this matters because the budget plan in WP5 is built
  on whichever of the two is authoritative. YES.

  Starting point for docs (confirm the URL still resolves and note the date):
  <https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time>

## 2. Devices exposed to this plan

Fill one row per backend that `describe_account.py` lists as non-simulator.

| Name | Qubits | Processor family + revision | Heavy-hex? | Operational | Typical pending jobs | Calibration date seen |
|---|---|---|---|---|---|---|
| ibm_fez | 156 | Heron r2 | Yes | True | 2 | 2026-10-24 |
| ibm_kingston | 156 | Heron r2 | Yes | True | 1 | 2026-10-24 |
| ibm_marrakesh | 156 | Heron r2 | Yes | True | 612 | 2026-10-24 |

- Which of these is the paper's **primary** target device, and why (width,
  median two-qubit error, queue depth)? ibm_fez
- Which is the **fallback** if the primary is down on run day? ibm_kingston
- Do any of them support fractional gates / dynamic circuits in a way that
  changes the transpiled two-qubit count? All of them suppor dynamic circuits and fractional gates.

## 3. Execution modes

This is the one that decides how the single monthly job is shaped.

| Question | Answer |
|---|---|
| Does `Batch(backend=...)` work on this plan? **[verify]** | Yes |
| Does `Session(backend=...)` work on this plan, or is it rejected? **[verify]** | No |
| If sessions are rejected, what is the exact error message? | Error code 1211: Program not found. |
| Maximum number of PUBs (circuits) in one `SamplerV2.run(...)` call — `backend.max_circuits` | 10Million |
| Maximum shots per PUB | NA |
| Is there a per-job wall-clock cap, and what is it? **[verify]** | NA |

  Starting point for docs (confirm and date it):
  <https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>

**Conclusion to record explicitly** — one of:

- [X] Batch works; the monthly run uses `execution_mode(backend, mode="batch")`.
- [ ] Batch is unavailable; the monthly run is **one** `SamplerV2` job with many
      PUBs via `sample_counts(backend, [c1, c2, ...])`, which the library already
      does in a single call.
- [ ] Sessions work and are worth using because: ______

**Used on 2026-10-26:** job mode (`--mode job`). Batch works on this plan,
but it only groups several jobs, and the hardware run is a single job of 47
PUBs, so it would add nothing.

Note that `dgssp.runtime.sample_counts` submits a list of circuits as one job
regardless, so the many-PUBs path needs no code change — only the wrapper in
`execution_mode` does.

## 4. Practical constraints for the monthly run

| Question | Answer |
|---|---|
| Longest observed queue wait | ibm_marrakesh |
| Time of day / day of week with the shortest queue | Weekends |
| Does an unfinished job keep consuming allowance if cancelled? **[verify]** | Will not continue to consume your allowance once the cancellation processing is complete |
| Can a job be recovered by id after the process dies (`fetch_result`)? | Yes |
| Where does the job ledger live for hardware runs? | `runs/jobs.jsonl` (gitignored) — copy the ids into `paper/hardware/` after each run |

## 5. Budget arithmetic for this paper

Fill once sections 1–4 are answered.

- QPU seconds available per month: 600 per 28-day rolling window (not a
  calendar month; https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time).
  The window in use runs 2026-10-24 → about 2026-11-21.
- Estimated QPU seconds per circuit at the paper's largest `n`: measured, not
  estimated. The main job (`dbfeqr7gdj9s73bbgh6g`, 47 PUBs × 4096 shots,
  twirled 32 × 128, DD off) was charged **75 s**, i.e. **≈ 1.6 s per circuit**
  on average. The probe (`dbf9tu7gdj9s73fn4u40`: the shortest and the longest
  circuit, 4096 shots each) was charged 7 s. Its execution span ran 1.58× the
  schedule model (scheduled duration + 250 µs `rep_delay` per shot), which puts
  the longest circuit (n = 4, λ = 5, 6400 cz) at about 4–5 s.
- Circuits per monthly job the budget allows: at the 50 % rule (300 s of a
  full window) and this mix, about **180 circuits** (≈ 2.5× this job). The
  planning estimate (`--per-shot-time 0.000622611 --overhead 0.3824`, ×2
  safety) gave 275.7 s against 75 s actual, so it over-estimates by about
  3.7×. IBM's rule of thumb (2 s + 0.35 ms per shot) gave 69 s and was the
  closest.
- Therefore the frozen hardware job is 10 instances × 3 noise scales
  (λ = 1, 3, 5) + 5 iteration-sweep variants × 3 scales + 2 drift repeats =
  **47 circuits** (192 512 shots).
- Months of hardware time the paper needs: one job, run 2026-10-26 (75 s,
  12.5 % of the window), plus the 7 s probe. Left in this window after both:
  516 s (8 min 36 s), enough for one optional follow-up job.

---

## Change log

| Date | What changed | Source |
|---|---|---|
| 2026-10-25 | Sections 1–4 filled | `describe_account.py`, dashboard |
| 2026-10-26 | §3 records job mode; §5 filled with the measured cost of the probe and the main job | `paper/hardware/usage.md`, `probe_2026-10-25.json` |
