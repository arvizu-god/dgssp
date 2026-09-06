# IBM Quantum account facts — TEMPLATE

**Status: UNFILLED.** Answer every question below from your own dashboard and
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
| Which plan is the account on (`Open`, `Pay-As-You-Go`, `Premium`)? | |
| Instance CRN(s) in use | |
| Monthly QPU-time allowance, exact figure and unit | |
| Does the allowance reset on a fixed calendar day, or a rolling window? **[verify]** | |
| Allowance remaining right now | |
| Where on the dashboard is the remaining allowance shown (page + label)? | |

**Notes on how usage is reported** (fill in what you actually observed):

- `service.usage()` — what keys did it return, and what do they mean? Its
  payload shape is set by the Platform API, not by Qiskit, so paste the literal
  dictionary here:

  ```
  (paste output)
  ```

- `job.usage()` — documented as "job usage in seconds". Confirm on a real job
  what that number counts (queue time? execution only?). **[verify]**
- `job.metrics()` — documented to carry `timestamps` and a `usage` sub-dict.
  Paste one real job's metrics here once you have run one:

  ```
  (paste output)
  ```

- Does the dashboard's reported usage agree with the sum of `job.usage()` over
  your jobs? **[verify]** — this matters because the budget plan in WP5 is built
  on whichever of the two is authoritative.

  Starting point for docs (confirm the URL still resolves and note the date):
  <https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time>

## 2. Devices exposed to this plan

Fill one row per backend that `describe_account.py` lists as non-simulator.

| Name | Qubits | Processor family + revision | Heavy-hex? | Operational | Typical pending jobs | Calibration date seen |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

- Which of these is the paper's **primary** target device, and why (width,
  median two-qubit error, queue depth)?
- Which is the **fallback** if the primary is down on run day?
- Do any of them support fractional gates / dynamic circuits in a way that
  changes the transpiled two-qubit count? **[verify]**

## 3. Execution modes

This is the one that decides how the single monthly job is shaped.

| Question | Answer |
|---|---|
| Does `Batch(backend=...)` work on this plan? **[verify]** | |
| Does `Session(backend=...)` work on this plan, or is it rejected? **[verify]** | |
| If sessions are rejected, what is the exact error message? | |
| Maximum number of PUBs (circuits) in one `SamplerV2.run(...)` call — `backend.max_circuits` | |
| Maximum shots per PUB | |
| Is there a per-job wall-clock cap, and what is it? **[verify]** | |

  Starting point for docs (confirm and date it):
  <https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>

**Conclusion to record explicitly** — one of:

- [ ] Batch works; the monthly run uses `execution_mode(backend, mode="batch")`.
- [ ] Batch is unavailable; the monthly run is **one** `SamplerV2` job with many
      PUBs via `sample_counts(backend, [c1, c2, ...])`, which the library already
      does in a single call.
- [ ] Sessions work and are worth using because: ______

Note that `dgssp.runtime.sample_counts` submits a list of circuits as one job
regardless, so the many-PUBs path needs no code change — only the wrapper in
`execution_mode` does.

## 4. Practical constraints for the monthly run

| Question | Answer |
|---|---|
| Longest observed queue wait | |
| Time of day / day of week with the shortest queue | |
| Does an unfinished job keep consuming allowance if cancelled? **[verify]** | |
| Can a job be recovered by id after the process dies (`fetch_result`)? | |
| Where does the job ledger live for hardware runs? | `runs/jobs.jsonl` (gitignored) — copy the ids into `paper/hardware/` after each run |

## 5. Budget arithmetic for this paper

Fill once sections 1–4 are answered.

- QPU seconds available per month: ______
- Estimated QPU seconds per circuit at the paper's largest `n`: ______
  (from `job.usage_estimation` on a dry run, or measured on a small job)
- Circuits per monthly job the budget allows: ______
- Therefore the frozen instance family is ______ instances × ______ noise
  scales × ______ iteration counts = ______ circuits.
- Months of hardware time the paper needs: ______

---

## Change log

| Date | What changed | Source |
|---|---|---|
| | | |
