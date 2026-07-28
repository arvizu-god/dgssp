# `dgssp.runtime`

## What the module does

Everything that talks to IBM Quantum: service creation, execution modes,
sampling and job bookkeeping. It is the **only** place in the library that
constructs a `QiskitRuntimeService` or a `SamplerV2`.

Centralising this has three consequences that matter for real-hardware work:

- **No import-time side effects.** Importing `dgssp` never touches the network and never rewrites saved credentials. Saving an account is an explicit call to `save_account`.
- **One batching primitive.** `sample_counts` accepts a *list* of circuits and submits them as a single job, so a sweep costs one queue wait instead of one per circuit.
- **Recoverable runs.** Every submission is appended to a JSON-lines ledger with its job id, so a run interrupted mid-queue can be recovered rather than re-executed.

The legacy `ibm_quantum` channel has been retired by IBM; this module uses
`ibm_quantum_platform` with a token and an instance CRN.

## Contents

| Name | Kind |
|---|---|
| `DEFAULT_RUN_LOG` | constant |
| `get_service` | function |
| `save_account` | function |
| `is_simulator` | function |
| `backend_name` | function |
| `execution_mode` | context manager |
| `sample_counts` | function |
| `log_job` | function |
| `fetch_result` | function |
| `read_job_log` | function |

---

### `DEFAULT_RUN_LOG` — constant

`"runs/jobs.jsonl"`, the default path of the job ledger, relative to the
working directory.

---

### `get_service(*, token=None, instance=None, name=None) -> QiskitRuntimeService`

Builds a service **without side effects** — in particular it never calls
`save_account`, which the previous implementation did implicitly.

- **Inputs:** `token: str | None` (an API token; if given, a service is built from it and nothing is written to disk); `instance: str | None` (instance CRN, used with `token`); `name: str | None` (name of a saved account to load when no token is passed).
- **Output:** a live `QiskitRuntimeService`.

---

### `save_account(token, instance=None, *, name=None, set_as_default=True, overwrite=True) -> None`

The opt-in counterpart. Nothing in the library calls it implicitly.

- **Inputs:** `token: str`; `instance: str | None` (CRN); `name: str | None`; `set_as_default: bool`; `overwrite: bool`.
- **Output:** `None`. Credentials are persisted to the local qiskit configuration.

---

### `is_simulator(backend) -> bool`

Best-effort simulator detection, preferring BackendV2 signals (class name,
backend name) and only falling back to the deprecated
`configuration().simulator` flag if nothing else is available.

- **Input:** `backend: Any`.
- **Output:** `True` for Aer, fake and simulated backends.

---

### `backend_name(backend) -> str`

Reads a backend's name across the V1 (method) / V2 (attribute) split.

- **Input:** `backend: Any`.
- **Output:** the name, or `str(backend)` if it exposes none.

---

### `execution_mode(backend, *, mode="auto")` — context manager

Yields an object that `SamplerV2` accepts as its `mode`. Simulators are yielded
as-is, since sessions are pointless locally. Real hardware is wrapped in a
`Batch` by default, which is the right container for a set of *independent*
circuits: they share one queue reservation without the serialisation a
`Session` imposes.

- **Inputs:** `backend: Any`; `mode: "auto" | "backend" | "session" | "batch"`.
- **Yields:** a backend, `Batch` or `Session`.

---

### `sample_counts(mode_or_backend, circuits, *, shots=10000, seed_simulator=None, run_log=DEFAULT_RUN_LOG, meta=None) -> list[dict[str, int]]`

The library's single execution primitive and its batching entry point. Every
sampler call in `dgssp` goes through here.

- **Inputs:** `mode_or_backend` (a backend, `Session` or `Batch`); `circuits` (one already-transpiled circuit, or a sequence of them — differing widths are fine, each becomes its own pub); `shots: int`; `seed_simulator: int | None` (forwarded to simulators, ignored by hardware); `run_log: str | None` (ledger path, or `None` to skip logging); `meta: dict | None` (extra fields for the ledger entry).
- **Output:** a list of counts dictionaries, one per input circuit, **in input order** — including when a single circuit is passed, so the return type never changes with arity.

Passing a list submits **one job**. This is what makes the multi-instance
sweeps in `dgssp.experiments` affordable on hardware.

---

### `log_job(job, meta, *, run_log=DEFAULT_RUN_LOG) -> str | None`

Records a submission in the ledger and the Python logger (`dgssp.runtime`).

- **Inputs:** `job` (the object returned by `sampler.run`); `meta: dict` (JSON-serialisable metadata); `run_log: str | None` (parent directories are created as needed).
- **Output:** the job id, or `None` if the object exposed none.

---

### `fetch_result(service, job_id) -> PrimitiveResult`

Re-fetches a completed job's result. Combined with the ledger, this makes a run
that outlives its launching process recoverable.

- **Inputs:** `service` (a `QiskitRuntimeService`); `job_id: str`.
- **Output:** the job's result object.

---

### `read_job_log(run_log=DEFAULT_RUN_LOG) -> list[dict]`

- **Input:** `run_log: str`, path of the JSON-lines ledger.
- **Output:** one record per submitted job, oldest first. An empty list if the file does not exist.
