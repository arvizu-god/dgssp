# Module reference — overview

`dgssp` is layered: each module depends only on the ones above it, so the
dependency graph is a DAG and any layer can be tested in isolation.

| Layer | Module | Depends on | One-line role |
|---|---|---|---|
| Data | [`instance`](instance.md) | — | Problem data structures and random generation. |
| Data | [`encoding`](encoding.md) | — | Offset + guard-bit sum-register arithmetic. |
| Data | [`decoding`](decoding.md) | `instance` | Bitstring ↔ subset translation. |
| Data | [`metrics`](metrics.md) | — | P(solution), distances, survival, bootstrap intervals. |
| Data | [`records`](records.md) | — | The frozen results schema. |
| Solvers | [`solvers.base`](solvers.base.md) | `instance` | Abstract solver interfaces. |
| Solvers | [`solvers.draper_grover`](solvers.draper_grover.md) | `instance`, `encoding` | The D-G circuit builder. |
| Solvers | [`solvers.classical`](solvers.classical.md) | `instance` | Exhaustive DP baseline / ground truth. |
| Infra | [`runtime`](runtime.md) | — | IBM service, execution modes, batched sampling, job log. |
| Infra | [`backends`](backends.md) | `runtime`, `decoding`, `metrics`, `solvers` | Backend discovery and calibration scoring. |
| Infra | [`transpilation`](transpilation.md) | — | SABRE layout search minimising 2q error. |
| Mitigation | [`mitigation.zne`](mitigation.zne.md) | `runtime`, `metrics`, `transpilation`, `solvers` | Scalar and pointwise ZNE, partial folding, one-job batches. |
| Mitigation | [`mitigation.mitiq_baseline`](mitigation.mitiq_baseline.md) | `runtime`, `metrics` | Mitiq folding + offline extrapolation of `P(solution)`. |
| Orchestration | [`execution`](execution.md) | all of the above | Three single-circuit executors. |
| Orchestration | [`experiments`](experiments.md) | all of the above | Multi-instance batch runner. |
| Orchestration | [`api`](api.md) | `execution` | Single-instance convenience entry point. |
| Orchestration | [`cli`](cli.md) | `experiments`, `runtime` | `dgssp-run` command line. |

## Conventions used everywhere

**Bit ordering.** Item index 0 is the least significant bit, which is the
*rightmost* character of a Qiskit bitstring. Only [`decoding`](decoding.md)
performs this translation.

**Sum encoding.** A subset sum `s` is stored in the sum register as
`s + offset` where `offset = -sum(negative items)`. The register carries one
guard bit beyond what the value range needs, so `modulus >= 2 * range_len`.
Only [`encoding`](encoding.md) decides this.

**Execution.** Every sampler call in the library goes through
`runtime.sample_counts`. Passing it a list of circuits submits one job.

**Transpile then fold.** Noise scaling operates on ISA-level circuits, at
optimization level 0, with the layout searched exactly once. Scales are real
numbers ≥ 1 and fits use the realised (two-qubit-count) scale.

**Metrics.** Every distribution metric and interval comes from
[`metrics`](metrics.md); every archived result follows [`records`](records.md).
