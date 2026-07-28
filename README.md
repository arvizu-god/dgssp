# dgssp

**dgssp** implements a Draper-adder + Grover quantum algorithm for the Subset Sum
Problem (SSP), together with a classical dynamic-programming baseline, IBM Quantum
execution helpers, and zero-noise-extrapolation error mitigation.

- clear **instance** data structures,
- pluggable **solvers** (Draper--Grover, classical DP),
- backend-agnostic **execution** (ideal simulator, noisy simulators, real hardware),
- **error mitigation** via per-bitstring ZNE, with a Mitiq baseline for comparison,
- a **multi-instance batch runner** that submits a whole sweep as one job.

> Status: 0.1.0. APIs may change.

---

## Installation

```bash
# from GitHub
pip install "git+https://github.com/arvizu-god/dgssp.git"

# with the optional extras
pip install "dgssp[mitiq,viz,data] @ git+https://github.com/arvizu-god/dgssp.git"

# editable development install
git clone https://github.com/arvizu-god/dgssp.git
cd dgssp
pip install -e ".[dev,mitiq,viz,data]"
```

Verify:

```bash
python -c "import dgssp; print(dgssp.__version__)"   # -> 0.1.0
```

Extras:

| Extra   | Pulls in            | Needed for                                |
|---------|---------------------|-------------------------------------------|
| `mitiq` | `mitiq`             | the Mitiq ZNE comparison baseline         |
| `data`  | `pandas`            | `BatchResult.to_dataframe()`              |
| `viz`   | `matplotlib`        | plotting in the example notebook          |
| `dev`   | pytest, ruff, mypy  | running the test suite                    |

---

## Quickstart

```python
from dgssp import SubsetSumInstance, BatchConfig, run_batch

instance = SubsetSumInstance(items=[3, 5, -2, 7], target=8, name="demo")

batch = run_batch([instance], BatchConfig(shots=8192, seed_simulator=1234))
result = batch.results[0]

print(result.solution_bitstrings)     # correct outcomes, from classical DP
print(result.solution_probability)    # measured probability of hitting one
print(batch.summary())
```

Or from the command line:

```bash
dgssp-run solve --items 3,5,-2,7 --target 8
dgssp-run batch --n-instances 10 --n-items 4 --json runs/sweep.json
```

---

## Sweeping many instances

`run_batch` computes the classical ground truth for every instance, uses it to pick
each instance's optimal Grover iteration count, and submits **all** the circuits as a
single job:

```python
from dgssp import BatchConfig, run_random_batch

batch = run_random_batch(
    20, 4, max_value=15, seed=0,
    config=BatchConfig(shots=8192, seed_simulator=1234),
)

df = batch.to_dataframe()                 # needs dgssp[data]
batch.to_json("runs/sweep.json")
print(batch.summary())
```

---

## Real hardware

Credentials are saved only when you ask, and importing `dgssp` never touches the
network:

```python
from dgssp import save_account, get_service, BatchConfig, BackendSelectionConfig, run_random_batch

save_account("<API_TOKEN>", "<INSTANCE_CRN>")     # one time, opt-in
service = get_service()

batch = run_random_batch(
    5, 3,
    config=BatchConfig(
        executor="noisy",
        service=service,
        backend_config=BackendSelectionConfig(min_qubits=8, real=True),
        shots=4096,
    ),
)
```

Every submission is appended to `runs/jobs.jsonl`. If a run is interrupted while
queued, recover it rather than re-running it:

```python
from dgssp import read_job_log, fetch_result

for record in read_job_log():
    print(record["job_id"], record["stage"], record["shots"])

result = fetch_result(service, "<job_id>")
```

---

## Error mitigation

Two independent approaches, meant to be compared:

```python
from dgssp import ZNESamplingConfig, BatchConfig, run_batch

cfg = BatchConfig(
    executor="optimized",
    backend=my_noisy_backend,
    zne_config=ZNESamplingConfig(scales=[1, 3, 5], method="linear", shots_per_scale=8192),
    run_mitiq_baseline=True,          # needs dgssp[mitiq]
    shots=8192,
)
result = run_batch([instance], cfg).results[0]

result.solution_probability             # unmitigated
result.mitigated_solution_probability   # dgssp per-bitstring ZNE
result.mitiq_solution_probability       # Mitiq ZNE on P(solution)
```

The pipeline transpiles **once**, folds the resulting ISA-level circuit at each noise
scale, and submits every scale in one job. Folding a logical circuit and transpiling
afterwards would let the optimizer cancel the folds; re-searching the layout per scale
would put the fitted points on different noise curves.

---

## Negative items

Signed instances are supported directly. Sums are shifted by `offset = -sum(negative
items)` so every encoded value is non-negative, and the sum register carries one guard
bit so `modulus >= 2 * range_len` and the encoding can never wrap:

```python
from dgssp import build_encoding

enc = build_encoding([3, -2, 4], target=1)
enc.offset, enc.n_sum, enc.encoded_target
```

The offset is added as an *uncontrolled* constant and removed by the subtractor, so the
sum register still returns to `|0...0>` after each Grover step.

---

## Development

```bash
pip install -e ".[dev,data]"
ruff check src tests
mypy src/dgssp
pytest
```

CI runs on Python 3.10--3.12. Hardware tests are skipped there; only ideal and
fake-backend paths execute.

---

## Documentation

Per-module reference lives in [`docs/modules/`](docs/modules/). Start with
[`docs/modules/overview.md`](docs/modules/overview.md).

---

## License

MIT. See [LICENSE](LICENSE).
