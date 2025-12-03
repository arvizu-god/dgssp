# dgssp

**dgssp** is a Python library implementing a Draper+Grover-based quantum algorithm for the Subset Sum Problem (SSP), together with a QPE-based variant and classical dynamic-programming baselines.

The goal is to provide a clean, research-grade implementation of the algorithm described in your MSc thesis, following Qiskit-style patterns:

- clear **problem / instance** data structures,
- pluggable **solvers** (DG, QPE, classical DP),
- backend-agnostic **execution** (simulators, fake backends, real hardware),
- optional **quantum error mitigation** (e.g. Zero-Noise Extrapolation).

> ⚠️ Status: early development (0.1.0). APIs may change.

---

## Installation

Once published to PyPI (or locally via `pip -e .`):

```bash
pip install dgssp
