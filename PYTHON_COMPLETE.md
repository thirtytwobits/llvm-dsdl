# Python Completion Plan (Parity with Other Backends)

## 1. Goal

Bring `dsdlc python` to the same practical support level as the mature backends in this repository (especially TypeScript and Rust) across:

- generation coverage
- runtime behavior coverage
- differential parity validation
- optimization/specialization validation
- packaging and demo readiness

## 2. Current Baseline (Already Done)

Python currently has:

- first-class CLI command (`dsdlc python`)
- deterministic package/module generation for fixture and full `uavcan` trees
- generated dataclasses with `serialize()` / `deserialize()`
- generated pure runtime (`_dsdl_runtime.py`)
- backend selector (`_runtime_loader.py`) with `auto|pure|accel`
- optional accelerator module (`_dsdl_runtime_accel`, CPython C API)
- integration lanes for:
  - generation hardening
  - runtime smoke (including optimized-lowered-serdes lane)
  - backend selection
  - pure-vs-accel parity
  - full-tree generation and determinism
  - benchmark harness

## 3. Remaining Gaps to Reach Parity

Compared to the most complete targets in this project, Python still lacks:

1. Broad **C<->Python differential parity suites** across fixture families.
2. Explicit **full-tree runtime correctness lanes** (not just generation/determinism/bench).
3. A **runtime specialization model** equivalent to TS/Rust (`portable|fast`) with semantic-diff and parity gates.
4. Strong **packaging ergonomics** (installable local package flow + accelerator staging path).
5. End-to-end **demo workflows** that exercise Python-generated types in live scenarios (matching Go/Rust/native demo depth).
6. Clear **performance gating policy** for accelerator regressions.

## 4. Target Definition: "Python Complete"

Python is complete when all are true:

1. `dsdlc python` is exercised by the same validation classes used by mature backends.
2. Differential parity vs C exists for representative fixture families and signed-narrow paths.
3. A specialization story is validated (profile or backend mode), with semantic-diff and parity guarantees.
4. Full `uavcan` runtime-execution correctness is covered in CI.
5. Packaging/install workflow is documented and reproducible for pure and optional accel paths.
6. Demo docs include Python as a first-class demo node/type-consumer workflow.

## 5. Workstreams

### Workstream A: C<->Python Differential Parity

Deliver:

- Add integration scripts mirroring TS-style parity families for Python:
  - runtime parity
  - service parity
  - variable/fixed array parity
  - bigint/int edge cases
  - float and saturation/truncation edge cases
  - utf8 and delimiter handling
  - union/composite variants
  - truncated decode/error behavior
  - padding/alignment behavior
- Add signed-narrow C<->Python parity lanes.
- Add optimized-lowered-serdes variants for parity lanes.

Exit criteria:

- Differential parity suite passes in RelWithDebInfo full matrix lanes.
- No nondeterministic or flaky parity failures across repeated runs.

### Workstream B: Runtime-Execution Coverage for Full `uavcan`

Deliver:

- Add full-tree runtime test that imports selected generated `uavcan` types and executes round-trip SerDes for:
  - message types
  - service request/response types
  - delimited/composite/union-heavy types
- Add negative-path runtime checks:
  - truncated buffers
  - invalid union tags
  - array length violations

Exit criteria:

- Dedicated `uavcan` runtime-execution test lanes for Python are green in matrix full run.

### Workstream C: Specialization Strategy Parity

Deliver:

- Introduce explicit runtime specialization flag for Python generation:
  - `--py-runtime-specialization <portable|fast>`
- Keep generated type files semantically stable across specializations.
- Add semantic-diff and parity checks:
  - portable vs fast generated runtime differences are intentional and constrained
  - outputs remain wire-compatible and parity-safe

Exit criteria:

- Specialization lane set mirrors TS/Rust confidence model:
  - generation
  - semantic-diff
  - C<->Python parity
  - runtime smoke

Notes:

- Existing `auto|pure|accel` backend selection remains runtime backend policy.
- Specialization controls generated pure-runtime behavior and optimization shape.

### Workstream D: Packaging + Accelerator Ergonomics

Deliver:

- Add optional emitted Python package metadata:
  - `pyproject.toml` (minimal local installability)
  - `py.typed` marker for typing-aware users
- Add consistent accelerator artifact staging command/target:
  - copy accelerator beside generated package when available
  - fail with actionable diagnostics if forced accel requested but unavailable
- Document pure-only and accel-enabled install/run workflows.

Exit criteria:

- A user can generate, install, and run produced Python artifacts in a clean venv with documented commands.

### Workstream E: Performance Policy + Regression Gates

Deliver:

- Expand benchmark harness corpus (small/medium/large payload families).
- Capture per-mode measurements (`pure`, `accel`, specialization variants).
- Add non-failing CI artifact reporting first, then optional regression threshold checks.

Exit criteria:

- Benchmark reports are emitted in CI for Python lanes.
- Accelerator regressions are detectable and attributable.

### Workstream F: Docs + Demo Completion

Deliver:

- Update `README.md`, `DESIGN.md`, `CONTRIBUTING.md`, `DEMO.md` with:
  - parity test taxonomy for Python
  - specialization and backend-selection model
  - packaging and accelerator usage
  - troubleshooting matrix
- Add a short Python demo flow equivalent in quality to existing language demos.

Exit criteria:

- Python is represented at the same depth as other mature backends in docs and demos.

## 6. Test Matrix Additions

Add Python-labeled tests so matrix workflows can run:

- Debug smoke:
  - fixtures generation hardening
  - runtime smoke
  - backend selection
- RelWithDebInfo full:
  - all C<->Python parity families
  - full `uavcan` runtime execution
  - determinism
  - specialization semantic-diff/parity lanes
- Release smoke:
  - minimal runtime round-trip
  - package import sanity

## 7. Implementation Order (Fastest Path)

1. Workstream A (differential parity) and B (full-tree runtime execution).
2. Workstream C (specialization model and gates).
3. Workstream D (packaging/accelerator ergonomics).
4. Workstream E (perf policy).
5. Workstream F (docs/demo closure).

This order maximizes confidence first, then polish.

## 8. Definition of Done Checklist

- [x] C<->Python parity families added and stable.
- [x] Signed-narrow parity (normal + optimized) passing.
- [x] Full `uavcan` runtime-execution lane added and passing.
- [x] Python specialization flag and semantic-diff/parity lanes implemented.
- [x] Packaging/install docs and accelerator staging flow complete.
- [x] Benchmark artifacts integrated and reviewed.
- [x] Demo and docs updated to parity depth.

## 9. Risk Notes and Mitigations

- Risk: Python-vs-C parity mismatch due to numeric edge handling.
  - Mitigation: prioritize signed-narrow and float edge corpus early.
- Risk: specialization introduces behavior drift.
  - Mitigation: semantic-diff + differential parity in same CI run.
- Risk: accelerator availability differs by environment.
  - Mitigation: keep pure runtime always functional; accel lanes optional but explicit.
- Risk: integration runtime on CI can be flaky.
  - Mitigation: deterministic fixtures, strict seed control, and tight failure diagnostics.
