# Go Integration Plan (MLIR-First, Shared-Infrastructure-Max)

Date: 2026-02-17  
Objective: Add a Go backend that reaches parity with current quality gates while maximizing reuse of existing shared MLIR/CodeGen infrastructure.

## What "Same Level" Means

By completion, Go should have:

1. `dsdlc go` command with deterministic output and strict-mode compatibility.
2. Full `uavcan` generation target(s) in CMake presets/targets.
3. Runtime + generated Go SerDes that passes compile checks.
4. Differential parity tests against C reference (random + directed edge vectors), same style as C++/Rust parity.
5. Maximum practical reuse of existing shared modules (`MlirLoweredFacts`, helper planners, statement planners, wire facts, etc.).

## Current Starting Point

1. Shared planning/facts modules already exist and are used by C++/Rust.
2. C remains the deepest MLIR-lowered oracle for behavior.
3. `go` toolchain is not currently installed in this environment (`go: command not found`).

## Architecture Approach

1. Keep MLIR as behavioral source of truth.
2. Add `GoEmitter` as a thin syntax backend on top of shared plans.
3. Expand shared plan/render layers first where C++/Rust still duplicate logic; use those for Go too.
4. Use C parity harnesses as acceptance oracle.

## Implementation Phases

### Phase 0: Toolchain + Scaffolding

1. Install/verify Go toolchain (`go`, `gofmt`, `go test`).
2. Add CLI command path: `dsdlc go`.
3. Add emitter options:
   - module/package naming
   - output layout
   - profile placeholder (if needed for future allocator/embedded variants)

Acceptance:
- `dsdlc go --help` path works.
- Basic single-type generation emits compilable Go.

### Phase 1: Shared-Core Lift (Before Deep Go Logic)

1. Continue lifting scalar/composite statement shaping into shared contracts.
2. Ensure shared contracts are language-neutral enough for Go:
   - expression-level normalization hooks
   - array/union/composite path metadata
   - delimiter and bounds semantics

Acceptance:
- New shared contracts used by C++/Rust with no regressions.
- Unit tests added for new shared planners/render helpers.

### Phase 2: Go Runtime + Type Model

1. Add `runtime/go/dsdl_runtime.go`:
   - bit I/O
   - scalar normalization helpers
   - delimiter/array/union validation helpers
   - canonical error codes aligned with existing runtime semantics
2. Map DSDL types to Go representation:
   - scalars, arrays, composites, tagged unions, services
3. Define generated API shape:
   - `Serialize` / `DeserializeWithConsumed`
   - deterministic constants/macros equivalent

Acceptance:
- Generated Go compiles for representative sample definitions.
- Runtime helper unit tests (or focused integration checks) pass.

### Phase 3: GoEmitter (Shared-Plan Consumption)

1. Implement `GoEmitter.cpp` with minimal backend-specific branching.
2. Reuse shared modules:
   - lowered facts
   - helper symbol resolution
   - section helper binding planning/rendering (or Go-specific renderer on same plan)
   - statement planning (including MLIR `step_index`)
   - array wire plan and type storage utilities
3. Generate per-type files + module/package index.

Acceptance:
- Full `uavcan` Go generation succeeds in strict mode.
- Output deterministic across repeated runs.

### Phase 4: Integration + Differential Tests

1. Add CMake integration tests:
   - `llvmdsdl-uavcan-go-generation`
   - `llvmdsdl-uavcan-go-build` (`go test` or `go test ./...`)
   - `llvmdsdl-uavcan-c-go-parity`
2. Port directed parity vectors used in C++/Rust parity suites:
   - union-tag errors
   - array-length errors
   - delimiter header errors
   - truncation/zero-extension paths
   - service request/response edge paths
3. Add signed narrow fixture parity for Go if feasible in same window.

Acceptance:
- Focused ring including Go passes.
- Full `ctest` passes with Go-enabled tests.

### Phase 5: Productization

1. Add `generate-uavcan-go` and include in aggregate targets.
2. Update `README.md` and `DESIGN.md` with:
   - backend status
   - API shape
   - known limits
3. Add demo commands and artifact paths.

Acceptance:
- One-command reproducible Go generation documented.

## Proposed Test Matrix Additions

1. Unit:
   - Go-specific renderer helpers (if introduced).
   - Any new shared statement-plan modules.
2. Integration:
   - strict full-tree Go generation
   - Go compile check
   - C/Go parity (random + directed)
3. Stability:
   - deterministic file ordering/contents checks.

## Autonomous Execution Estimate

Assuming maximum autonomous execution and no major spec surprises:

1. **Core Go backend to "useful + compiling + full uavcan generation"**: 8-12 hours.
2. **Bring to current parity bar (C parity harness + directed vectors + CI-level integration quality)**: +8-12 hours.
3. **Total to “same level”**: **16-24 hours** of focused work (roughly **2 long workdays**, or **1.5 very strong days**).

## Time To First Demo Today (By 8 PM)

If we start now and run hard in autonomous mode:

1. Likely by 8 PM:
   - `dsdlc go` exists
   - substantial shared-plan reuse
   - full `uavcan` generation likely green
   - Go compile checks likely green
2. Possible but not guaranteed by 8 PM:
   - full C/Go parity suite with all directed vectors matching current C++/Rust depth

## Risks and Mitigations

1. Risk: Go type/union modeling mismatch.
   - Mitigation: reuse C as oracle; keep union encoding semantics in shared plans.
2. Risk: hidden backend assumptions in shared modules.
   - Mitigation: add Go while still refactoring shared plans; do not finalize abstractions beforehand.
3. Risk: schedule risk from toolchain setup.
   - Mitigation: install/verify Go first; fail fast if unavailable.

## Recommended Execution Order

1. Install Go toolchain.
2. Implement minimal `dsdlc go` + runtime + compile pass.
3. Add full `uavcan` generation target.
4. Add parity harness and directed vectors.
5. Tighten shared-plan usage and remove duplicated Go-specific semantics as we go.
