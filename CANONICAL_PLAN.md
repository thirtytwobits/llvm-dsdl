# Canonical Remaining Work Plan

Date: February 19, 2026  
Scope: Remaining work after completion of the MLIR-max lowering program for current backend set.

## 1. Purpose

This is the single canonical plan for work remaining in this project.  
It supersedes and consolidates prior planning documents.

## 2. Current Baseline (What Is Already Done)

As of February 19, 2026:

1. Core MLIR-max lowering phases are complete for the current backend scope.
2. Backends in active scope are implemented and integrated:
   - C
   - C++ (`std`, `pmr`)
   - Rust (`std`, `no-std-alloc`, runtime specialization `portable|fast`)
   - Go
   - TypeScript
3. TypeScript completion backlog (strict + compat + parity + optimized lanes) is complete per prior plan records.
4. Lowered contract markers and hard-fail validation are enforced end-to-end.
5. Cross-language parity model is established with inventory/pass-line invariants.
6. Optional optimization pipeline exists (`--optimize-lowered-serdes`) with parity-preserving gates.

## 3. Retained Non-Negotiable Contract/Data

The following project rules are retained and should not regress:

1. Lowered SerDes contract remains versioned and producer-marked (currently version `1`), as defined in `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/LOWERED_SERDES_CONTRACT.md`.
2. Consumers fail hard on missing/mismatched required lowered contract metadata.
3. `--strict` remains default spec-first mode.
4. `--compat-mode` remains migration mode for legacy/non-conformant trees, with deterministic warnings and documented migration back to strict mode.
5. Parity gates remain invariant-based (not smoke-only), including optimized variants.

## 4. Remaining Workstreams (Canonical Backlog)

## Workstream A: Render-IR Convergence Completion (Priority P0)

Objective: finish optional-stretch convergence so backends are thin render adapters over shared lowered products.

Tasks:

1. Inventory remaining backend-local semantic traversal branches in:
   - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/CppEmitter.cpp`
   - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/RustEmitter.cpp`
   - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/GoEmitter.cpp`
   - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/TsEmitter.cpp`
2. Move remaining semantics into shared planning/render modules (`*Plan*`, `LoweredRenderIR`, helper resolvers/descriptors).
3. Keep backend-local code limited to syntax/runtime binding and language surface.
4. Add/expand unit tests for each new shared planner or render primitive.

Acceptance:

1. No new backend-local semantic fallback paths introduced.
2. Existing parity and integration suites remain green, including optimized lanes.
3. `DESIGN.md` reflects final shared-vs-backend boundary with concrete examples.

## Workstream B: Profile and Runtime Specialization Expansion (Priority P1)

Objective: add profile flexibility without wire-semantic drift.

Tasks:

1. Expand profile matrix where beneficial (embedded/runtime-specialized variants) while preserving lowered semantics.
2. For each profile addition, add:
   - generation gate
   - compile/build gate
   - parity-equivalence gate
   - semantic-diff gate where relevant
3. Preserve strict/compat behavior equivalence guarantees for supported profiles.

Acceptance:

1. Profile additions have deterministic, documented CLI UX.
2. Parity results are unchanged relative to canonical strict semantics.
3. No contract-version drift is required unless schema changes are intentional.

## Workstream C: Test and Validation Hardening (Priority P1)

Objective: strengthen confidence and lower regression risk.

Tasks:

1. Expand directed parity vectors for deep-nesting and cross-family mixed error paths.
2. Add targeted negative tests for malformed/missing lowered metadata families across all emitters.
3. Add reliability checks for deterministic generation artifacts under concurrent test execution.
4. Keep TypeScript strict/compat parity gates continuously green in the same invariant model as other language pairs.

Acceptance:

1. Full workflow and focused differential/parity rings pass consistently.
2. No flaky nondeterminism observed in generation/parity artifacts.
3. New edge vectors are documented and reproducible.

## Workstream D: Productization and Release Readiness (Priority P0)

Objective: make the project easier to ship, demo, and consume.

Tasks:

1. Keep one canonical demo flow current (`DEMO.md`) with short and scale-up paths.
2. Keep one canonical architecture snapshot current (`DESIGN.md`).
3. Keep one canonical status/runbook current (`README.md`) with strict/compat clarity.
4. Add/maintain a release checklist (toolchain versions, full gate command set, artifact expectations).
5. Keep formatting/lint utilities (`check-format`, `format-source`) healthy and documented.

Acceptance:

1. A clean environment can execute documented setup, generate outputs, and run critical gates.
2. Docs do not contain stale experimental caveats that conflict with tested status.
3. Demo remains runnable in under five minutes for fixture corpus.

## Workstream E: TODO Closure and Upstream Coordination (Priority P2)

Objective: close remaining explicit TODOs and avoid silent backlog leakage.

Tasks:

1. Fix `public_regulated_data_types/verify` handling of `#` inside string literals.
2. Classify `Status.0.1.dsdl` TODO as:
   - upstream content roadmap item, or
   - local action item with owner/date.
3. Add a tiny “open TODO ledger” section in this plan and keep it current.

Acceptance:

1. Repo-level TODO scan returns only intentionally deferred, documented items.
2. Each deferred item has owner, scope, and disposition.

## 5. Milestones

## M1: Convergence and Release Baseline

Target:

1. Workstream A initial pass complete.
2. Workstream D release checklist added.
3. Workstream E TODO dispositions recorded.

Exit criteria:

1. Full test workflow is green.
2. No unresolved ambiguity on strict/compat semantics in docs.

## M2: Expanded Profiles and Hardening

Target:

1. Workstream B prioritized profile additions complete.
2. Workstream C expanded edge/negative coverage complete.

Exit criteria:

1. Parity invariants remain stable under added profiles and optimized lowering.
2. No semantic drift in generated wire behavior.

## M3: Stabilized “Next-Backend-Ready” State

Target:

1. Backends are demonstrably thin adapters over shared render/lowered layers.
2. Regression model is robust enough to absorb additional target languages with low risk.

Exit criteria:

1. Shared layer ownership and extension points are clear and documented.
2. New backend onboarding checklist is documented (inputs, invariants, gates).

## 8. Canonical Verification Commands

Primary:

1. `cmake --workflow --preset full`
2. `ctest --test-dir build/dev-homebrew --output-on-failure`
3. `ctest --test-dir build/dev-homebrew -L ts --output-on-failure`
4. `cmake --build build/dev-homebrew --target check-format -j1`

Optional (performance/parity drift checks):

1. optimization-enabled parity workflows (`--optimize-lowered-serdes`)
2. differential parity workflows by label/preset

## 6. Plan Maintenance Rules

1. This file is the only active planning source for remaining project work.
2. Completed items are recorded here with date and evidence command(s).
3. New TODOs found in code are appended to Section 5/Workstream E immediately.
4. Contract schema changes require synchronized updates to:
   - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/LOWERED_SERDES_CONTRACT.md`
   - tests
   - this canonical plan milestones/acceptance if scope changes.
