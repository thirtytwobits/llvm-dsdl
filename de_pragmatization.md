# De-Pragmatization Plan: MLIR-Authoritative Multi-Backend Lowering

## Summary

This plan defines how to move `llvm-dsdl` from the current pragmatic architecture to a stricter compiler architecture where:

1. DSDL wire-semantics invariants are enforced primarily in dialect verifiers.
2. All language backends consume one canonical lowered wire-program model.
3. Backend emitters focus on syntax/runtime API binding, not semantic orchestration.
4. Hand-written runtime code is limited to low-level primitives.

This is an internal architecture-hardening plan, not a public CLI/API redesign.

Compatibility policy for this plan is hard-cut:

1. Breaking internal refactors are acceptable.
2. No backward-compatibility shims, dual paths, or migration scaffolding should be introduced.
3. Required updates are limited to this repository's own consumers (tests, `dsdld`, VSCode client integration, docs, and build tooling).

## Why This Exists

`DESIGN.md` explicitly calls out current pragmatic boundaries:

1. Some invariants are validated in transform-contract code instead of op verifiers.
2. C has the deepest direct MLIR-to-code path.
3. Other backends are MLIR-informed but still do language-local rendering choreography.
4. Runtime primitives are hand-maintained.

This plan closes those gaps intentionally and incrementally.

## Scope

In scope:

1. Dialect verifier hardening for wire-semantics invariants.
2. Canonical wire-program lowering representation and consumers.
3. Shared backend execution/render planning across C/C++/Rust/Go/TS/Python.
4. Runtime helper generation consolidation.
5. Convergence and parity gate expansion.
6. Documentation and contributor guardrails.

Out of scope:

1. Replacing low-level bit/float/buffer primitives in `runtime/`.
2. New end-user CLI behavior changes in `tools/dsdlc/main.cpp`.
3. Backend removals or profile removals.
4. Windows-specific runtime/bundling re-architecture.
5. Maintaining compatibility layers for pre-refactor internal APIs.

## Architecture End-State

At completion:

1. Invalid DSDL IR fails verifier checks before transform passes.
2. A single canonical lowered wire-program contract is the semantic source of truth.
3. Every backend consumes the same semantic operation stream and helper contracts.
4. Backend-local logic is constrained to:
   - naming/syntax
   - package/module layout
   - runtime primitive call spelling
   - profile/memory policy surface wiring
5. Cross-backend byte parity and malformed-input contract parity are enforced by CI.

## Workstream A: Verifier-First Invariant Enforcement

Goal:
Move high-value correctness checks from transform-contract code to dialect op verifiers.

Tasks:

1. Inventory all contract checks currently implemented in:
   - `include/llvmdsdl/Transforms/LoweredSerDesContract.h`
   - `lib/Transforms/Passes.cpp`
   - `lib/Transforms/ConvertDSDLToEmitC.cpp`
2. Classify each check as:
   - verifier-level invariant (must hold for valid IR)
   - transform-time pipeline precondition
   - backend preflight assertion
3. Implement verifier-level checks in:
   - `include/llvmdsdl/IR/DSDLOps.td`
   - `lib/IR/DSDLOps.cpp`
4. Keep transform checks only where pass-local context is required.
5. Add negative verifier lit tests for malformed IR rejection.

Acceptance criteria:

1. Core wire invariants fail at verifier stage.
2. Transform contract layer is smaller and focused on pass preconditions.
3. Existing lowering/pass tests remain green.

## Workstream B: Canonical Wire-Program Contract v2

Goal:
Define and adopt a backend-neutral operation contract so all emitters consume one semantic body plan.

Tasks:

1. Introduce/upgrade shared operation model in:
   - `include/llvmdsdl/CodeGen/LoweredRenderIR.h`
   - `include/llvmdsdl/CodeGen/RuntimeLoweredPlan.h`
   - `include/llvmdsdl/CodeGen/ScriptedOperationPlan.h`
2. Ensure operation model carries all information required for:
   - scalar normalize/sign-extend
   - union tag read/write/validate/mask
   - fixed/variable array count/prefix/validate
   - delimiter and capacity guards
   - alignment/padding
3. Eliminate backend-specific semantic lookups that bypass the shared contract.
4. Add a versioned schema marker for the wire-program contract.

Acceptance criteria:

1. Backends can render from a single operation contract without semantic fallbacks.
2. Operation contract versioning is explicit and validated.

## Workstream C: Shared Emitter Execution Engine

Goal:
Stop duplicating control-flow orchestration in backend emitters.

Tasks:

1. Consolidate common execution skeleton for native backends around:
   - `include/llvmdsdl/CodeGen/NativeFunctionSkeleton.h`
   - `include/llvmdsdl/CodeGen/NativeEmitterTraversal.h`
2. Consolidate scripted orchestration around:
   - `include/llvmdsdl/CodeGen/ScriptedOperationPlan.h`
   - `include/llvmdsdl/CodeGen/RuntimeHelperBindings.h`
3. Restrict backend files:
   - `lib/CodeGen/CEmitter.cpp`
   - `lib/CodeGen/CppEmitter.cpp`
   - `lib/CodeGen/RustEmitter.cpp`
   - `lib/CodeGen/GoEmitter.cpp`
   - `lib/CodeGen/TsEmitter.cpp`
   - `lib/CodeGen/PythonEmitter.cpp`
   to rendering and runtime binding responsibilities.
4. Add static assertions/tests that detect reintroduction of inline semantic fallback patterns.

Acceptance criteria:

1. Semantic orchestration duplication is removed from backend emitters.
2. Fallback signature scans remain clean in integration tests.

## Workstream D: Deepen Non-C Backends to C-Level MLIR Authority

Goal:
Reduce the architectural asymmetry where C is deepest in the MLIR pipeline.

Tasks:

1. Ensure C++/Rust/Go/TS/Python all consume shared lowered facts and operation contracts at parity depth.
2. Move any remaining emitter-local semantic derivations into shared planning layers.
3. Make backend preflight contract validation mandatory and uniform.
4. Add convergence-report dimensions specifically scoring:
   - verifier-first checks
   - contract-v2 usage
   - fallback-free semantic execution

Acceptance criteria:

1. C is no longer the only backend with deep semantic authority from IR.
2. Convergence score floor is raised and enforced.

## Workstream E: Runtime Boundary Minimization

Goal:
Keep runtime hand-written code limited to primitive operations; generate semantic wrappers.

Tasks:

1. Enumerate runtime functions per language and classify as:
   - primitive (keep hand-written)
   - semantic wrapper (candidate for generation)
2. Generate helper/wrapper layer from lowered contracts where feasible.
3. Keep manually maintained files focused on:
   - bit extraction/copy
   - float packing/unpacking
   - bounded buffer access
4. Update runtime docs in:
   - `runtime/` sources
   - `DESIGN.md`
   - `README.md`
   - `CONTRIBUTING.md`

Acceptance criteria:

1. Runtime semantics above primitive level are generated/shared where practical.
2. Primitive layer remains explicit and audited.

## Workstream F: Test and Regression Gate Expansion

Goal:
Make the de-pragmatized architecture enforceable by tests and CI policy.

Tasks:

1. Add/expand unit tests for:
   - verifier invariants
   - contract-v2 planning
   - helper-binding completeness
2. Expand integration gates:
   - determinism across all backends
   - cross-backend byte parity fixture families
   - malformed-input parity across runtime modes
3. Expand convergence tooling in `tools/convergence/`:
   - score penalties for backend-local semantic fallbacks
   - report cards for contract-v2 adoption
4. Keep full matrix green under standard workflows in `CMakePresets.json`.

Acceptance criteria:

1. CI fails on semantic divergence regressions.
2. Convergence/parity reports become release-blocking evidence.

## Workstream G: Migration, Cleanup, and Hard Cut

Goal:
Enforce a no-shim hard cut and remove any temporary migration scaffolding.

Tasks:

1. Do not add compatibility wrappers/aliases for refactored planning APIs.
2. Delete fallback branches and any temporary dual-path code as each slice lands.
3. Update in-repo call sites immediately (tests, tools, docs, editor integrations).
4. Simplify docs and contributor guidance to one canonical path only.
5. Confirm no references to compatibility shims/migration machinery remain.

Acceptance criteria:

1. No dual-path semantic architecture remains.
2. No compatibility shim layer exists in codegen/planning surfaces.
3. Contributor docs reflect only canonical shared-lowering flow.

## Execution Order

1. Workstream A
2. Workstream B
3. Workstream C
4. Workstream D
5. Workstream E
6. Workstream F
7. Workstream G

## Risks and Mitigations

Risk: behavior drift while moving checks and orchestration.
Mitigation: add golden determinism/parity baselines before each migration slice.

Risk: over-generalization hurts readability.
Mitigation: share semantics/execution only; keep syntax rendering backend-local.

Risk: performance regressions in shared planning layers.
Mitigation: benchmark and convergence checks per backend and runtime mode.

Risk: long-lived migration shims become permanent.
Mitigation: do not introduce shims; apply immediate hard-cut updates across repo-local consumers.

## Definition of Done

1. Verifier-first invariant enforcement is in place for core wire semantics.
2. All backends consume a canonical lowered wire-program contract.
3. Backend emitters no longer contain semantic fallback choreography.
4. Runtime layers are limited to primitives plus generated/shared semantics above them.
5. Convergence/parity/malformed-contract gates are green and release-blocking.
6. Documentation clearly describes the non-pragmatic architecture as the only supported model.
