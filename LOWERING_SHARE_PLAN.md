# MLIR-Lowered Sharing Plan: Pull More Logic Out of Language-Specific Codegen

## Summary
This plan extracts remaining shared serialization/deserialization semantics from language-specific emitters into backend-neutral planning and helper-resolution layers. The immediate target is TypeScript and Python (highest duplication), followed by cleanup passes for C++, Rust, and Go.

## Goals
1. Keep semantic authority in lowered MLIR facts and shared planning utilities.
2. Reduce duplicated emitter logic without changing generated public APIs.
3. Preserve byte-for-byte behavior parity across supported languages.
4. Keep low-level runtime primitives hand-written and language-native.

## Non-Goals
1. Replacing runtime bit/float primitives in `runtime/`.
2. Introducing a new CLI surface for `dsdlc`.
3. Rewriting the C backend architecture in this slice.

## Current Duplication Hotspots
1. `lib/CodeGen/TsEmitter.cpp` and `lib/CodeGen/PythonEmitter.cpp` both implement:
   - Section helper-name lookup and symbol wiring.
   - Union tag control flow and validation choreography.
   - Array prefix/length validation control flow.
   - Alignment and padding choreography.
   - Delimited composite read/write guard logic.
2. `lib/CodeGen/CppEmitter.cpp`, `lib/CodeGen/RustEmitter.cpp`, and `lib/CodeGen/GoEmitter.cpp` duplicate traversal and orchestration patterns around lowered render IR callbacks.
3. Error/diagnostic text is partially duplicated across emitters, risking drift.

## Architecture Direction
1. Keep `LoweredRenderIR` as the canonical ordered body skeleton.
2. Introduce shared runtime-body planning primitives that are language-neutral.
3. Centralize helper symbol lookup and field helper binding resolution.
4. Keep language emitters focused on syntax rendering and runtime call names.

## Workstream A: Baseline, Invariants, and Safety Rails
1. Snapshot TS/Python generated outputs for representative fixtures before refactor.
2. Add parity fixtures that compare serialized bytes across TS/Python/C++ (or Rust baseline).
3. Add deterministic regeneration checks for TS/Python.
4. Define malformed-input contract invariants as non-regression gates.

Deliverables:
1. Additional lit/integration fixtures in `test/lit/` and `test/integration/`.
2. Baseline output snapshots for at-risk types (unions, var arrays, delimited composites).

## Workstream B: Generalize Runtime Planning Types
1. Introduce shared runtime planning types not branded as TypeScript-only.
2. Move planning logic from `TsLoweredPlan` naming into generic runtime plan APIs.
3. Preserve `semanticFieldName` and lowered-order guarantees.
4. Remove temporary TypeScript-prefixed aliases/wrappers once migration is complete.

Proposed files:
1. `include/llvmdsdl/CodeGen/RuntimeLoweredPlan.h` (new)
2. `lib/CodeGen/RuntimeLoweredPlan.cpp` (new)
3. Remove `include/llvmdsdl/CodeGen/TsLoweredPlan.h` temporary wrapper file.
4. Remove `lib/CodeGen/TsLoweredPlan.cpp` temporary wrapper file.

## Workstream C: Centralize Helper Name Resolution and Binding Lookup
1. Extract duplicated TS/Python helper lookup code into shared utility APIs.
2. Share section helper names and per-field helper binding resolution.
3. Keep only language-specific symbol sanitation/name-mangling in emitters.

Proposed files:
1. `include/llvmdsdl/CodeGen/RuntimeHelperBindings.h` (new)
2. `lib/CodeGen/RuntimeHelperBindings.cpp` (new)
3. `lib/CodeGen/TsEmitter.cpp` (consume shared helper binding lookup)
4. `lib/CodeGen/PythonEmitter.cpp` (consume shared helper binding lookup)

## Workstream D: Shared Control-Flow Rendering Plan for Scripted Backends
1. Add a shared intermediate body-control plan for scripted/runtime-managed backends.
2. Model repeated shapes explicitly:
   - Union tag read/write/validate/mask.
   - Fixed/variable array loops.
   - Scalar/composite dispatch.
   - Delimiter and capacity checks.
3. Keep language emitters responsible for:
   - Syntax surface.
   - Runtime primitive function names.
   - Type annotation style.

Proposed files:
1. `include/llvmdsdl/CodeGen/ScriptedBodyPlan.h` (new)
2. `lib/CodeGen/ScriptedBodyPlan.cpp` (new)

## Workstream E: Migrate TypeScript Emitter to Shared Runtime Body Plan
1. Rewire `lib/CodeGen/TsEmitter.cpp` to consume shared helper lookups and shared body-control plan.
2. Remove duplicated per-field/per-array/per-union orchestration logic.
3. Preserve current TypeScript API semantics (`number | bigint` behavior, error strings, signatures).

Validation gates:
1. Existing TS unit, lit, and integration tests pass.
2. No deterministic output diffs except intentional formatting-only changes.

## Workstream F: Migrate Python Emitter to Shared Runtime Body Plan
1. Rewire `lib/CodeGen/PythonEmitter.cpp` to consume the same shared plan and helper lookup utilities.
2. Remove duplicated orchestration logic mirrored from TypeScript.
3. Preserve malformed-input contract across portable/fast/accel runtime modes.

Validation gates:
1. Existing Python integration tests pass in all runtime modes.
2. Portable/fast/accel malformed decode parity remains intact.

## Workstream G: Native Emitter Orchestration Cleanup (C++/Rust/Go)
1. Identify shared traversal boilerplate in:
   - `lib/CodeGen/CppEmitter.cpp`
   - `lib/CodeGen/RustEmitter.cpp`
   - `lib/CodeGen/GoEmitter.cpp`
2. Introduce shared helper(s) for alignment/padding and lowered-step traversal orchestration.
3. Keep per-language runtime primitive calls and ownership/memory semantics intact.

Potential shared utility:
1. `include/llvmdsdl/CodeGen/NativeEmitterTraversal.h` (new)
2. `lib/CodeGen/NativeEmitterTraversal.cpp` (new)

## Workstream H: Diagnostic and Contract Message Catalog
1. Centralize recurring error text used by multiple emitters.
2. Use shared constants/templates for messages that must remain parity-locked.
3. Keep language-specific wrapping (`throw`, `raise`, error codes) local.

Proposed files:
1. `include/llvmdsdl/CodeGen/CodegenDiagnosticText.h` (new)
2. `lib/CodeGen/CodegenDiagnosticText.cpp` (new)

## Workstream I: Test Expansion and Regression Armor
1. Unit tests:
   - Shared helper binding lookup resolution.
   - Shared runtime planning for unions/arrays/composites.
2. Integration tests:
   - Boundary values for signed/unsigned scalar semantics.
   - Variable array prefix and validation parity.
   - Delimited composite malformed header handling.
3. Determinism and parity:
   - Golden output checks for TS/Python.
   - Cross-language serialized-byte parity fixtures.

Proposed test files:
1. `test/unit/RuntimeLoweredPlanTests.cpp` (new)
2. `test/unit/RuntimeHelperBindingsTests.cpp` (new)
3. Existing updates in `test/unit/HelperBindingRenderTests.cpp` and integration/lit suites.

## Workstream J: Documentation and Maintainer Guidance
1. Update `DESIGN.md` with new shared layers and reduced emitter responsibilities.
2. Update `README.md` backend notes to clarify generated vs hand-written boundaries.
3. Update `CONTRIBUTING.md` with guardrails to prevent semantic logic re-duplication.
4. Add a short migration note for any renamed internal planning APIs.

## Execution Order
1. A -> B -> C
2. D -> E -> F
3. G -> H
4. I and J continuously, with a final hardening pass after G/H

## Acceptance Criteria
1. TS/Python orchestration semantics are driven by shared planning/binding layers.
2. Native emitter traversal duplication is reduced without behavior changes.
3. Cross-language parity fixtures pass for representative corpus.
4. Deterministic generation checks pass for TS/Python.
5. Full `ctest` sweep passes on the matrix baseline used by the project.
6. Documentation reflects the new architecture and extension workflow.

## Risk Management
1. Risk: behavior drift during refactor.
   - Mitigation: baseline snapshots + byte parity fixtures before edits.
2. Risk: over-generalization that harms readability.
   - Mitigation: keep syntax emission local; share only orchestration/semantic decisions.
3. Risk: rollout churn from API renaming.
   - Mitigation: temporary compatibility aliases and staged file migration.

## Definition of Done
1. No duplicated semantic orchestration remains between TS and Python emitters.
2. Shared planning and helper lookup utilities are the single source of truth for these semantics.
3. C++/Rust/Go share traversal helpers where practical, without regressing performance or readability.
4. Tests and docs are updated, and full suite is green.
