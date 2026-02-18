# Plan: Maximally Lowered DSDL MLIR Across All Targets

## Objective

Make lowered DSDL MLIR the single source of truth for SerDes behavior across all current targets:

1. C
2. C++23 (`std`)
3. C++23 (`pmr`)
4. Rust (`std`)
5. Go

Backend emitters should become syntax/type adapters over a shared lowered program contract, not independent semantic implementations.

## Execution Status (February 18, 2026)

1. Phase 0 (Contract Freeze): in progress, core contract versioning implemented. Phases 2 and 3 started.
2. Added canonical lowered-contract markers on module and `dsdl.serialization_plan`.
3. Added consumer-side contract checks in lowered-fact collection and `convert-dsdl-to-emitc`.
4. Added positive/negative tests for contract presence and pass preconditions.
5. Added contract reference: `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/LOWERED_SERDES_CONTRACT.md`.
6. Tightened `lower-dsdl-serialization` canonicalization: missing/invalid core metadata now hard-fails instead of silently defaulting.
7. Added strict lowering contract lit negatives for missing step kind, missing union metadata, and invalid variable-array prefix width.
8. Added UTF-8 scalar helper coverage to keep scalar normalization aligned with lowered helper contract expectations.
9. Added shared lowered-body composition API: `buildLoweredBodyPlan(section, sectionFacts, direction)`.
10. Rewired C++, Rust, and Go body emitters to consume shared lowered body-plan products for statement order, helper bindings, capacity checks, and union tag helper usage.
11. Added unit coverage for shared lowered-body composition in `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/unit/LoweredBodyPlanTests.cpp`.
12. Validation: full `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20), including parity and generation suites.
13. Tightened lowered-fact ingestion to hard-fail on empty or unresolved step-level helper symbols for array-prefix/array-validate, scalar unsigned/signed/float, and delimiter-validate families.
14. Added integration guards that reject generated fallback saturation signatures in C++, Rust, and Go UAVCAN generation checks.
15. Revalidated after fallback-elimination guards: full `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20).
16. Removed backend-local saturation fallback code paths from C++, Rust, and Go scalar serialize emitters; lowered helper bindings are now required for normalization in these paths.
17. Tightened `convert-dsdl-to-emitc` step-level contract checks to require non-empty and symbol-resolved helper bindings for scalar, variable-array, and delimiter helper families.
18. Added negative contract lit coverage for unresolved scalar/array-prefix/delimiter helper symbols:
    - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/lit/convert-dsdl-to-emitc-scalar-helper-symbol-contract.mlir`
    - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/lit/convert-dsdl-to-emitc-array-helper-symbol-contract.mlir`
    - `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/lit/convert-dsdl-to-emitc-delimiter-helper-symbol-contract.mlir`
19. Revalidated fallback-elimination updates: manual `dsdl-opt` checks for new negatives plus full `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20).
20. Completed Go fallback elimination parity with C++/Rust for remaining helper-dependent paths:
    - scalar serialize/deserialize (unsigned, signed, float) now hard-fail on missing lowered helper bindings
    - variable-array prefix/validate helper absences now hard-fail instead of backend-local bounds fallback
    - delimiter-validate helper absences now hard-fail instead of backend-local remaining-bytes fallback
21. Revalidated after Go hard-fail alignment:
    - manual negative checks via `build/dev-homebrew/tools/dsdl-opt/dsdl-opt` for scalar/array-prefix/delimiter unresolved-helper lit cases
    - full `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
22. Removed obsolete unused C++ backend helper (`signedGetter`) after helper-contract hardening and revalidated with full `ctest --test-dir build/dev-homebrew --output-on-failure` (20/20).
23. Expanded Phase 3 generation hardening guards in C++/Rust/Go integration checks:
    - now require helper call-site usage for scalar/array-prefix/array-validate/delimiter paths in generated outputs
    - reject residual inline fallback signatures for array-length checks, delimiter remaining-bytes checks, and direct runtime scalar reads in generated deserialize paths
    - preserved existing saturation fallback rejection and Go union/capacity helper checks
24. Revalidated guard expansion:
    - targeted generation gate `ctest --test-dir build/dev-homebrew --output-on-failure -R "uavcan-(cpp|rust|go)-generation"` passed (3/3)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
25. Started Phase 4 parity invariant convergence by hardening differential parity to the same summary/inventory/count model:
    - differential harness now emits canonical summary and inventory markers plus explicit directed-baseline marker
    - differential CMake gate now enforces summary parse, inventory consistency, random pass-line count, and directed pass-line count invariants
26. Revalidated differential parity hardening:
    - targeted gate `ctest --test-dir build/dev-homebrew --output-on-failure -R differential-parity` passed (1/1)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
27. Continued Phase 4 parity invariant convergence by normalizing core parity marker schema across C/C++/Rust/Go harnesses and gates:
    - canonicalized parity summary markers to `random_iterations`, `random_cases`, `directed_cases`
    - canonicalized inventory markers to `random_cases`, `directed_cases`
    - updated `RunCppCParity.cmake`, `RunCRustParity.cmake`, and `RunCGoParity.cmake` parsers to consume the unified marker schema
28. Revalidated marker-schema normalization:
    - targeted parity gate `ctest --test-dir build/dev-homebrew --output-on-failure -R "uavcan-(cpp-c|cpp-pmr-c|c-rust|c-go)-parity"` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
29. Completed signed-narrow parity marker-schema cleanup for C/Go to match family naming conventions:
    - normalized signed-narrow C/Go inventory marker prefix to `signed-narrow-c-go-parity`
    - updated signed-narrow C/Go CMake parser to consume the normalized inventory marker
30. Revalidated signed-narrow marker cleanup:
    - targeted signed-narrow parity gate `ctest --test-dir build/dev-homebrew --output-on-failure -R "signed-narrow-c-go-parity|signed-narrow-c-rust-parity|signed-narrow-cpp-(c|pmr-c)-parity"` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
31. Strengthened directed-path hardening in core C/C++ and C/Rust parity harnesses:
    - added explicit per-scenario directed `INFO` markers covering union-tag, delimiter, array-length, truncation/zero-extension, serialize-buffer, and scalar edge families
    - added CMake required-marker checks in `RunCppCParity.cmake` and `RunCRustParity.cmake` so missing directed scenarios now fail parity gates
32. Revalidated directed-marker hardening:
    - targeted parity gate `ctest --test-dir build/dev-homebrew --output-on-failure -R "uavcan-(cpp-c|cpp-pmr-c|c-rust)-parity"` passed (3/3)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)
33. Extended signed-narrow parity directed-path hardening to C/C++ and C/Rust:
    - added explicit per-scenario directed `INFO` markers to `SignedNarrowCppCParityMain.cpp` and `SignedNarrowCRustParityMain.rs` for saturating/truncating serialize and sign-extension coverage points
    - expanded signed-narrow parity gates (`RunSignedNarrowCppCParity.cmake`, `RunSignedNarrowCRustParity.cmake`) with required-marker assertions so missing directed scenarios fail even when aggregate counts still match
34. Revalidated signed-narrow directed-marker hardening:
    - targeted signed-narrow parity gate `ctest --test-dir build/dev-homebrew --output-on-failure -R "signed-narrow-c-go-parity|signed-narrow-c-rust-parity|signed-narrow-cpp-(c|pmr-c)-parity"` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (20/20)

## Definition Of "Maximally Lowered"

A target is "maximally lowered" when:

1. Serialize/deserialize control flow is derived from lowered MLIR program facts, not backend-local semantic branching.
2. Wire-semantics decisions (alignment, truncation/extension, union tag behavior, array prefix behavior, delimiter validation, buffer checks) are represented in MLIR-lowered contracts.
3. Backend code generators only decide:
   surface type declarations, symbol naming/layout, runtime API binding, and target-language syntax rendering.
4. No backend has semantic fallback logic that can diverge from lowered MLIR intent.

## Current Baseline

1. `lower-dsdl-serialization` and `convert-dsdl-to-emitc` are active.
2. C++/Rust/Go already consume shared MLIR-derived facts/helpers through shared CodeGen modules.
3. Shared planning modules exist (`MlirLoweredFacts`, statement plans, helper symbol resolution, helper-binding plans, array wire plans, descriptor contracts).
4. Strong parity gates exist for C<->C++, C<->Rust, C<->Go (including signed-narrow fixtures and directed vectors).
5. Remaining gap: non-C backends still own portions of control-flow/body shaping that should be represented by a tighter lowered contract.

## End-State Requirements

1. A backend-agnostic lowered SerDes program contract exists and is versioned.
2. Every per-field/per-section step used by backend emission is traceable to lowered MLIR metadata.
3. Backends consume one shared lowering product for body generation decisions.
4. All parity suites assert inventory and pass-line invariants.
5. No semantic fallback patterns remain in backend emitters.

## Convergence Feature Matrix

| Capability | C | C++ std | C++ pmr | Rust std | Go | Target State |
|---|---|---|---|---|---|---|
| MLIR schema + section plan ingestion | Yes | Yes | Yes | Yes | Yes | Keep |
| Shared helper symbol contracts | Yes | Yes | Yes | Yes | Yes | Keep |
| Shared statement/array helper planners | Yes | Yes | Yes | Yes | Yes | Keep |
| Full SerDes body control-flow driven by one lowered contract | High | Partial | Partial | Partial | Partial | Full |
| Backend-local semantic fallbacks eliminated | High | Partial | Partial | Partial | Partial | Full |
| Inventory + pass-line parity invariants | High | Medium | Medium | Medium | High | Full |

## Phases

### Phase 0: Contract Freeze

1. Define and document a canonical lowered SerDes contract schema for backend consumption.
2. Add explicit versioning to lowered contract metadata.
3. Add validator checks that fail when required lowered attributes are missing.

Gate:

1. Contract document merged.
2. Contract validator test suite green.

### Phase 1: Lowered Program Completion

1. Extend lowering so all SerDes-relevant decisions are represented in lowered metadata/program facts.
2. Ensure every union, variable-array, delimiter, and service-section path includes canonical lowered keys.
3. Remove ambiguous fallback defaults in lowering where practical.

Gate:

1. `llvmdsdl-uavcan-mlir-lowering` green.
2. Lit coverage added for each required lowered semantic family.

### Phase 2: Shared Backend Consumption Layer

1. Introduce a single shared "lowered body plan" API consumed by C++, Rust, and Go.
2. Rewire backend emitters to consume this one shared body plan for serialize and deserialize emission.
3. Keep language-specific code limited to syntax/rendering and container API differences.

Gate:

1. C++, Rust, Go emitters no longer branch on ad hoc semantic interpretation for covered families.
2. Unit tests validate identical shared-plan interpretation paths.

### Phase 3: Fallback Elimination

1. Enumerate backend fallback patterns and remove them family-by-family.
2. Add hard-fail guards when required lowered information is absent.
3. Add integration checks that reject known fallback signatures in generated outputs.

Gate:

1. No known fallback signatures remain for scalar/array/union/delimiter/service core paths.
2. All generation integration tests remain green.

### Phase 4: Cross-Language Parity Hardening

1. Ensure every parity suite has:
   summary marker, inventory marker, random pass-line count check, and directed pass-line count check.
2. Expand directed vectors for any newly lowered edge families.
3. Keep signed-narrow fixture parity aligned across C++, Rust, and Go.

Gate:

1. All parity suites enforce the same invariant model.
2. Differential and full workflows green.

### Phase 5: Optimization Pass Readiness

1. Add canonicalization opportunities on lowered SerDes facts where safe.
2. Add optional optimization passes that preserve wire semantics.
3. Validate that optimized output remains parity-equivalent.

Gate:

1. Optimization pass pipeline available and tested.
2. Parity unchanged under optimization-enabled workflows.

## Semantic Family Rollout Order

1. Scalar normalize/read/write
2. Alignment/padding and deterministic zeroing
3. Variable-array prefix/length validation and loop shape
4. Union tag mask/validate/dispatch
5. Delimiter validation for nested composites
6. Service request/response section boundaries
7. Composite recursion and mixed nested error paths

## Testing Strategy

1. Unit tests for each shared lowered-plan module.
2. Lit tests for lowering metadata completeness and verifier constraints.
3. Integration generation checks for C/C++/Rust/Go.
4. Parity tests include `llvmdsdl-uavcan-cpp-c-parity`, `llvmdsdl-uavcan-cpp-pmr-c-parity`, `llvmdsdl-uavcan-c-rust-parity`, `llvmdsdl-uavcan-c-go-parity`, and signed-narrow variants for all supported cross-language pairs.
5. Full-suite workflow gate is `cmake --workflow --preset full`.

## Completion Criteria

1. C++ std/pmr, Rust, and Go body generation decisions are fully driven by shared lowered contracts.
2. Backend semantic fallback logic is removed for core SerDes families.
3. All parity suites use inventory/pass-line invariants and stay green.
4. Full workflow remains green with strict mode.
5. `DESIGN.md` updated to describe final architecture and remaining intentional backend-specific behavior.

## Optional Stretch After Completion

1. Add a language-agnostic intermediate "render IR" generated from lowered MLIR for even thinner backends.
2. Introduce profile knobs (`no_std`, embedded allocators, runtime specialization) without changing lowered semantics.
3. Add first non-C-like target once lowered contract stability is proven.
