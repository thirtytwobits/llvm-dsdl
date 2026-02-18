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

1. Phases 0-5: convergence and optimization-readiness gates are complete for current backend scope, including optimization-enabled parity coverage and full-suite revalidation.
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
35. Started Phase 5 optimization-pass readiness with a dedicated optional MLIR pipeline:
    - added `optimize-dsdl-lowered-serdes` pipeline registration in `registerDSDLPasses()`
    - added shared helper API `addOptimizeLoweredSerDesPipeline(pm)` for emitter and lowering consumers
    - constrained optimization scope to nested `func.func` ops to preserve lowered `dsdl.serialization_plan` contracts
36. Added codegen workflow toggle for optimization-enabled runs:
    - introduced `--optimize-lowered-serdes` in `dsdlc --help`
    - threaded optimization option through C/C++/Rust/Go emit options and shared lowered-fact collection
    - C backend now runs the optional optimization pipeline between `lower-dsdl-serialization` and `convert-dsdl-to-emitc` when enabled
37. Expanded optimization readiness coverage:
    - added lit coverage `test/lit/optimize-dsdl-lowered-serdes-pipeline.mlir`
    - expanded `test/integration/RunDsdlOptSanity.cmake` to validate baseline + optimized lowering/convert pipelines and deterministic repeat runs
38. Revalidated Phase 5 slice:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R llvmdsdl-dsdl-opt-sanity` passed (1/1)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R llvmdsdl-lit` passed (1/1)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-unit-tests|llvmdsdl-signed-narrow-cpp-c-parity-optimized"` passed (2/2)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 24`
    - manual optimization-flag smoke checks passed:
      `build/dev-homebrew/tools/dsdlc/dsdlc c --root-namespace-dir test/lit/fixtures --strict --optimize-lowered-serdes --out-dir build/dev-homebrew/tmp-opt-c-1`
      `build/dev-homebrew/tools/dsdlc/dsdlc cpp --root-namespace-dir test/lit/fixtures --strict --optimize-lowered-serdes --cpp-profile std --out-dir build/dev-homebrew/tmp-opt-cpp-1`
39. Extended optimization-enabled signed-narrow parity coverage to all language pairs:
    - generalized `RunSignedNarrowCRustParity.cmake` and `RunSignedNarrowCGoParity.cmake` with optional `DSDLC_EXTRA_ARGS`
    - registered `llvmdsdl-signed-narrow-c-rust-parity-optimized` and `llvmdsdl-signed-narrow-c-go-parity-optimized` in integration CTest
40. Revalidated optimization-enabled signed-narrow parity family:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-signed-narrow-c-rust-parity-optimized|llvmdsdl-signed-narrow-c-go-parity-optimized"` passed (2/2)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L optimized` passed (3/3) including C/C++, C/Rust, and C/Go optimized parity gates
41. Extended optimization-enabled parity coverage to full `uavcan` cross-language suites:
    - generalized `RunCppCParity.cmake`, `RunCRustParity.cmake`, and `RunCGoParity.cmake` with optional `DSDLC_EXTRA_ARGS`
    - registered optimized `uavcan` parity tests:
      `llvmdsdl-uavcan-c-go-parity-optimized`,
      `llvmdsdl-uavcan-cpp-c-parity-optimized`,
      `llvmdsdl-uavcan-cpp-pmr-c-parity-optimized`,
      `llvmdsdl-uavcan-c-rust-parity-optimized`
42. Revalidated optimization-enabled `uavcan` parity invariants:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-c-go-parity-optimized|llvmdsdl-uavcan-cpp-c-parity-optimized|llvmdsdl-uavcan-cpp-pmr-c-parity-optimized|llvmdsdl-uavcan-c-rust-parity-optimized"` passed (4/4)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 28`
43. Revalidated aggregate optimization-enabled parity family gate:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L optimized` passed (7/7) spanning signed-narrow and full `uavcan` optimized parity suites
44. Revalidated full suite after optimization-enabled parity expansion:
    - `ctest --test-dir build/dev-homebrew --output-on-failure` passed (28/28)
45. Extended full-`uavcan` MLIR lowering integration to include optimized pipeline invariants:
    - `RunUavcanMlirLowering.cmake` now validates baseline + optimized lowering/convert pipelines and optimized-convert determinism checks
46. Extended differential parity integration with optimization-enabled execution:
    - `RunDifferentialParity.cmake` now accepts optional `DSDLC_EXTRA_ARGS`
    - added `llvmdsdl-differential-parity-optimized` in integration CTest registration
47. Revalidated optimized integration expansion:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R llvmdsdl-uavcan-mlir-lowering` passed (1/1)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R llvmdsdl-differential-parity-optimized` passed (1/1)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L optimized` passed (8/8)
48. Revalidated full suite after optimized MLIR/differential integration expansion:
    - `ctest --test-dir build/dev-homebrew --output-on-failure` passed (29/29)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 29`
49. Updated architecture documentation for current lowered-contract + optimization state:
    - refreshed `DESIGN.md` "Current State vs Target State" with lowered-contract enforcement, optional optimization pipeline/CLI flag, and optimization-enabled parity coverage status
    - added explicit "Intentional Backend-Specific Behavior" section to separate rendering/runtime concerns from lowered wire-semantics ownership
50. Added optimization-focused preset/workflow automation:
    - introduced test presets `test-optimized`, `test-optimized-homebrew`, and `test-optimized-llvm-env` in `CMakePresets.json`
    - introduced workflow presets `optimized`, `optimized-homebrew`, and `optimized-llvm-env` in `CMakePresets.json`
    - documented optimized workflow and label-gate invocation in `README.md`
51. Revalidated optimized workflow automation:
    - `cmake --list-presets=all` confirms optimized presets/workflows are registered
    - `cmake --workflow --preset optimized-homebrew` passed end-to-end (8 optimized tests)
52. Revalidated optimized integration hardening after script/preset updates:
    - `ctest --test-dir build/dev-homebrew --output-on-failure` passed (29/29)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L optimized` passed (8/8)
53. Started Optional Stretch item 1 with a shared language-agnostic render-IR layer:
    - added `LoweredRenderIR` (`buildLoweredBodyRenderIR`) to represent backend-agnostic body steps (`field`, `padding`, `union-dispatch`) and helper binding plans from lowered facts
    - rewired C++, Rust, and Go body emitters to consume render-IR steps instead of backend-local section/union traversal branches
    - added unit coverage in `test/unit/LoweredRenderIRTests.cpp` and registered it in the unit test binary
54. Revalidated render-IR convergence slice:
    - `cmake --build build/dev-homebrew -j8` passed
    - `ctest --test-dir build/dev-homebrew --output-on-failure` passed (29/29)
55. Started Optional Stretch item 2 (profile knobs) with Rust `no_std` + `alloc` support:
    - implemented `--rust-profile no-std-alloc` generation path (removed placeholder "not implemented" failure)
    - Rust crate emission now configures Cargo defaults by profile (`std` default for `std`, empty default-feature set for `no-std-alloc`)
    - generated `lib.rs` now uses `#![cfg_attr(not(feature = "std"), no_std)]` with `extern crate alloc` under non-std builds
    - Rust runtime vector alias now switches between `std::vec::Vec` and `alloc::vec::Vec` via feature gating
    - generalized Rust integration scripts to accept `RUST_PROFILE`
    - added `uavcan` no-std integration coverage:
      `llvmdsdl-uavcan-rust-generation-no-std-alloc`,
      `llvmdsdl-uavcan-rust-cargo-check-no-std-alloc`
56. Revalidated Rust profile-knob expansion end-to-end:
    - targeted Rust profile gate:
      `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-rust-generation-no-std-alloc|llvmdsdl-uavcan-rust-cargo-check-no-std-alloc|llvmdsdl-uavcan-rust-generation|llvmdsdl-uavcan-rust-cargo-check"` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (31/31)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 31`
57. Extended Rust `no_std` profile validation to parity-equivalence gates:
    - generalized C/Rust parity harness scripts with explicit `RUST_PROFILE` support and dependency feature pinning (`default-features = false` for no-std lane):
      `RunCRustParity.cmake`, `RunSignedNarrowCRustParity.cmake`,
      `CRustParityCargo.toml.in`, `SignedNarrowCRustParityCargo.toml.in`
    - added no-std parity tests:
      `llvmdsdl-signed-narrow-c-rust-parity-no-std-alloc`,
      `llvmdsdl-uavcan-c-rust-parity-no-std-alloc`
    - added dedicated `rust-no-std` test label lane and automation:
      test presets `test-rust-no-std`, `test-rust-no-std-homebrew`, `test-rust-no-std-llvm-env`
      workflow presets `rust-no-std`, `rust-no-std-homebrew`, `rust-no-std-llvm-env`
58. Revalidated no-std parity automation and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L rust-no-std` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (33/33)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 33`
    - `cmake --list-presets=all` confirms rust-no-std presets/workflows are registered
59. Continued Optional Stretch item 2 with Rust runtime specialization knobs:
    - added CLI/runtime profile plumbing for `--rust-runtime-specialization <portable|fast>`
    - Rust Cargo emission now declares `runtime-fast` feature and includes it in defaults when `fast` is selected
    - Rust runtime now includes feature-gated fast aligned-copy path (`runtime-fast`) while preserving portable fallback semantics
    - added integration runtime-specialization coverage:
      `llvmdsdl-uavcan-rust-generation-runtime-fast`,
      `llvmdsdl-uavcan-rust-cargo-check-runtime-fast`,
      `llvmdsdl-uavcan-rust-runtime-specialization-diff`
60. Added dedicated runtime-specialization lane automation:
    - added `rust-runtime-specialization` test/workflow presets:
      `test-rust-runtime-specialization`, `test-rust-runtime-specialization-homebrew`, `test-rust-runtime-specialization-llvm-env`
      `rust-runtime-specialization`, `rust-runtime-specialization-homebrew`, `rust-runtime-specialization-llvm-env`
    - added `generate-uavcan-rust-runtime-fast` generation target and included it under `generate-uavcan-all`
61. Revalidated runtime-specialization expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L rust-runtime-specialization` passed (3/3)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (37/37)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 37`
62. Extended runtime-specialization coverage to no-std combinations and parity-equivalence gates:
    - generalized C/Rust parity harness scripts with explicit `RUST_RUNTIME_SPECIALIZATION` support and no-std fast dependency feature pinning:
      `RunCRustParity.cmake`, `RunSignedNarrowCRustParity.cmake`
    - added runtime-specialization parity tests:
      `llvmdsdl-signed-narrow-c-rust-parity-runtime-fast`,
      `llvmdsdl-uavcan-c-rust-parity-runtime-fast`,
      `llvmdsdl-uavcan-c-rust-parity-no-std-runtime-fast`
63. Expanded generation/cargo/runtime-specialization diff matrix for no-std + fast:
    - added no-std + fast integration tests:
      `llvmdsdl-uavcan-rust-generation-no-std-runtime-fast`,
      `llvmdsdl-uavcan-rust-cargo-check-no-std-runtime-fast`,
      `llvmdsdl-uavcan-rust-runtime-specialization-diff-no-std-alloc`
64. Revalidated expanded runtime/no-std parity matrix and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L rust-runtime-specialization` passed (9/9)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L rust-no-std` passed (9/9)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (43/43)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 43`
65. Added generation-target automation parity for the new profile matrix:
    - added `generate-uavcan-rust-no-std-runtime-fast`
    - included it under aggregate `generate-uavcan-all`
    - revalidated configure/build registration (`cmake --build build/dev-homebrew -j8`) and test inventory remains stable (`Total Tests: 43`)
66. Started Optional Stretch item 3 with first non-C-like backend scaffolding:
    - added TypeScript emitter (`emitTs`) and CLI command `dsdlc ts`
    - TypeScript backend emits namespace-mirrored declaration files, root `index.ts`, and `package.json`
    - TypeScript emission is gated by lowered-schema validation (`collectLoweredFactsFromMlir`) to preserve MLIR contract checks
67. Added TypeScript integration and generation automation:
    - added `llvmdsdl-uavcan-ts-generation` integration gate (`RunUavcanTsGeneration.cmake`)
    - added generation target `generate-uavcan-ts` and included it under `generate-uavcan-all`
68. Added dedicated TypeScript test/workflow automation:
    - test presets: `test-ts`, `test-ts-homebrew`, `test-ts-llvm-env`
    - workflow presets: `ts`, `ts-homebrew`, `ts-llvm-env`
69. Revalidated non-C-like target bootstrap and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R llvmdsdl-uavcan-ts-generation` passed (1/1)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (44/44)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 44`
70. Added and validated dedicated TypeScript lane ergonomics:
    - `cmake --list-presets=all` confirms `test-ts*` + `ts*` presets/workflows are registered
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (1/1)
    - `cmake --build build/dev-homebrew --target generate-uavcan-ts -j4` passed and generated `build/dev-homebrew/generated/uavcan/ts`
71. Expanded TypeScript integration coverage with compile validation:
    - added `RunUavcanTsTypecheck.cmake` and integration test `llvmdsdl-uavcan-ts-typecheck`
    - `llvmdsdl-uavcan-ts-typecheck` runs `dsdlc ts` then `tsc --noEmit` against generated `uavcan` output
    - gate is enabled conditionally when `tsc` is available in `PATH`
72. Updated TypeScript lane/documentation ergonomics for generation + compile gates:
    - refreshed `test-ts*`/`ts*` preset display metadata to describe generation+typecheck scope
    - updated `README.md`, `DEMO.md`, and `DESIGN.md` with explicit TypeScript compile-gate invocation/coverage notes
73. Revalidated TypeScript compile-gate expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-typecheck"` passed (2/2)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (2/2)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (45/45)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 45` in environments with `tsc` available
74. Hardened TypeScript root-module emission for compile safety:
    - fixed `index.ts` generation to avoid `export *` symbol collisions across large trees
    - root module now emits collision-safe namespace exports (`export * as <alias> from "./path"`)
    - strengthened `RunUavcanTsGeneration.cmake` to reject wildcard root re-exports and require namespace alias exports
    - revalidated `llvmdsdl-uavcan-ts-typecheck`, `-L ts`, and full-suite gates after the fix (all passing)
75. Added TypeScript determinism integration coverage:
    - added `RunUavcanTsDeterminism.cmake` and integration test `llvmdsdl-uavcan-ts-determinism`
    - determinism gate verifies identical generated file inventory and byte-for-byte output across two strict `dsdlc ts` runs
76. Updated TypeScript lane ergonomics/documentation for determinism:
    - refreshed `test-ts*`/`ts*` preset display metadata to describe generation+typecheck+determinism scope
    - updated `README.md`, `DEMO.md`, and `DESIGN.md` to include TypeScript determinism gate usage/coverage
77. Revalidated TypeScript determinism expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-determinism|llvmdsdl-uavcan-ts-typecheck"` passed (3/3)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (3/3)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (46/46)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 46`
78. Added TypeScript consumer-smoke integration coverage:
    - added `RunUavcanTsConsumerSmoke.cmake` and integration test `llvmdsdl-uavcan-ts-consumer-smoke`
    - consumer-smoke gate generates `uavcan` TypeScript output, compiles a tiny consumer module that imports root `index.ts` aliases, and validates end-user import/type usage with `tsc --noEmit`
79. Updated TypeScript lane ergonomics/documentation for consumer smoke:
    - refreshed `test-ts*`/`ts*` preset display metadata to describe generation+typecheck+determinism+consumer-smoke scope
    - updated `README.md`, `DEMO.md`, and `DESIGN.md` with explicit consumer-smoke gate usage/coverage notes
80. Revalidated TypeScript consumer-smoke expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-determinism|llvmdsdl-uavcan-ts-typecheck|llvmdsdl-uavcan-ts-consumer-smoke"` passed (4/4)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (4/4)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (47/47)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 47`
81. Added TypeScript root-index contract integration coverage:
    - added `RunUavcanTsIndexContract.cmake` and integration test `llvmdsdl-uavcan-ts-index-contract`
    - index-contract gate validates root `index.ts` export shape: no wildcard re-exports, alias uniqueness, module-target uniqueness, and one-to-one coverage of all generated type modules
82. Updated TypeScript lane ergonomics/documentation for index contract:
    - refreshed `test-ts*`/`ts*` preset display metadata to describe generation+typecheck+determinism+consumer-smoke+index-contract scope
    - updated `README.md`, `DEMO.md`, and `DESIGN.md` with explicit index-contract usage/coverage notes
83. Revalidated TypeScript index-contract expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-determinism|llvmdsdl-uavcan-ts-index-contract|llvmdsdl-uavcan-ts-typecheck|llvmdsdl-uavcan-ts-consumer-smoke"` passed (5/5)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (5/5)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (48/48)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 48`
84. Added initial TypeScript runtime-backed SerDes emission scaffolding:
    - TypeScript backend now emits generated runtime module `dsdl_runtime.ts`
    - per-type `serialize*` / `deserialize*` helpers are emitted for currently supported lowered section families (fixed-size, non-union scalar paths)
    - unsupported section families now emit explicit runtime stubs that throw at execution time instead of silent behavior divergence
85. Expanded TypeScript integration gates for runtime-backed output shape:
    - strengthened `llvmdsdl-uavcan-ts-generation` checks to require generated `dsdl_runtime.ts`
    - generation gate now verifies at least one generated type module exports runtime `serialize*` / `deserialize*` helper signatures
    - adjusted TypeScript index-contract inventory accounting to exclude generated runtime support module from alias/module-one-to-one checks
86. Added fixture C<->TypeScript runtime parity smoke coverage:
    - added `RunFixturesCTsRuntimeParity.cmake` and integration test `llvmdsdl-fixtures-c-ts-runtime-parity`
    - gate generates C + TypeScript from shared fixture corpus, executes both runtimes for `vendor.Type.1.0`, and asserts serialized bytes plus decode values/consumed-size parity
87. Revalidated TypeScript runtime-slice expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-fixtures-c-ts-runtime-parity|llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-determinism|llvmdsdl-uavcan-ts-index-contract|llvmdsdl-uavcan-ts-typecheck|llvmdsdl-uavcan-ts-consumer-smoke"` passed (6/6)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (6/6)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (49/49)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 49`
88. Updated architecture/demo/status documentation for the TypeScript runtime-slice milestone:
    - refreshed `README.md`, `DEMO.md`, and `DESIGN.md` to reflect generated TypeScript runtime helpers and fixture parity smoke coverage
    - clarified optional-stretch caveat wording: TypeScript runtime work is now scaffolded with fixture parity smoke, with broader semantic-family/runtime parity remaining
89. Extended TypeScript runtime-backed SerDes support to fixed-length scalar arrays:
    - runtime section planner now accepts fixed arrays for bool/unsigned/signed scalar element families
    - generated TypeScript serialize/deserialize helpers now emit deterministic fixed-array loops with strict fixed-length checks
    - variable arrays remain explicit runtime-stub paths pending dedicated lowered-array-prefix/validation parity rollout
90. Added dedicated TypeScript fixed-array runtime smoke coverage:
    - added `RunTsRuntimeFixedArraySmoke.cmake` and integration test `llvmdsdl-ts-runtime-fixed-array-smoke`
    - gate synthesizes a tiny fixed-array DSDL namespace, generates TypeScript, runs runtime-backed serialize/deserialize under Node, and validates byte + decoded-value output
91. Revalidated TypeScript fixed-array runtime expansion and full-suite stability:
    - `ctest --test-dir build/dev-homebrew --output-on-failure -R "llvmdsdl-ts-runtime-fixed-array-smoke|llvmdsdl-fixtures-c-ts-runtime-parity|llvmdsdl-uavcan-ts-generation|llvmdsdl-uavcan-ts-determinism|llvmdsdl-uavcan-ts-index-contract|llvmdsdl-uavcan-ts-typecheck|llvmdsdl-uavcan-ts-consumer-smoke"` passed (7/7)
    - `ctest --test-dir build/dev-homebrew --output-on-failure -L ts` passed (7/7)
    - full suite `ctest --test-dir build/dev-homebrew --output-on-failure` passed (50/50)
    - `ctest --test-dir build/dev-homebrew -N` now reports `Total Tests: 50`
92. Refreshed TypeScript runtime-slice documentation for fixed-array coverage:
    - updated `README.md`, `DEMO.md`, and `DESIGN.md` to include `llvmdsdl-ts-runtime-fixed-array-smoke` and current runtime-support boundaries

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
5. Optional stretch has begun with shared render-IR convergence plus profile-knob expansion (`no_std` and runtime-specialization Rust generation/cargo/parity/diff lanes), and first non-C-like target progression (TypeScript declarations + generated runtime scaffolding + compile/determinism/consumer-smoke/index-contract + fixture runtime parity smoke + fixed-array runtime smoke gates), while preserving parity invariants.

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
3. Integration generation checks for C/C++/Rust/Go/TypeScript.
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

Current progress:

1. Item 1 has started and is now integrated into C++/Rust/Go body emission for shared step traversal.
2. Item 2 has started with Rust `no_std` + `alloc` profile support plus runtime specialization (`portable|fast`), with dedicated generation/cargo/parity/semantic-diff lanes (`rust-no-std` + `rust-runtime-specialization` labels/workflows), including no-std+fast validation.
3. Item 3 has started with an experimental TypeScript backend (`dsdlc ts`) that now includes declaration generation, generated runtime scaffolding, dedicated `uavcan` generation + deterministic-output + `tsc --noEmit` typecheck + consumer-smoke + root-index-contract gating, fixture C<->TS runtime parity smoke plus fixed-array runtime smoke coverage, and standalone `ts` test/workflow lanes.
