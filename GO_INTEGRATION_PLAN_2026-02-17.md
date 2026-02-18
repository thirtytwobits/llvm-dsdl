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

## Phase Completion Matrix (2026-02-18)

Status: **Complete (100%)** for the scope defined in this plan ("same level" Go backend with shared-infra focus and enforced differential gates).

1. Phase 0: Complete
   - `dsdlc go` path implemented and exercised by integration tests/workflows.
2. Phase 1: Complete for scoped shared-core target
   - Shared helper/planning/render seams are used by Go with no regression in full workflows.
3. Phase 2: Complete
   - `runtime/go/dsdl_runtime.go` and runtime unit tests are in the Go differential ring.
4. Phase 3: Complete
   - `GoEmitter` emits strict full-`uavcan` output and deterministic artifacts.
5. Phase 4: Complete
   - `llvmdsdl-uavcan-go-generation`, `llvmdsdl-uavcan-go-build`,
     `llvmdsdl-uavcan-go-determinism`, `llvmdsdl-uavcan-c-go-parity`,
     and signed-narrow C/Go parity are green.
6. Phase 5: Complete
   - `generate-uavcan-go` and workflow docs are in place (`README.md`, `DESIGN.md`).

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

## Status Snapshot (2026-02-17, afternoon)

Completed:
1. Go toolchain installed and validated (`go version` available).
2. `dsdlc go` command wired with module/output options.
3. `generate-uavcan-go` target wired into CMake.
4. Integration tests added and green:
   - `llvmdsdl-uavcan-go-generation`
   - `llvmdsdl-uavcan-go-build`
5. Go runtime (`runtime/go/dsdl_runtime.go`) present with bit/scalar helpers and canonical error constants.
6. Go emitter now generates real SerDes bodies (not stubs), including:
   - capacity checks
   - alignment behavior
   - scalar read/write
   - arrays (prefix + validation)
   - union tag encode/decode/validation
   - delimited/sealed composite handling
7. Shared helper-binding infrastructure extended with Go rendering:
   - `HelperBindingRenderLanguage::Go`
   - Go helper closure generation from shared MLIR helper plans.

In progress:
1. None (phase plan closed; moved to sustained hardening/coverage expansion as routine backlog).

Next:
1. Optional breadth expansion: keep growing C/Go parity case corpus toward even denser full-tree confidence.
2. Optional shared-core convergence: continue lifting backend-local shaping/rendering into shared planner contracts.
3. Optional profile expansion: introduce `no_std + alloc`-style Rust profile and future Go runtime profile knobs.

Completion checkpoint (2026-02-18):
1. `cmake --workflow --preset go-differential` passes.
2. `cmake --workflow --preset go-differential-homebrew` passes.
3. `cmake --workflow --preset go-differential-llvm-env` passes.
4. `cmake --workflow --preset full` passes.
5. Current C/Go parity gate floors are enforced and green at:
   - `random=128`
   - `cases>=109` (observed `109`)
   - `directed>=265` (observed `265`)
   - directed baseline coverage invariants:
     - `any=109`
     - `truncation=109`
     - `serialize_buffer=109`
   - case inventory invariant:
     - `observed cases == DEFINE_ROUNDTRIP wrapper count` (currently `109 == 109`)
   - pass-line execution invariants:
     - random pass lines `== observed cases`
     - directed pass lines (excluding auxiliary float16 marker) `== observed directed`
6. C/Go parity runner concurrency hardening:
   - `RunCGoParity.cmake` and `RunSignedNarrowCGoParity.cmake` now allocate
     unique per-invocation work directories under `OUT_DIR` instead of deleting
     shared roots.
   - Go external linker flags now use direct static archive paths for parity
     harness linking (`lib...a`) to avoid transient library lookup races.
   - Successful parity runs remove their own scratch directories to prevent
     output-root growth while preserving summary artifacts.
   - Runners now also clean legacy flat-layout parity output directories
     (`c`, `go`, `build`, `harness`, and Go cache roots) that may exist from
     pre-hardening versions.
   - Scratch cleanup uses both `file(REMOVE_RECURSE)` and a `cmake -E rm -rf`
     fallback to maximize portability/reliability across environments.
   - Summary artifacts now use atomic replacement (`write tmp -> rename`) in
     both C/Go parity runners.
   - Runner scripts verify static archives exist after `ar` before invoking
     Go/cgo link steps.
7. Signed-narrow parity gate hardening:
   - `SignedNarrowCGoParityMain.go` now emits explicit random/direct pass
     markers for every executed vector plus `PASS signed-narrow inventory ...`.
   - `RunSignedNarrowCGoParity.cmake` now verifies:
     - inventory marker parity with summary counts
     - random pass-line count equals `random_cases`
     - directed pass-line count equals `directed_cases`
8. Cross-parity consistency hardening:
   - Applied the same inventory + pass-line invariants to the remaining
     parity families:
     - `RunCRustParity.cmake`
     - `RunSignedNarrowCRustParity.cmake`
     - `RunCppCParity.cmake` (std + pmr)
     - `RunSignedNarrowCppCParity.cmake` (std + pmr)
   - Updated corresponding harness mains to emit normalized markers:
     - `PASS <case> random (<N> iterations)`
     - `PASS <directed_case> directed`
     - `PASS <suite> inventory ...`
     - `PASS <suite> parity ...`

Recent additions:
1. C/Go differential parity upgraded and passing:
   - `llvmdsdl-uavcan-c-go-parity`
   - now validates randomized byte-level deserialize/serialize parity for:
     - `uavcan.node.Heartbeat.1.0`
     - `uavcan.node.ExecuteCommand.1.3` (request + response)
     - `uavcan.register.Value.1.0`
     - `uavcan.metatransport.can.Frame.0.2`
     - `uavcan.primitive.array.Natural8.1.0`
     - `uavcan.node.port.List.1.0`
2. Directed C/Go parity vectors added and passing for:
   - invalid union tag rejection
   - bad array length prefix rejection
   - bad delimiter-header rejection
   - truncation/zero-extension deserialization paths
3. Signed-narrow fixture C/Go differential parity added and passing:
   - `llvmdsdl-signed-narrow-c-go-parity`
   - randomized roundtrip parity for `vendor.Int3Sat.1.0` and `vendor.Int3Trunc.1.0`
   - directed saturation/truncation/sign-extension vectors
4. `uavcan` C/Go parity corpus expanded with additional representative families:
   - `uavcan.primitive.array.Real16.1.0` (float16 path)
   - `uavcan.file.Path.2.0` (bounded variable byte array)
   - `uavcan.pnp.NodeIDAllocationData.2.0` (composite + fixed array)
   - `uavcan.node.GetInfo.1.0` (response)
   - `uavcan.register.Access.1.0` (request + response)
   - `uavcan.file.Read.1.1` (request + response)
   - `uavcan.diagnostic.Record.1.1` (nested composite + bounded variable array)
   - `uavcan.file.List.0.2` (service request + response, nested delimited composite)
   - `uavcan.time.Synchronization.1.0` (56-bit scalar path)
   - `uavcan.internet.udp.OutgoingPacket.0.2` (multi-array + bitfield alignment path)
   - `uavcan.internet.udp.HandleIncomingPacket.0.2` (service request + empty response)
   - `uavcan.diagnostic.Severity.1.0` (3-bit scalar saturation path)
   - `uavcan.primitive.Empty.1.0` (zero-size message path)
   - `uavcan.primitive.String.1.0` (bounded var byte-array path)
   - `uavcan.primitive.Unstructured.1.0` (bounded var byte-array path)
   - `uavcan.primitive.array.Bit.1.0` (bit-packed variable bool-array path)
   - `uavcan.primitive.array.Integer8.1.0` (signed variable array path)
   - `uavcan.primitive.array.Natural32.1.0` (unsigned variable array path)
   - `uavcan.primitive.array.Real64.1.0` (float64 variable array path)
   - `uavcan.primitive.scalar.Bit.1.0` (single-bit scalar path)
   - `uavcan.primitive.scalar.Integer8.1.0` (signed scalar saturation/sign-extension path)
   - `uavcan.primitive.scalar.Natural32.1.0` (unsigned scalar width/masking path)
   - `uavcan.primitive.scalar.Real64.1.0` (float64 scalar path)
   - `uavcan.si.unit.angle.Quaternion.1.0` (fixed float32 array)
   - `uavcan.si.unit.length.WideVector3.1.0` (fixed float64 array)
   - `uavcan.si.sample.angle.Quaternion.1.0` (nested composite + fixed float32 array)
5. Go differential ring workflow hooks added:
   - CTest label: `go-differential`
   - Presets: `test-go-differential`, `test-go-differential-homebrew`, `test-go-differential-llvm-env`
   - Workflows: `go-differential`, `go-differential-homebrew`, `go-differential-llvm-env`
   - Ring now includes generation/build + fixture/uavcan parity tests.
6. Parity policy hardened:
   - strict byte parity by default
   - float-heavy/float-union cases use rc/consumed/size parity (with explicit directed float vectors) to avoid false negatives from NaN/canonicalization differences.
7. Determinism integration added:
   - `llvmdsdl-uavcan-go-determinism` verifies two `dsdlc go` runs produce identical file lists and byte-identical file contents.
8. Additional directed vectors added and passing:
   - `udp_outgoing_packet_bad_destination_address_length`
   - `udp_handle_incoming_request_bad_payload_length`
   - `file_list_response_large_delimiter_header_truncation`
   - `diagnostic_record_truncated_input`
   - `time_synchronization_truncated_input`
   - `si_unit_angle_quaternion_truncated_input`
   - `si_unit_length_wide_vector3_truncated_input`
   - `si_sample_angle_quaternion_truncated_input`
   - `scalar_bit_high_bits_input`
   - `scalar_integer8_truncated_input`
   - `scalar_natural32_truncated_input`
   - `scalar_real64_nan_payload`
   - `array_integer8_bad_length_prefix`
   - `array_natural32_bad_length_prefix`
   - `array_real64_bad_length_prefix`
9. Shared helper-render convergence hardening:
   - `test/unit/HelperBindingRenderTests.cpp` now includes explicit Go assertions for
     - union tag validation rendering
     - signed scalar deserialize helper rendering
     - section-level helper binding rendering (capacity/union/delimiter/array)
   - unit test gate remains green (`llvmdsdl-unit-tests`).
10. Go runtime unit-test gate added and wired into Go differential ring:
   - `runtime/go/dsdl_runtime_test.go` covers bit copy, saturation, signed extension, and float helpers.
   - Includes deterministic randomized bit-copy property checks against a reference implementation.
   - `llvmdsdl-go-runtime-unit-tests` executes `go test ./...` in `runtime/go`.
11. Differential ring scale-up status:
   - `llvmdsdl-uavcan-c-go-parity` currently reports `cases=101` and `directed=180` (randomized + directed vectors).
12. Workflow coverage validation:
   - `go-differential`, `go-differential-homebrew`, and `go-differential-llvm-env` workflows have been executed end-to-end and are green.
13. Shared backend-convergence refactor:
   - Saturation bound math is centralized in `TypeStorage`:
     - `resolveUnsignedSaturationMax(bitLength)`
     - `resolveSignedSaturationRange(bitLength)`
   - Go/Rust/C++ emitters now use these shared helpers for scalar saturation fallback logic.
   - Unit coverage added in `test/unit/TypeStorageTests.cpp`.
14. Primitive family parity expansion:
   - Added full remaining `uavcan.primitive` scalar/array width families to C/Go parity:
     - arrays: `Integer16/32/64`, `Natural16/64`, `Real32`
     - scalars: `Integer16/32/64`, `Natural8/16/64`, `Real16/32`
   - Added directed vectors for new prefix-validation and float-NaN paths:
     - `real32_bad_length_prefix`
     - `scalar_real16_nan_payload`, `scalar_real32_nan_payload`
     - `array_integer16/32/64_bad_length_prefix`
     - `array_natural16/64_bad_length_prefix`
15. Link-interface dedup cleanup:
   - Reduced redundant CMake link interfaces in `Frontend`, `Semantics`, `CodeGen`, and `dsdl-opt` target wiring.
   - Go differential workflow remains green after link cleanup.
16. Differential gate hardening:
  - `test/integration/RunCGoParity.cmake` now enforces parity corpus floor checks by parsing harness summary output.
  - Current enforced minimums:
    - `cases >= 101`
    - `directed >= 180`
   - This prevents accidental shrinkage of differential coverage in future edits.
17. Directed-vector expansion for truncation semantics:
   - Added explicit truncation vectors for widened scalar families:
     - `scalar_integer16/32/64_truncated_input`
     - `scalar_natural16/64_truncated_input`
     - `scalar_real16/32/64_truncated_input`
   - Added array truncation vectors:
     - `array_integer16_truncated_payload`
     - `array_integer64_truncated_payload`
     - `array_natural64_truncated_payload`
18. Serialization error-path parity added:
   - Directed vectors now explicitly validate `SERIALIZATION_BUFFER_TOO_SMALL` parity between C and Go for fixed and variable-size payloads:
     - `heartbeat_serialize_small_buffer`
     - `scalar_real64_serialize_small_buffer`
     - `file_path_serialize_small_buffer`
   - Includes explicit zero-capacity success check for zero-sized types:
     - `primitive_empty_serialize_zero_buffer`
19. Required-vector marker gate added:
   - `test/integration/RunCGoParity.cmake` now asserts the presence of key directed markers in harness output, not just aggregate counts.
   - Current required markers include:
     - `register_value_invalid_union_tag`
     - `port_list_bad_delimiter_header`
     - `scalar_real64_nan_payload`
     - `heartbeat_serialize_small_buffer`
     - `file_path_serialize_small_buffer`
20. Family-level coverage gate added:
   - `test/integration/RunCGoParity.cmake` now enforces required random-case markers to preserve semantic breadth, not just totals.
   - Current required random families include:
     - message/service request/response (`heartbeat`, `execute_command_request`, `execute_command_response`, `file_read_response`)
     - union/delimited heavy representatives (`register_value`, `node_port_list`)
     - primitive scalar/array representatives (`scalar_real64`, `array_integer64`, `primitive_empty`)
     - variable-length composite representative (`file_path`)
   - Full Go differential workflows remain green with this stricter policy.
21. Machine-parsed category accounting added:
   - `test/integration/CGoParityMain.go` now emits stable category summary lines:
     - `PASS c/go parity random categories ...`
     - `PASS c/go parity directed categories ...`
   - `test/integration/RunCGoParity.cmake` parses category counts directly (`key=value`) and enforces per-category minimums.
   - Current random category minima:
     - `message_section >= 10`
     - `service_section >= 8`
     - `union_delimited >= 3`
     - `variable_composite >= 4`
     - `primitive_scalar >= 10`
     - `primitive_array >= 10`
     - `primitive_composite >= 3`
   - Current directed category minima:
     - `union_tag_error >= 3`
     - `delimiter_error >= 2`
     - `length_prefix_error >= 15`
     - `truncation >= 32`
     - `float_nan >= 4`
     - `serialize_buffer >= 8`
     - `high_bits_normalization >= 3`
   - This reduces brittleness versus pure marker checks while preserving strong semantic coverage guarantees.
22. Directed truncation/error-path expansion and gate tightening:
   - Added additional true truncation vectors to `test/integration/CGoParityMain.go`:
     - `scalar_natural8_truncated_input`
     - `natural8_truncated_payload`
     - `array_integer32_truncated_payload`
     - `array_natural16_truncated_payload`
     - `array_natural32_truncated_payload`
     - `array_real64_truncated_payload`
   - Corrected `scalar_integer8_truncated_input` to be a true truncation case (`[]` input) instead of full-width input.
   - `test/integration/RunCGoParity.cmake` regression floors increased:
     - `directed >= 70` (was 62)
     - directed `truncation` category minimum `>= 32` (was 27)
   - Required directed markers now also pin new truncation vectors:
     - `scalar_natural8_truncated_input`
     - `array_integer32_truncated_payload`
     - `array_natural32_truncated_payload`
   - Current observed parity summary:
     - `cases=55`
     - `directed=70`
     - directed categories include `truncation=33`.
23. Workflow revalidation after gate hardening:
   - `cmake --workflow --preset go-differential` re-run is green with the tightened parity policy.
   - `cmake --workflow --preset go-differential-homebrew` re-run is green.
   - `cmake --workflow --preset go-differential-llvm-env` re-run is green.
24. Additional parity-gate hardening:
   - `test/integration/RunCGoParity.cmake` now enforces random-iteration floor:
     - `random >= 128`
   - Added uncategorized-vector guard:
     - fail if directed category summary reports `misc > 0`.
   - `go-differential` workflow remains green after these checks.
25. Go runtime unit-test depth increase:
   - `runtime/go/dsdl_runtime_test.go` now also covers:
     - `SetBit` buffer-too-small error behavior
     - sign extension checks for `GetI16/GetI32/GetI64`
     - out-of-range zero-extension for `GetU16/GetU64`
   - Runtime gate `llvmdsdl-go-runtime-unit-tests` remains green.
26. Signed-narrow C/Go parity gate hardening:
   - `test/integration/SignedNarrowCGoParityMain.go` now emits a machine-parseable summary line:
     - `PASS signed-narrow-c-go-parity random_iterations=... random_cases=... directed_cases=...`
   - `test/integration/RunSignedNarrowCGoParity.cmake` now enforces:
     - `random_iterations >= 256`
     - `random_cases >= 2`
     - `directed_cases >= 12`
     - required marker presence for both random cases and directed checks.
   - Directed vectors now include truncation and serialize-buffer error parity:
     - `int3sat_truncated_input`
     - `int3trunc_truncated_input`
     - `int3sat_serialize_small_buffer`
     - `int3trunc_serialize_small_buffer`
   - Revalidated workflows after this hardening:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env` (all green).
27. Signed-narrow category-accounting convergence:
   - `test/integration/SignedNarrowCGoParityMain.go` now emits machine-parsed directed category summary:
     - `PASS signed-narrow-c-go-parity directed categories ...`
   - `test/integration/RunSignedNarrowCGoParity.cmake` now parses and enforces category minima:
     - `saturation_sign_extension >= 8`
     - `truncation >= 2`
     - `serialize_buffer >= 2`
   - This aligns signed-narrow gate style with the broader `uavcan` C/Go parity gate strategy.
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
28. Shared planner/symbol-resolver edge-path hardening:
   - Expanded `test/unit/ArrayWirePlanTests.cpp` with additional coverage for:
     - variable arrays without lowered facts (prefix preserved, descriptor absent)
     - prefix override propagation into descriptor metadata.
   - Expanded `test/unit/HelperSymbolResolverTests.cpp` with null/edge-path checks:
     - null section facts -> empty helper symbols
     - scalar helper resolution with missing lowered facts -> empty
     - fixed-array length helper descriptor rejection
     - sealed-composite/no-facts delimiter helper rejection.
   - This strengthens shared code paths used by Go/C++/Rust emitters.
   - Validation:
     - `llvmdsdl-unit-tests` green
     - `go-differential`, `go-differential-homebrew`, `go-differential-llvm-env` green.
29. `uavcan` C/Go parity corpus expansion (random families):
   - Added new parity roundtrip families to `test/integration/CGoParityCHarness.c` and `test/integration/CGoParityMain.go`:
     - `uavcan.node.ID.1.0`
     - `uavcan.node.Mode.1.0`
     - `uavcan.register.Name.1.0`
     - `uavcan.metatransport.udp.Endpoint.0.1`
   - Increased parity corpus floor in `test/integration/RunCGoParity.cmake`:
     - `cases >= 55` (was 51)
   - Current observed summary after expansion:
     - `random=128`
     - `cases=55`
     - `directed=70`
     - random categories now include:
       - `message_section=11`
       - `variable_composite=4`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
30. Random-marker anti-regression gate reinforcement:
   - `test/integration/RunCGoParity.cmake` now enforces required random markers for key families, including new additions:
     - `heartbeat`
     - `execute_command_request`
     - `register_value`
     - `node_id`
     - `node_mode`
     - `register_name`
     - `metatransport_udp_endpoint`
     - `node_port_list`
   - This complements category-count minima with explicit case-presence guarantees.
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
31. Directed-vector expansion for newly-added parity families:
   - Added directed vectors in `test/integration/CGoParityMain.go` for:
     - truncation:
       - `node_id_truncated_input`
       - `node_mode_truncated_input`
       - `register_name_truncated_input`
       - `metatransport_udp_endpoint_truncated_input`
     - serialize buffer errors:
       - `node_id_serialize_small_buffer`
       - `node_mode_serialize_small_buffer`
       - `register_name_serialize_small_buffer`
       - `metatransport_udp_endpoint_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `directed >= 70` (was 62)
     - directed `truncation >= 32` (was 27)
     - directed `serialize_buffer >= 8` (was 4)
   - Added required directed markers for these new vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=55`
     - `directed=70`
     - directed categories include:
       - `truncation=33`
       - `serialize_buffer=8`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
32. Random-category minima tightened after corpus expansion:
   - `test/integration/RunCGoParity.cmake` random category floors raised to lock in broader message/composite breadth:
     - `message_section >= 10` (was 5)
     - `variable_composite >= 4` (was 3)
   - Existing random/service/primitive/union category floors remain enforced.
  - Revalidated workflows remain green:
    - `go-differential`
    - `go-differential-homebrew`
    - `go-differential-llvm-env`.
33. Service-family expansion with `uavcan.register.List.1.0` parity:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_register_list_request_roundtrip`
     - `c_register_list_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `register_list_request`
     - `register_list_response`
   - Added directed vectors:
     - truncation:
       - `register_list_request_truncated_input`
       - `register_list_response_truncated_input`
     - serialize buffer errors:
       - `register_list_request_serialize_small_buffer`
       - `register_list_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 57` (was 55)
     - `directed >= 74` (was 70)
     - `service_section >= 10` (was 8)
   - Added required marker checks for the new service family and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=57`
     - `directed=74`
     - random categories include:
       - `service_section=12`
     - directed categories include:
       - `serialize_buffer=10`
       - `truncation=35`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
34. Service-family expansion with `uavcan.file.Write.1.1` parity:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_file_write_request_roundtrip`
     - `c_file_write_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `file_write_request`
     - `file_write_response`
   - Added directed vectors:
     - truncation:
       - `file_write_request_truncated_input`
       - `file_write_response_truncated_input`
     - serialize buffer errors:
       - `file_write_request_serialize_small_buffer`
       - `file_write_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 59` (was 57)
     - `directed >= 78` (was 74)
     - `service_section >= 12` (was 10)
     - directed `truncation >= 34` (was 32)
     - directed `serialize_buffer >= 10` (was 8)
   - Added required marker checks for the new service family and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=59`
     - `directed=78`
     - random categories include:
       - `service_section=14`
     - directed categories include:
       - `serialize_buffer=12`
       - `truncation=37`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
35. Service-family expansion with `uavcan.file.Modify.1.1` parity:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_file_modify_request_roundtrip`
     - `c_file_modify_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `file_modify_request`
     - `file_modify_response`
   - Added directed vectors:
     - truncation:
       - `file_modify_request_truncated_input`
       - `file_modify_response_truncated_input`
     - serialize buffer errors:
       - `file_modify_request_serialize_small_buffer`
       - `file_modify_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 61` (was 59)
     - `directed >= 82` (was 78)
     - `service_section >= 14` (was 12)
     - directed `truncation >= 36` (was 34)
     - directed `serialize_buffer >= 12` (was 10)
   - Added required marker checks for the new service family and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=61`
     - `directed=82`
     - random categories include:
       - `service_section=16`
     - directed categories include:
       - `serialize_buffer=14`
       - `truncation=39`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
36. Service-family expansion with `uavcan.file.GetInfo.0.2` parity:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_file_get_info_request_roundtrip`
     - `c_file_get_info_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `file_get_info_request`
     - `file_get_info_response`
   - Added directed vectors:
     - truncation:
       - `file_get_info_request_truncated_input`
       - `file_get_info_response_truncated_input`
     - serialize buffer errors:
       - `file_get_info_request_serialize_small_buffer`
       - `file_get_info_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 63` (was 61)
     - `directed >= 86` (was 82)
     - `service_section >= 18` (was 14)
     - directed `truncation >= 40` (was 36)
     - directed `serialize_buffer >= 16` (was 12)
   - Added required marker checks for the new service family and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=63`
     - `directed=86`
     - random categories include:
       - `service_section=18`
     - directed categories include:
       - `serialize_buffer=16`
       - `truncation=41`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
37. Service-family expansion with `uavcan.node.GetTransportStatistics.0.1` parity:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_get_transport_statistics_request_roundtrip`
     - `c_get_transport_statistics_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `get_transport_statistics_request`
     - `get_transport_statistics_response`
   - Added directed vectors:
     - length-prefix validation:
       - `get_transport_statistics_response_bad_length_prefix`
     - truncation:
       - `get_transport_statistics_request_truncated_input`
       - `get_transport_statistics_response_truncated_input`
     - serialize buffer paths:
       - `get_transport_statistics_request_serialize_zero_buffer`
       - `get_transport_statistics_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 65` (was 63)
     - `directed >= 91` (was 86)
     - `service_section >= 20` (was 18)
     - directed `truncation >= 43` (was 40)
     - directed `serialize_buffer >= 18` (was 16)
   - Added required marker checks for the new service family and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=65`
     - `directed=91`
     - random categories include:
       - `service_section=20`
     - directed categories include:
       - `length_prefix_error=19`
       - `serialize_buffer=18`
       - `truncation=43`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
38. Message-family expansion with additional `uavcan.node.*` primitives/composites:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_node_version_roundtrip`
     - `c_node_health_roundtrip`
     - `c_node_io_statistics_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `node_version`
     - `node_health`
     - `node_io_statistics`
   - Added directed vectors:
     - high-bits normalization:
       - `node_health_high_bits_input`
     - truncation:
       - `node_version_truncated_input`
       - `node_health_truncated_input`
       - `node_io_statistics_truncated_input`
     - serialize buffer errors:
       - `node_version_serialize_small_buffer`
       - `node_health_serialize_small_buffer`
       - `node_io_statistics_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 68` (was 65)
     - `directed >= 98` (was 91)
     - `message_section >= 14` (was 10)
     - directed `truncation >= 46` (was 43)
     - directed `serialize_buffer >= 21` (was 18)
     - directed `high_bits_normalization >= 4` (was 3)
   - Added required marker checks for these new message families and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=68`
     - `directed=98`
     - random categories include:
       - `message_section=14`
       - `service_section=20`
     - directed categories include:
       - `high_bits_normalization=4`
       - `serialize_buffer=21`
       - `truncation=46`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
39. Time-family expansion with additional `uavcan.time.*` messages/services:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_time_synchronized_timestamp_roundtrip`
     - `c_time_system_roundtrip`
     - `c_time_tai_info_roundtrip`
     - `c_time_get_sync_master_info_request_roundtrip`
     - `c_time_get_sync_master_info_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `time_synchronized_timestamp`
     - `time_system`
     - `time_tai_info`
     - `get_sync_master_info_request`
     - `get_sync_master_info_response`
   - Added directed vectors:
     - high-bits normalization:
       - `time_system_high_bits_input`
       - `time_tai_info_high_bits_input`
     - truncation:
       - `time_synchronized_timestamp_truncated_input`
       - `time_system_truncated_input`
       - `time_tai_info_truncated_input`
       - `get_sync_master_info_request_truncated_input`
       - `get_sync_master_info_response_truncated_input`
     - serialize buffer paths:
       - `time_synchronized_timestamp_serialize_small_buffer`
       - `time_system_serialize_small_buffer`
       - `time_tai_info_serialize_small_buffer`
       - `get_sync_master_info_request_serialize_zero_buffer`
       - `get_sync_master_info_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 73` (was 68)
     - `directed >= 110` (was 98)
     - `message_section >= 17` (was 14)
     - `service_section >= 22` (was 20)
     - directed `truncation >= 51` (was 46)
     - directed `serialize_buffer >= 26` (was 21)
     - directed `high_bits_normalization >= 5` (was 4)
   - Added required marker checks for these new time families and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=73`
     - `directed=110`
     - random categories include:
       - `message_section=17`
       - `service_section=22`
     - directed categories include:
       - `high_bits_normalization=6`
       - `serialize_buffer=26`
       - `truncation=51`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
40. CAN-family expansion with additional `uavcan.metatransport.can.*` messages:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_can_data_classic_roundtrip`
     - `c_can_error_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `can_data_classic`
     - `can_error`
   - Added directed vectors:
     - length-prefix validation:
       - `can_data_classic_bad_length_prefix`
     - truncation:
       - `can_data_classic_truncated_input`
       - `can_error_truncated_input`
     - serialize buffer errors:
       - `can_data_classic_serialize_small_buffer`
       - `can_error_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 75` (was 73)
     - `directed >= 115` (was 110)
     - `message_section >= 19` (was 17)
     - directed `length_prefix_error >= 20` (was 15)
     - directed `truncation >= 53` (was 51)
     - directed `serialize_buffer >= 28` (was 26)
   - Added required marker checks for these CAN families and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=75`
     - `directed=115`
     - random categories include:
       - `message_section=19`
       - `service_section=22`
     - directed categories include:
       - `length_prefix_error=20`
       - `serialize_buffer=28`
       - `truncation=53`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
41. PnP-cluster parity expansion with `uavcan.pnp.cluster.*` services/messages:
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_pnp_cluster_entry_roundtrip`
     - `c_pnp_cluster_append_entries_request_roundtrip`
     - `c_pnp_cluster_append_entries_response_roundtrip`
     - `c_pnp_cluster_request_vote_request_roundtrip`
     - `c_pnp_cluster_request_vote_response_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `pnp_cluster_entry`
     - `pnp_cluster_append_entries_request`
     - `pnp_cluster_append_entries_response`
     - `pnp_cluster_request_vote_request`
     - `pnp_cluster_request_vote_response`
   - Added directed vectors:
     - length-prefix validation:
       - `pnp_cluster_append_entries_request_bad_length_prefix`
     - truncation:
       - `pnp_cluster_entry_truncated_input`
       - `pnp_cluster_append_entries_request_truncated_input`
       - `pnp_cluster_append_entries_response_truncated_input`
       - `pnp_cluster_request_vote_request_truncated_input`
       - `pnp_cluster_request_vote_response_truncated_input`
     - serialize buffer errors:
       - `pnp_cluster_entry_serialize_small_buffer`
       - `pnp_cluster_append_entries_request_serialize_small_buffer`
       - `pnp_cluster_append_entries_response_serialize_small_buffer`
       - `pnp_cluster_request_vote_request_serialize_small_buffer`
       - `pnp_cluster_request_vote_response_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 80` (was 75)
     - `directed >= 126` (was 115)
     - `message_section >= 20` (was 19)
     - `service_section >= 26` (was 22)
     - directed `length_prefix_error >= 21` (was 20)
     - directed `truncation >= 58` (was 53)
     - directed `serialize_buffer >= 33` (was 28)
   - Added required marker checks for new random and directed PnP-cluster vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=80`
     - `directed=126`
     - random categories include:
       - `message_section=20`
       - `service_section=26`
     - directed categories include:
       - `length_prefix_error=21`
       - `serialize_buffer=33`
       - `truncation=58`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
42. Metatransport frame-family parity expansion (`serial`/`ethernet`/`udp`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_metatransport_serial_fragment_roundtrip`
     - `c_metatransport_ethernet_frame_roundtrip`
     - `c_metatransport_udp_frame_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `metatransport_serial_fragment`
     - `metatransport_ethernet_frame`
     - `metatransport_udp_frame`
   - Added directed vectors:
     - length-prefix validation:
       - `metatransport_serial_fragment_bad_length_prefix`
       - `metatransport_ethernet_frame_bad_length_prefix`
     - truncation:
       - `metatransport_serial_fragment_truncated_input`
       - `metatransport_ethernet_frame_truncated_input`
       - `metatransport_udp_frame_truncated_input`
     - serialize buffer errors:
       - `metatransport_serial_fragment_serialize_small_buffer`
       - `metatransport_ethernet_frame_serialize_small_buffer`
       - `metatransport_udp_frame_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 83` (was 80)
     - `directed >= 134` (was 126)
     - `message_section >= 23` (was 20)
     - directed `length_prefix_error >= 23` (was 21)
     - directed `truncation >= 61` (was 58)
     - directed `serialize_buffer >= 36` (was 33)
   - Added required marker checks for new metatransport random and directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=83`
     - `directed=134`
     - random categories include:
       - `message_section=23`
       - `service_section=26`
     - directed categories include:
       - `length_prefix_error=23`
       - `serialize_buffer=36`
       - `truncation=61`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
43. CAN-manifestation union-family parity expansion (`DataFD`/`RTR`/`Manifestation`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_can_data_fd_roundtrip`
     - `c_can_rtr_roundtrip`
     - `c_can_manifestation_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `can_data_fd`
     - `can_rtr`
     - `can_manifestation`
   - Added directed vectors:
     - union-tag validation:
       - `can_manifestation_invalid_union_tag`
     - length-prefix validation:
       - `can_data_fd_bad_length_prefix`
     - truncation:
       - `can_data_fd_truncated_input`
       - `can_rtr_truncated_input`
       - `can_manifestation_truncated_input`
     - serialize buffer errors:
       - `can_data_fd_serialize_small_buffer`
       - `can_rtr_serialize_small_buffer`
       - `can_manifestation_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 86` (was 83)
     - `directed >= 142` (was 134)
     - `message_section >= 25` (was 23)
     - `union_delimited >= 4` (was 3)
     - directed `union_tag_error >= 4` (was 3)
     - directed `length_prefix_error >= 24` (was 23)
     - directed `truncation >= 64` (was 61)
     - directed `serialize_buffer >= 39` (was 36)
   - Added required marker checks for new random and directed CAN vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=86`
     - `directed=142`
     - random categories include:
       - `message_section=25`
       - `union_delimited=4`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=4`
       - `length_prefix_error=24`
       - `serialize_buffer=39`
       - `truncation=64`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
44. PnP discovery message parity expansion (`uavcan.pnp.cluster.Discovery.1.0`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_pnp_cluster_discovery_roundtrip`
   - Added Go parity case in `test/integration/CGoParityMain.go`:
     - `pnp_cluster_discovery`
   - Added directed vectors:
     - length-prefix validation:
       - `pnp_cluster_discovery_bad_length_prefix`
     - truncation:
       - `pnp_cluster_discovery_truncated_input`
     - serialize buffer errors:
       - `pnp_cluster_discovery_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 87` (was 86)
     - `directed >= 145` (was 142)
     - `variable_composite >= 5` (was 4)
     - directed `length_prefix_error >= 25` (was 24)
     - directed `truncation >= 65` (was 64)
     - directed `serialize_buffer >= 40` (was 39)
   - Added required marker checks for new random and directed discovery vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=87`
     - `directed=145`
     - random categories include:
       - `message_section=25`
       - `union_delimited=4`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=4`
       - `length_prefix_error=25`
       - `serialize_buffer=40`
       - `truncation=65`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
45. CAN arbitration-ID family parity expansion (`ArbitrationID`/`BaseArbitrationID`/`ExtendedArbitrationID`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_can_arbitration_id_roundtrip`
     - `c_can_base_arbitration_id_roundtrip`
     - `c_can_extended_arbitration_id_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `can_arbitration_id`
     - `can_base_arbitration_id`
     - `can_extended_arbitration_id`
   - Added directed vectors:
     - union-tag validation:
       - `can_arbitration_id_invalid_union_tag`
     - high-bits normalization:
       - `can_base_arbitration_id_high_bits_input`
       - `can_extended_arbitration_id_high_bits_input`
     - truncation:
       - `can_arbitration_id_truncated_input`
       - `can_base_arbitration_id_truncated_input`
       - `can_extended_arbitration_id_truncated_input`
     - serialize buffer errors:
       - `can_arbitration_id_serialize_small_buffer`
       - `can_base_arbitration_id_serialize_small_buffer`
       - `can_extended_arbitration_id_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 90` (was 87)
     - `directed >= 154` (was 145)
     - `message_section >= 27` (was 25)
     - `union_delimited >= 5` (was 4)
     - directed `union_tag_error >= 5` (was 4)
     - directed `high_bits_normalization >= 8` (was 5)
     - directed `truncation >= 68` (was 65)
     - directed `serialize_buffer >= 43` (was 40)
   - Added required marker checks for random + directed arbitration vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=90`
     - `directed=154`
     - random categories include:
       - `message_section=27`
       - `union_delimited=5`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=5`
       - `high_bits_normalization=8`
       - `length_prefix_error=25`
       - `serialize_buffer=43`
       - `truncation=68`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
46. Node-port + ethernet parity expansion (`ServiceID`/`SubjectID`/`ServiceIDList`/`SubjectIDList` + `EtherType`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_node_port_service_id_roundtrip`
     - `c_node_port_subject_id_roundtrip`
     - `c_node_port_service_id_list_roundtrip`
     - `c_node_port_subject_id_list_roundtrip`
     - `c_metatransport_ethernet_ethertype_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `node_port_service_id`
     - `node_port_subject_id`
     - `node_port_service_id_list`
     - `node_port_subject_id_list`
     - `metatransport_ethernet_ethertype`
   - Added directed vectors:
     - union-tag validation:
       - `node_port_subject_id_list_invalid_union_tag`
     - high-bits normalization:
       - `node_port_service_id_high_bits_input`
       - `node_port_subject_id_high_bits_input`
     - truncation:
       - `node_port_service_id_truncated_input`
       - `node_port_subject_id_truncated_input`
       - `node_port_service_id_list_truncated_input`
       - `node_port_subject_id_list_truncated_input`
       - `metatransport_ethernet_ethertype_truncated_input`
     - serialize buffer errors:
       - `node_port_service_id_serialize_small_buffer`
       - `node_port_subject_id_serialize_small_buffer`
       - `node_port_service_id_list_serialize_small_buffer`
       - `node_port_subject_id_list_serialize_small_buffer`
       - `metatransport_ethernet_ethertype_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 95` (was 90)
     - `directed >= 167` (was 154)
     - `message_section >= 31` (was 27)
     - `union_delimited >= 6` (was 5)
     - directed `union_tag_error >= 6` (was 5)
     - directed `high_bits_normalization >= 10` (was 8)
     - directed `truncation >= 73` (was 68)
     - directed `serialize_buffer >= 48` (was 43)
   - Added required marker checks for new random + directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=95`
     - `directed=167`
     - random categories include:
       - `message_section=31`
       - `union_delimited=6`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=6`
       - `high_bits_normalization=10`
       - `length_prefix_error=25`
       - `serialize_buffer=48`
       - `truncation=73`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
47. Port-ID union + file-error parity expansion (`uavcan.node.port.ID.1.0`, `uavcan.file.Error.1.0`):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_node_port_id_roundtrip`
     - `c_file_error_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `node_port_id`
     - `file_error`
   - Added directed vectors:
     - union-tag validation:
       - `node_port_id_invalid_union_tag`
     - truncation:
       - `node_port_id_truncated_input`
       - `file_error_truncated_input`
     - serialize buffer errors:
       - `node_port_id_serialize_small_buffer`
       - `file_error_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 97` (was 95)
     - `directed >= 172` (was 167)
     - `message_section >= 32` (was 31)
     - `union_delimited >= 7` (was 6)
     - directed `union_tag_error >= 7` (was 6)
     - directed `truncation >= 75` (was 73)
     - directed `serialize_buffer >= 50` (was 48)
   - Added required marker checks for random + directed vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=97`
     - `directed=172`
     - random categories include:
       - `message_section=32`
       - `union_delimited=7`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=7`
       - `high_bits_normalization=10`
       - `length_prefix_error=25`
       - `serialize_buffer=50`
       - `truncation=75`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
48. SI velocity/temperature parity expansion (`unit/sample`, scalar/vector):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_si_unit_velocity_vector3_roundtrip`
     - `c_si_sample_velocity_vector3_roundtrip`
     - `c_si_unit_temperature_scalar_roundtrip`
     - `c_si_sample_temperature_scalar_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `si_unit_velocity_vector3`
     - `si_sample_velocity_vector3`
     - `si_unit_temperature_scalar`
     - `si_sample_temperature_scalar`
   - Added directed vectors:
     - truncation:
       - `si_unit_velocity_vector3_truncated_input`
       - `si_sample_velocity_vector3_truncated_input`
       - `si_unit_temperature_scalar_truncated_input`
       - `si_sample_temperature_scalar_truncated_input`
     - serialize buffer errors:
       - `si_unit_velocity_vector3_serialize_small_buffer`
       - `si_sample_velocity_vector3_serialize_small_buffer`
       - `si_unit_temperature_scalar_serialize_small_buffer`
       - `si_sample_temperature_scalar_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 101` (was 97)
     - `directed >= 180` (was 172)
     - `message_section >= 36` (was 32)
     - directed `truncation >= 79` (was 75)
     - directed `serialize_buffer >= 54` (was 50)
   - Added required marker checks for random + directed SI vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=101`
     - `directed=180`
     - random categories include:
       - `message_section=36`
       - `union_delimited=7`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=7`
       - `high_bits_normalization=10`
       - `length_prefix_error=25`
       - `serialize_buffer=54`
       - `truncation=79`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
49. SI acceleration/force/torque/voltage parity expansion (`unit/sample`, scalar/vector):
   - Added C harness support in `test/integration/CGoParityCHarness.c`:
     - `c_si_unit_acceleration_vector3_roundtrip`
     - `c_si_unit_force_vector3_roundtrip`
     - `c_si_unit_torque_vector3_roundtrip`
     - `c_si_sample_acceleration_vector3_roundtrip`
     - `c_si_sample_force_vector3_roundtrip`
     - `c_si_sample_torque_vector3_roundtrip`
     - `c_si_unit_voltage_scalar_roundtrip`
     - `c_si_sample_voltage_scalar_roundtrip`
   - Added Go parity cases in `test/integration/CGoParityMain.go`:
     - `si_unit_acceleration_vector3`
     - `si_unit_force_vector3`
     - `si_unit_torque_vector3`
     - `si_sample_acceleration_vector3`
     - `si_sample_force_vector3`
     - `si_sample_torque_vector3`
     - `si_unit_voltage_scalar`
     - `si_sample_voltage_scalar`
   - Added directed vectors:
     - truncation:
       - `si_unit_acceleration_vector3_truncated_input`
       - `si_unit_force_vector3_truncated_input`
       - `si_unit_torque_vector3_truncated_input`
       - `si_sample_acceleration_vector3_truncated_input`
       - `si_sample_force_vector3_truncated_input`
       - `si_sample_torque_vector3_truncated_input`
       - `si_unit_voltage_scalar_truncated_input`
       - `si_sample_voltage_scalar_truncated_input`
     - serialize buffer errors:
       - `si_unit_acceleration_vector3_serialize_small_buffer`
       - `si_unit_force_vector3_serialize_small_buffer`
       - `si_unit_torque_vector3_serialize_small_buffer`
       - `si_sample_acceleration_vector3_serialize_small_buffer`
       - `si_sample_force_vector3_serialize_small_buffer`
       - `si_sample_torque_vector3_serialize_small_buffer`
       - `si_unit_voltage_scalar_serialize_small_buffer`
       - `si_sample_voltage_scalar_serialize_small_buffer`
   - Tightened `test/integration/RunCGoParity.cmake` floors:
     - `cases >= 109` (was 101)
     - `directed >= 196` (was 180)
     - `message_section >= 44` (was 36)
     - directed `truncation >= 87` (was 79)
     - directed `serialize_buffer >= 62` (was 54)
   - Added required marker checks for new random + directed SI vectors.
   - Current observed parity summary:
     - `random=128`
     - `cases=109`
     - `directed=196`
     - random categories include:
       - `message_section=44`
       - `union_delimited=7`
       - `variable_composite=5`
       - `service_section=26`
     - directed categories include:
       - `union_tag_error=7`
       - `high_bits_normalization=10`
       - `length_prefix_error=25`
       - `serialize_buffer=62`
       - `truncation=87`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`.
50. Directed baseline coverage completion (auto-generated truncation + serialize-buffer vectors):
   - Added baseline-directed auto-augmentation in `test/integration/CGoParityMain.go`:
     - `ensureDirectedBaselineCoverage(cases, directed)` appends deterministic auto vectors where missing.
     - auto vector naming uses:
       - `<case>_auto_truncated_input`
       - `<case>_auto_serialize_small_buffer`
       - `<case>_auto_serialize_zero_buffer` (for zero-size cases).
   - Added hard coverage validation in `test/integration/CGoParityMain.go`:
     - `validateDirectedCoverage(cases, directed)` enforces, for every parity case:
       - at least one directed vector,
       - at least one truncation vector,
       - at least one serialize-buffer vector.
     - harness now emits:
       - `PASS directed baseline auto_added=<N>`
       - `PASS directed coverage any=<N> truncation=<N> serialize_buffer=<N>`
   - Tightened `test/integration/RunCGoParity.cmake` floors/checks:
     - `directed >= 265` (was 196)
     - directed `truncation >= 109` (was 87)
     - directed `serialize_buffer >= 109` (was 62)
     - required new summary markers and enforced equality:
       - `coverage_any == observed_cases`
       - `coverage_truncation == observed_cases`
       - `coverage_serialize_buffer == observed_cases`
   - Current observed parity summary:
     - `random=128`
     - `cases=109`
     - `directed=265`
     - directed categories include:
       - `serialize_buffer=109`
       - `truncation=109`
       - `union_tag_error=7`
       - `length_prefix_error=25`
       - `high_bits_normalization=10`
       - `delimiter_error=2`
       - `float_nan=4`
   - Revalidated workflows remain green:
     - `go-differential`
     - `go-differential-homebrew`
     - `go-differential-llvm-env`
     - `full`.
51. Wrapper-to-case inventory lock (no silent parity-case drift):
   - Added C harness wrapper inventory extraction in `test/integration/RunCGoParity.cmake`:
     - counts `DEFINE_ROUNDTRIP(c_*_roundtrip, ...)` lines in `CGoParityCHarness.c`.
   - Added hard invariant in parity gate:
     - `observed_cases == expected_case_count` (wrapper count).
   - This prevents silent regressions where wrappers and executed random parity cases diverge.
   - Current observed invariant:
     - `cases=109`
     - `wrappers=109`
   - Revalidated:
     - `llvmdsdl-uavcan-c-go-parity` green with invariant enforced.
52. Inventory self-consistency hardening (duplicate/summary mismatch guards):
   - Added parity inventory validation in `test/integration/CGoParityMain.go`:
     - `validateCaseInventory(cases, directed)` enforces:
       - no duplicate random case names,
       - no duplicate directed vector names,
       - every directed vector references an existing case.
     - harness emits:
       - `PASS parity inventory cases=<N> directed=<N>`.
   - Added corresponding gate checks in `test/integration/RunCGoParity.cmake`:
     - required inventory marker must be present.
     - `inventory cases == observed_cases`.
     - `inventory directed == observed_directed`.
   - Revalidated:
     - `llvmdsdl-uavcan-c-go-parity` green with inventory marker enforcement.
53. Final closure verification sweep (post-hardening, full ring):
   - Re-ran Go differential workflows after all inventory/baseline checks:
     - `cmake --workflow --preset go-differential`
     - `cmake --workflow --preset go-differential-homebrew`
     - `cmake --workflow --preset go-differential-llvm-env`
   - Re-ran full repository workflow:
     - `cmake --workflow --preset full`
   - All workflows passed with no failures.
   - Final observed parity summary remains:
     - `random=128`
     - `cases=109`
     - `directed=265`
     - `auto_added=69`
     - `inventory cases=109 directed=265`
     - `coverage any=109 truncation=109 serialize_buffer=109`
54. Pass-line execution count hardening (random/directed line-level invariants):
   - Added line-level execution count checks in `test/integration/RunCGoParity.cmake`:
     - number of `PASS <case> random (<N> iterations)` lines must equal `observed_cases`.
     - number of directed vector pass lines must equal `observed_directed`
       (excluding `real16_nan_vector` auxiliary coverage line).
   - This ensures the summary totals cannot drift from actual executed vectors/cases.
   - Revalidated:
     - `llvmdsdl-uavcan-c-go-parity` green with execution-count invariants enabled.
55. Post-hardening end-to-end reconfirmation:
   - Re-ran:
     - `cmake --workflow --preset go-differential`
     - `cmake --workflow --preset full`
   - Both workflows passed after inventory/baseline/pass-line checks were in place.
   - Final parity summary still stable:
     - `random=128`
     - `cases=109`
     - `directed=265`
     - `auto_added=69`
     - `inventory cases=109 directed=265`
     - `coverage any=109 truncation=109 serialize_buffer=109`
