# Workday Plan: MLIR-First Convergence Demo

Date: February 16, 2026  
Target demo time: 4:00 PM (local)

## Status At 1:00 PM Gate

- Gate outcome: on track.
- Completed:
- Added lowered helper families in MLIR pass for signed scalars, float scalars, and variable-array length validation.
- Wired `convert-dsdl-to-emitc` to validate, declare, and consume these helper symbols for typed C SerDes emission.
- Expanded lowered scalar helper coverage to array element fields and unsigned 64-bit paths.
- Enforced typed-path requirements in `convert-dsdl-to-emitc` so variable arrays/scalar fields require lowered helper metadata.
- Added lit coverage for the new helper families in `test/lit/lower-dsdl-serialization-helpers.mlir`.
- Added lit coverage for unsigned 64-bit full-width helper behavior in `test/lit/lower-dsdl-serialization-unsigned64.mlir`.
- Added fixture coverage in `test/lit/fixtures/vendor/Helpers.1.0.dsdl` and extended dsdl-opt integration sanity checks.
- Extended strict uavcan integration checks to assert helper declarations and helper call-site usage in generated C implementations.
- Added strict integration guards ensuring fallback array-length and scalar-saturation logic are absent from generated C impl output.
- Added MLIR-generated delimiter-header validation helpers for delimited composite fields and wired typed C deserialization to call them.
- Added lowering metadata propagation for composite sealing/extent into serialization-plan IO ops.
- Added fixture coverage for delimited composites in `test/lit/fixtures/vendor/Delimited.1.0.dsdl` and `test/lit/fixtures/vendor/UsesDelimited.1.0.dsdl`.
- Added strict integration guards that require lowered delimiter-helper declarations/calls and reject inline delimiter fallback checks.
- Updated typed C delimiter serialization path to invoke lowered delimiter-header validation helper as well.
- Enforced helper-only typed scalar SerDes emission (unsigned/signed/float) with deterministic invalid-argument fail-fast on impossible unsupported typed steps.
- Added strict integration guards rejecting scalar deserialize raw-assignment fallback patterns.
- Added MLIR-generated union-tag IO normalization helpers and wired typed C union SerDes to consume them.
- Added MLIR-generated array-length-prefix IO normalization helpers and wired typed C variable-array SerDes to consume them.
- Added typed-lowering requirements for union-tag IO and array-length-prefix IO helpers (fail-fast if missing).
- Updated lowered capacity-check helper generation to use section-level `max_bits` metadata instead of summing plan steps (fixes union over-estimation path).
- Updated typed C serialization alignment emission to explicitly zero alignment padding bits (deterministic wire output in non-byte-aligned paths).
- Added lit coverage for union-tag helper attributes/functions and array-length-prefix helper attributes/functions.
- Added strict integration guards requiring lowered union-tag IO/array-length-prefix helper declarations and call-site usage.
- Added strict integration guards rejecting inline union-tag and array-length-prefix fallback assignments in generated C.
- Added `test/lit/fixtures/vendor/UnionTag.1.0.dsdl` so dsdl-opt sanity always covers lowered union-tag helper declaration emission.
- Added differential integration test `llvmdsdl-differential-parity` that:
- generates llvm-dsdl C for full `uavcan`,
- generates Nunavut C for representative targets,
- executes randomized deserialize/serialize parity checks for:
- `uavcan.node.Heartbeat.1.0`,
- `uavcan.node.ExecuteCommand.Request.1.3`,
- `uavcan.node.ExecuteCommand.Response.1.3`,
- `uavcan.register.Value.1.0`,
- `uavcan.metatransport.can.Frame.0.2`,
- `uavcan.primitive.array.Real32.1.0`.
- Differential test currently enforces full byte parity on non-float-focused cases and return-code/size parity on float-involved cases to avoid NaN payload canonicalization instability.
- Added reproducible demo artifact automation:
- new CMake target `generate-demo-2026-02-16`,
- new CMake workflows `demo` and `demo-homebrew`,
- auto-generated `build/<preset>/demo-2026-02-16/DEMO.md` with exact commands, artifact counts, selected test results, and command logs.
- Fixed empty-section MLIR lowering blocker:
- `lowerToMLIR` now emits a neutral no-op plan step for empty serialization sections so `dsdl.serialization_plan` verifier requirements are satisfied.
- Full `uavcan` MLIR now lowers through `dsdl-opt --pass-pipeline 'builtin.module(lower-dsdl-serialization)'` successfully.
- Added fixture coverage in `test/lit/fixtures/vendor/EmptyService.1.0.dsdl` to keep this path covered by `llvmdsdl-dsdl-opt-sanity`.
- Added full-tree MLIR integration gate:
- new test `llvmdsdl-uavcan-mlir-lowering` runs `dsdlc mlir --strict` on full `uavcan`, then validates both `lower-dsdl-serialization` and `convert-dsdl-to-emitc` pass pipelines plus helper markers.
- Demo artifact generation now requires full-`uavcan` lowering success (fallback path removed).
- Added cross-backend C/C++ parity gate:
- new test `llvmdsdl-uavcan-cpp-c-parity` generates full `uavcan` C + C++ std outputs and compares deserialize/serialize result-code, consumed/produced sizes, and output bytes across randomized inputs for representative types.
- Added cross-backend C/Rust parity gate scaffolding:
- new test `llvmdsdl-uavcan-c-rust-parity` generates full `uavcan` C + Rust outputs, links generated C implementation units into a Rust harness, and compares deserialize/serialize result-code, consumed/produced sizes, and output bytes for representative types.
- Test is auto-registered only when `cargo` is available in the build environment.
- Fixed C++ serialize alignment determinism to match C MLIR-backed output:
- C++ emitter now zeros alignment padding bits explicitly during serialization (previously advanced offset only), eliminating byte-level drift.
- Verification:
- `ctest --test-dir build/dev-homebrew --output-on-failure` is green.
- Manual FileCheck runs for lowering lit test files are green.

## Status Update: Post-Rust Toolchain Enablement

- `cargo` + `rustc` are now available in the local environment and CMake auto-registers `llvmdsdl-uavcan-c-rust-parity`.
- Activated and stabilized the full C/Rust parity gate:
- fixed Rust runtime literal suffix portability (`U32`/`U8` -> Rust-valid suffixes),
- fixed Rust constant typing for string-valued DSDL constants (`&'static str`),
- fixed composite-array serialize expression bug (`&Result` vs `Result`),
- fixed Rust serialize alignment padding determinism (explicit zeroing to match C/C++),
- fixed Rust deserialize truncation-safety slice clamping (no panic on short buffers).
- Updated C/Rust parity harness expectation policy:
- deserialize return-code parity is always enforced,
- deserialize consumed-size parity is now enforced on both success and error paths,
- serialize return-code/size/byte parity is enforced on success paths.
- Added a non-breaking generated Rust API extension:
- `deserialize_with_consumed(&mut self, &[u8]) -> (i8, usize)`
- Existing ergonomic `deserialize(&mut self, &[u8]) -> Result<usize, i8>` is preserved and now wraps the new method.
- Verification status:
- `ctest --test-dir build/dev-homebrew -R llvmdsdl-uavcan-c-rust-parity --output-on-failure` is green.
- Added dedicated generated-Rust compile gate `llvmdsdl-uavcan-rust-cargo-check` (`cargo check` on full generated `uavcan` crate).
- Expanded C/Rust parity representative set with `uavcan.register.Value.1.0` to cover large tagged-union behavior in addition to Heartbeat, ExecuteCommand (request/response), and CAN Frame.
- Added deterministic directed error-path parity checks to both cross-backend suites:
- bad union-tag deserialize (`uavcan.metatransport.can.Frame.0.2`)
- bad union-tag serialize (`uavcan.metatransport.can.Frame.0.2`)
- bad variable-array-length deserialize (`uavcan.node.ExecuteCommand.Response.1.3`)
- bad variable-array-length serialize (`uavcan.node.ExecuteCommand.Response.1.3`)
- bad delimiter-header deserialize (`uavcan.node.port.List.1.0`)
- too-small serialize buffer (`uavcan.node.Heartbeat.1.0`)
- truncated/empty-input deserialize zero-extension (`uavcan.node.Heartbeat.1.0`)
- nested-composite bad union-tag deserialize (`uavcan.node.port.List.1.0 -> SubjectIDList`)
- deeper nested-chain bad union-tag deserialize (first section valid, second nested section invalid) (`uavcan.node.port.List.1.0`)
- deeper delimiter-chain failure (first two sections valid, third delimiter invalid) (`uavcan.node.port.List.1.0`)
- nested-composite bad array-length serialize (`uavcan.node.port.List.1.0 -> SubjectIDList.sparse_list`)
- multi-level delimiter-header failure (first delimited section valid, second invalid) (`uavcan.node.port.List.1.0`)
- service request-path bad array-length serialize (`uavcan.node.ExecuteCommand.Request.1.3`)
- service request-path truncated payload success+reserialize parity (`uavcan.node.ExecuteCommand.Request.1.3`)
- service request-path too-small serialize buffer (`uavcan.node.ExecuteCommand.Request.1.3`)
- service request-path note: deserialize "bad length" negative case is not applicable for `ExecuteCommand.Request` because the length prefix range matches capacity (`<=255`), so invalid-length request deserialize is not a reachable representation error for this type.
- service response-path truncated payload success+reserialize parity (`uavcan.node.ExecuteCommand.Response.1.3`)
- Directed checks now assert shared runtime error-code parity, not just random roundtrip parity.
- `ctest --test-dir build/dev-homebrew --output-on-failure` is green (`10/10`).
- Added C/C++ PMR profile parity gate `llvmdsdl-uavcan-cpp-pmr-c-parity` using the same directed harness logic as the std profile.
- Updated C++ parity harness to use profile-agnostic member SerDes calls, enabling parity validation for both std and PMR variants.
- `ctest --test-dir build/dev-homebrew --output-on-failure` is green (`11/11`).
- MLIR-first convergence hardening for non-C backends:
- C++ and Rust emitters now validate MLIR schema coverage (`dsdl.schema` + expected `dsdl.serialization_plan` sections) against semantic definitions before emitting code.
- C++ and Rust now also validate plan/step metadata presence (union tag metadata, variable-array prefix widths, composite metadata, and core IO attributes) so missing MLIR facts fail fast.
- C++ and Rust now run `lower-dsdl-serialization` during validation and require lowered helper metadata families (union-tag, array-length prefix/validate, scalar normalize helpers, delimiter validate helpers) before emission.
- This removes the previous “module ignored” behavior and ensures both backends consume MLIR structure even where helper-body lowering is still language-specific.
- C++ and Rust now consume lowered helper symbols directly for union-tag normalization and variable-array prefix/length validation:
- each emitted SerDes function binds deterministic local helper closures named from lowered MLIR helper symbols,
- union-tag serialize/deserialize paths call lowered helper bindings before runtime bit I/O,
- variable-array serialize/deserialize paths call lowered prefix/validate helper bindings instead of relying only on ad hoc inline checks.
- Added integration guards to keep this locked:
- `RunUavcanCppGeneration.cmake` now fails if generated C++ (`std`/`pmr`) headers do not contain MLIR-derived union-tag, array-prefix, and array-validate helper bindings.
- `RunUavcanRustGeneration.cmake` now fails if generated Rust files do not contain MLIR-derived union-tag, array-prefix, and array-validate helper bindings.
- Scalar helper convergence completed for non-C backends:
- C++ and Rust now capture lowered scalar helper symbols per field (unsigned/signed/float) from lowered MLIR and emit deterministic local helper bindings for those symbols.
- C++ and Rust scalar serialize/deserialize paths now route normalization through these lowered helper bindings for both plain scalar fields and scalar array elements.
- Added integration guards requiring MLIR scalar helper bindings (`scalar_unsigned`, `scalar_signed`, `scalar_float`) in generated C++ and Rust outputs.
- Delimiter helper convergence completed for non-C backends:
- C++ and Rust now capture lowered delimiter-header validation helper symbols per delimited composite field and emit deterministic local helper bindings for those symbols.
- C++ and Rust nested composite serialize/deserialize paths now invoke lowered delimiter helper bindings (instead of only inline ad hoc delimiter checks).
- Added integration guards requiring MLIR delimiter helper bindings (`validate_delimiter_header`) in generated C++ and Rust outputs.
- Capacity-check helper convergence completed for non-C backends:
- `lower-dsdl-serialization` now tags each plan with `lowered_capacity_check_helper`.
- C++ and Rust now require this lowered helper metadata during MLIR coverage validation and bind/call the lowered capacity helper in top-level serialize preflight.
- Added integration guards requiring MLIR capacity-check helper bindings (`capacity_check`) in generated C++ and Rust outputs.
- Strengthened integration guards to require capacity-helper call-site patterns (not only symbol presence) in generated C++/Rust outputs.
- Added lit coverage for lowered plan-level capacity helper metadata in:
- `test/lit/lower-dsdl-serialization.mlir`
- `test/lit/lower-dsdl-serialization-helpers.mlir`
- Union-tag validation helper convergence completed for non-C backends:
- `lower-dsdl-serialization` now tags union plans with `lowered_union_tag_validate_helper`.
- C++ and Rust now require this lowered union-tag validation metadata during MLIR coverage validation.
- C++ and Rust union serialize/deserialize paths now bind and invoke the lowered union-tag validation helper before branch dispatch.
- Added integration guards requiring MLIR union-tag validation helper bindings and call-site patterns in generated C++ and Rust outputs.
- Added lit coverage for lowered plan-level union-tag validation helper metadata in:
- `test/lit/lower-dsdl-serialization.mlir`
- Introduced a shared non-C helper descriptor layer for MLIR-derived SerDes helper contracts:
- new shared model/API in:
- `include/llvmdsdl/CodeGen/SerDesHelperDescriptors.h`
- `lib/CodeGen/SerDesHelperDescriptors.cpp`
- wired into both C++ and Rust emitters so capacity-check and union-tag-validate helper-body semantics are derived from one descriptor contract (allowed tags + required bits), reducing duplicated backend logic.
- extended shared descriptors to variable-array helper contracts:
- shared array descriptors now carry prefix symbol, validate symbol, prefix width, and capacity.
- C++ and Rust helper-binding emission now consumes this shared array descriptor contract in both serialize and deserialize helper generation paths.
- build integration updated in `lib/CodeGen/CMakeLists.txt`.
- extended shared descriptors to scalar and delimiter helper contracts:
- shared scalar descriptors now carry kind (`unsigned`/`signed`/`float`), ser/deser symbols, bit-length, and cast mode.
- shared delimiter descriptors now carry deterministic helper symbols for delimited composite validation.
- C++ and Rust now consume shared scalar/delimiter descriptor contracts in both serialize and deserialize helper-binding generation paths.
- C++ deserialize helper generation now uses the shared descriptor API (matching C++ serialize path) instead of ad hoc per-category branching.
- Rust serialize/deserialize helper generation now uses shared descriptor-driven helper-body emission for scalar and delimiter families (matching C++ behavior).
- Extended descriptor API to type-level overloads for call-site reuse:
- `buildScalarHelperDescriptor(const SemanticFieldType&, ...)`
- `buildArrayLengthHelperDescriptor(const SemanticFieldType&, ...)`
- `buildDelimiterValidateHelperDescriptor(const SemanticFieldType&, ...)`
- C++ and Rust call-site emission now uses these shared descriptors (not just helper-binding generation) for:
- scalar serialize/deserialize helper selection,
- variable-array prefix/length helper selection,
- delimiter-header validation helper selection.
- Removed now-dead per-backend helper-binding accessor methods superseded by descriptor-driven call-site logic.
- Focused MLIR/convergence/parity ring remains green after the refactor:
- `ctest --test-dir build/dev-homebrew -R "llvmdsdl-dsdl-opt-sanity|llvmdsdl-uavcan-mlir-lowering|llvmdsdl-uavcan-rust-generation|llvmdsdl-uavcan-cpp-generation|llvmdsdl-uavcan-cpp-c-parity|llvmdsdl-uavcan-cpp-pmr-c-parity|llvmdsdl-uavcan-c-rust-parity|llvmdsdl-signed-narrow-cpp-c-parity|llvmdsdl-signed-narrow-cpp-pmr-c-parity|llvmdsdl-signed-narrow-c-rust-parity" --output-on-failure` -> `10/10` passed.
- Added directed saturation/truncation edge-vector parity assertions in both cross-backend harnesses:
- saturation edge: `uavcan.node.Health.1.0` serialize with out-of-range value (`0xFF`) must clamp to `0x03`.
- truncation edge: `uavcan.time.SynchronizedTimestamp.1.0` serialize with `0xFEDCBA9876543210` must truncate to 56-bit payload bytes `10 32 54 76 98 BA DC`.
- Added signed scalar edge-vector parity assertions for `uavcan.primitive.scalar.Integer8.1.0`:
- directed deserialize vectors for `0x80 -> -128` and `0xFF -> -1`.
- directed serialize vectors for signed negative values preserving two's-complement byte output.
- Added random roundtrip parity coverage for `uavcan.primitive.scalar.Integer8.1.0` in both C/C++ and C/Rust parity suites.
- Added additional random roundtrip parity coverage for `uavcan.node.Health.1.0` and `uavcan.time.SynchronizedTimestamp.1.0` in both C/C++ and C/Rust parity suites.
- Extended `test/integration/CRustParityCHarness.c` with C roundtrip hooks and directed serialize hooks for Health/Timestamp so Rust directed checks compare against C-side behavior directly.
- Extended `test/integration/CRustParityCHarness.c` further with `c_integer8_roundtrip` for signed scalar parity vectors.
- Preserved deterministic parity baseline vectors by moving new random parity cases after existing high-risk corpus cases (avoids perturbing previous RNG stream for `uavcan.register.Value.1.0`).
- Added signed narrow-width fixture corpus for explicit sub-byte signed semantics:
- `test/integration/fixtures/signed_narrow/vendor/Int3Sat.1.0.dsdl`
- `test/integration/fixtures/signed_narrow/vendor/Int3Trunc.1.0.dsdl`
- Added dedicated parity harness and gates for this corpus:
- `test/integration/SignedNarrowCppCParityMain.cpp`
- `llvmdsdl-signed-narrow-cpp-c-parity` (`std`)
- `llvmdsdl-signed-narrow-cpp-pmr-c-parity` (`pmr`)
- Added matching signed-narrow C/Rust parity harness and gate:
- `test/integration/SignedNarrowCRustParityCHarness.c`
- `test/integration/SignedNarrowCRustParityMain.rs`
- `llvmdsdl-signed-narrow-c-rust-parity`
- Directed vectors now lock saturating/truncating int3 behavior plus sub-byte signed sign-extension parity across C/C++.
- Directed vectors now also lock sub-byte signed serialize/deserialize parity across C/Rust.
- Refactored MLIR lowered-facts extraction/validation into a shared CodeGen module:
- `include/llvmdsdl/CodeGen/MlirLoweredFacts.h`
- `lib/CodeGen/MlirLoweredFacts.cpp`
- C++ and Rust emitters now both use `collectLoweredFactsFromMlir(...)` (backend label parameterized) instead of duplicated per-backend validation logic.
- Shared lowered-field helpers (`findLoweredFieldFacts`, `loweredFieldArrayPrefixBits`) now back both C++ and Rust emitters for field-level helper lookup/prefix-width access.
- Shared lowered-facts keying utility (`loweredTypeKey`) now backs both emitters, removing duplicated per-backend key formatting logic.
- Added shared section helper-binding planner for C++/Rust helper emission convergence:
- `include/llvmdsdl/CodeGen/SectionHelperBindingPlan.h`
- `lib/CodeGen/SectionHelperBindingPlan.cpp`
- C++ and Rust helper-binding emitters now consume `buildSectionHelperBindingPlan(...)` for serialize/deserialize paths, removing duplicated per-backend descriptor-walk logic.
- Added unit coverage for shared helper-binding planner directionality/union/array/delimiter behavior:
- `test/unit/SectionHelperBindingPlanTests.cpp`
- Added shared helper-binding renderer entry-point for plan-walk convergence:
- `renderSectionHelperBindings(...)` in `include/llvmdsdl/CodeGen/HelperBindingRender.h` + `lib/CodeGen/HelperBindingRender.cpp`.
- C++ and Rust helper-binding emitters now use this shared plan renderer for both serialize/deserialize helper binding emission.
- Expanded helper-binding renderer coverage with:
- `test/unit/HelperBindingRenderTests.cpp` (including plan-level render assertions).
- Added shared helper-symbol resolver module to centralize field/section helper symbol selection:
- `include/llvmdsdl/CodeGen/HelperSymbolResolver.h`
- `lib/CodeGen/HelperSymbolResolver.cpp`
- C++ and Rust now both use shared helper-symbol resolution for:
- section helpers (capacity/union-tag mask/union-tag validate)
- scalar helper call-site selection (ser/deser)
- array prefix/validate helper call-site selection
- delimiter helper call-site selection
- Added unit coverage for resolver contracts:
- `test/unit/HelperSymbolResolverTests.cpp`
- Added shared wire-layout facts module for union tag bit resolution:
- `include/llvmdsdl/CodeGen/WireLayoutFacts.h`
- `lib/CodeGen/WireLayoutFacts.cpp`
- C++ emitter, Rust emitter, and section helper-binding planner now share one `resolveUnionTagBits(...)` implementation.
- Added unit coverage for shared wire-layout facts:
- `test/unit/WireLayoutFactsTests.cpp`
- Focused convergence ring remains green after this refactor (`10/10`), and full suite remains green.
- Full suite remains green after this expansion (`ctest --test-dir build/dev-homebrew --output-on-failure`: `14/14`).
- `cmake --build --preset build-dev-homebrew --target generate-demo-2026-02-16` is green and includes C/Rust parity in the demo run.
- Secondary cleanup completed:
- reduced Rust warning volume by removing an unused serialize local and redundant cast parentheses in generated Rust expressions.
- Added shared field-level SerDes statement planning for non-C backends:
- `include/llvmdsdl/CodeGen/SerDesStatementPlan.h`
- `lib/CodeGen/SerDesStatementPlan.cpp`
- C++ and Rust now both consume one shared `buildSectionStatementPlan(...)` contract for:
- ordered non-union field walks (alignment + padding/value path shaping)
- union branch selection input (non-padding alternatives with MLIR-derived per-field prefix/facts metadata)
- Added unit coverage for shared statement-plan behavior:
- `test/unit/SerDesStatementPlanTests.cpp`
- Focused convergence ring remains green after this change (`10/10`), and full suite remains green (`14/14`).
- Added shared array-element normalization utility:
- `arrayElementType(const SemanticFieldType&)` in:
- `include/llvmdsdl/CodeGen/TypeStorage.h`
- `lib/CodeGen/TypeStorage.cpp`
- Rewired C++ and Rust array SerDes element-path shaping to use this shared utility (removing duplicated per-backend array-element metadata resets).
- Extended unit coverage in `test/unit/TypeStorageTests.cpp`.
- Focused convergence ring and full suite remain green after this slice (`10/10`, `14/14`).
- MLIR-first statement planning now consumes lowered step ordering metadata:
- `LoweredFieldFacts` now captures lowered `step_index` from MLIR (`collectLoweredFactsFromMlir`).
- `buildSectionStatementPlan(...)` now orders section and union field steps by lowered step index when available (stable fallback to semantic order otherwise).
- Added ordering coverage in `test/unit/SerDesStatementPlanTests.cpp`.
- Focused convergence ring and full suite remain green after this slice (`10/10`, `14/14`).
- Added shared array-wire planning contract for non-C backends:
- `include/llvmdsdl/CodeGen/ArrayWirePlan.h`
- `lib/CodeGen/ArrayWirePlan.cpp`
- `buildArrayWirePlan(...)` now centralizes variable-array detection, effective prefix-width resolution, and lowered array-helper descriptor selection.
- C++ and Rust emitters now consume this shared array-wire plan in both serialize and deserialize array paths.
- Added unit coverage in `test/unit/ArrayWirePlanTests.cpp`.
- Focused convergence ring and full suite remain green after this slice (`10/10`, `14/14`).

## Objective

By 4:00 PM, show `llvm-dsdl` to the OpenCyphal community as a credible MLIR-first replacement path for Nunavut, with clear evidence of:

- End-to-end generation for `uavcan` in C, C++, and Rust.
- Growing shared MLIR execution core for serialization/deserialization behavior.
- Deterministic tests and reproducible outputs.

## 4:00 PM Definition Of Done

- `uavcan` generation is green for `c`, `cpp`, and `rust`.
- C backend uses lowered MLIR helpers for:
- capacity checks
- union-tag validation
- fixed unsigned scalar normalization
- At least one additional nontrivial semantic family is lowered and consumed today.
- Differential checks pass on representative sample types.
- Demo artifacts and commands are captured in a reproducible folder.

## Time-Boxed Schedule

### 9:00-9:30 AM: Baseline Freeze

- Run full test suite and record baseline.
- Save baseline generated outputs for demo sample types.
- Create a demo artifact folder:
- `build/demo-2026-02-16/`

Acceptance:

- Baseline commands and results are reproducible from clean state.

### 9:30-11:00 AM: Lowering Expansion (MLIR)

- Extend `lower-dsdl-serialization` with new helper families:
- signed scalar normalization helpers
- float conversion helpers
- array prefix validation helpers
- Ensure helper symbols are deterministic and section-aware (`request`/`response`).
- Add/update lit tests for every new helper family.

Acceptance:

- New helper families are present in lowered IR and lit checks pass.

### 11:00 AM-12:00 PM: C Backend Consumption

- Make `convert-dsdl-to-emitc` consume newly lowered helper symbols.
- Fail fast when required helper symbols are missing.
- Remove fallback logic for newly covered semantics.

Acceptance:

- Strict `uavcan` C generation still passes.
- Integration checks prove helper declarations, calls, and helper bodies are emitted.

### 12:00-12:30 PM: Midday Checkpoint

- Validate progress against 1:00 PM gate.
- Adjust scope if needed to protect 4:00 PM demo quality.

### 12:30-2:00 PM: Cross-Language Reuse Push

- Reuse lowered helper semantics in C++ and Rust generation paths where feasible.
- Keep language-specific emitters focused on:
- type declarations
- namespace/module layout
- allocator/profile surface differences
- Avoid duplicating semantics that are now represented by shared helpers.

Acceptance:

- No regression in `llvmdsdl-uavcan-cpp-generation` and `llvmdsdl-uavcan-rust-generation`.

### 2:00-3:00 PM: Differential Validation

- Run byte-level comparisons on representative set:
- `uavcan.node.Heartbeat.1.0`
- `uavcan.register.Value.1.0`
- `uavcan.metatransport.can.Frame.0.2`
- one service request/response pair
- Validate:
- deserialize status code parity
- consumed-size parity
- serialize roundtrip parity

Acceptance:

- Differential sample suite passes and artifacts are captured.

### 3:00-3:40 PM: Demo Packaging

- Prepare `build/demo-2026-02-16/DEMO.md`:
- exact commands
- expected output snippets
- known limitations
- Produce concise capability matrix for presentation.
- Update `README.md` and `DESIGN.md` with current status snapshot.

Acceptance:

- One-command or minimal-command demo flow is documented and runnable.

### 3:40-4:00 PM: Final Rehearsal

- Run demo commands end-to-end.
- Confirm generated output snippets used in presentation are current.
- Freeze branch state for presentation.

Acceptance:

- Dry run succeeds without improvisation.

## Hard Gates

### 1:00 PM Gate

- New helper families lowered and lit-tested. (Done)

### 2:30 PM Gate

- Strict `uavcan` generation green with helper usage checks. (Done)

### 3:15 PM Gate

- Differential sample suite green. (Done)

### 3:40 PM Gate

- Demo docs and artifacts frozen.

## Next Autonomous Block

- Continue reducing backend divergence in cross-language SerDes behavior:
- drive additional MLIR-shared helper reuse into Rust/C++ emission points where semantics are still emitted ad hoc.
- Expand directed parity from current union/array/delimiter/buffer/truncation/nested/multi-level-delimiter/service checks to deeper nested composite chains and additional mixed success/failure service-path buffers.
- Next convergence target: reduce duplicated backend helper emission and helper-symbol resolution by introducing shared renderer + resolver layers for C++/Rust, then lock with directed saturation/truncation edge-vector parity assertions.
- Status: shared descriptor layer now covers capacity-check, union-tag-validate, array-length prefix/validate, scalar normalization (unsigned/signed/float), and delimiter validation across helper-body generation, plan-walk rendering, and helper call-site symbol resolution in both C++/Rust.
- Directed saturation/truncation edge-vector parity assertions are now in place for signed/unsigned clipping families currently exercised by `uavcan` (`Health` saturating uint2 and `SynchronizedTimestamp` truncating uint56).
- Signed edge vectors are now expanded with `uavcan.primitive.scalar.Integer8.1.0` directed checks (`0x80/-128`, `0xFF/-1`) across C/C++ and C/Rust parity harnesses.
- Signed narrow-width edge vectors (sub-byte signed widths) are now covered by a dedicated fixture corpus and parity gates in both C/C++ (std+pmr) and C/Rust.
- Field-level shared statement planning is now in place for C++/Rust section walks and union-branch shaping.
- Field-level planning is now additionally MLIR-step-index aware (statement order comes from lowered facts when present).
- Array wire-shaping prelude is now shared; next target remains lifting backend-specific scalar/composite statement emission branches into shared plan/render contracts (keeping language syntax emission thin).

## Implementation Focus Areas

### Primary

- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/Transforms/Passes.cpp`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/Transforms/ConvertDSDLToEmitC.cpp`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/lit/lower-dsdl-serialization.mlir`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/integration/RunDsdlOptSanity.cmake`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/test/integration/RunUavcanStrictGeneration.cmake`

### Secondary

- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/CppEmitter.cpp`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/lib/CodeGen/RustEmitter.cpp`

## Demo Narrative (Community-Facing)

- We parse/analyze DSDL once, lower semantics into MLIR once, and reuse lowering products across language targets.
- C path is currently furthest along and demonstrates concrete MLIR-shared helper usage in emitted code.
- C++ and Rust generation are available today and are converging toward the same MLIR-shared behavioral core.
- This architecture enables stronger validation, better optimization opportunities, and reduced backend duplication versus template-only generation.

## If Time Runs Tight

- Prioritize robust C MLIR-core convergence and test evidence.
- Keep C++/Rust working and document partial convergence status honestly.
- Do not skip differential checks for the representative sample set.

## End-Of-Day Deliverables

- Passing test results (saved transcript).
- Demo artifact bundle in `build/demo-2026-02-16/`.
- Updated status in:
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/README.md`
- `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/DESIGN.md`
- This plan file with completed tasks checked.
