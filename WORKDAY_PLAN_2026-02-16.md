# Workday Plan: MLIR-First Convergence Demo

Date: February 16, 2026  
Target demo time: 4:00 PM (local)

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

- New helper families lowered and lit-tested.

### 2:30 PM Gate

- Strict `uavcan` generation green with helper usage checks.

### 3:15 PM Gate

- Differential sample suite green.

### 3:40 PM Gate

- Demo docs and artifacts frozen.

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
