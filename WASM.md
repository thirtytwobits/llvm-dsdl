# DSDL to WebAssembly Plan (No High-Level Language Codegen)

## 1. Objective

Build a first-class `dsdlc wasm` backend that compiles DSDL directly through the existing compiler pipeline:

`DSDL -> Semantic model -> MLIR (DSDL dialect + lowering) -> LLVM IR -> WebAssembly`

with **no TypeScript/ECMAScript (or other high-level language) source generation**.

## 2. Clarification: What "No Codegen" Means Here

This plan avoids high-level source generation (TS/JS/Python/C++ outputs).
It still performs compiler code generation to machine-independent binary form:

- MLIR generation/lowering
- LLVM IR generation
- WebAssembly binary emission (`.wasm`)

## 3. Scope

### In scope

- New `dsdlc wasm` command.
- Deterministic WebAssembly module generation from any DSDL root namespace.
- Wasm export ABI for per-type serialize/deserialize entry points.
- Runtime support compiled into Wasm (bit operations, numeric conversion, bounds checks).
- Optional sidecar metadata (`manifest.json`) describing exports and type mapping.
- Test coverage in lit + integration + matrix workflows.

### Out of scope (initially)

- Generating TS/JS glue.
- Browser networking stack integration (Cyphal transport in-browser).
- WASI component model packaging as a required path (can be a follow-up).
- Windows-specific packaging polish (Linux/macOS first).

## 4. Public Interface

### 4.1 CLI

Add:

`dsdlc wasm --root-namespace-dir <dir> --out-dir <dir> [options]`

Proposed options:

- `--wasm-target <wasm32-unknown-unknown|wasm32-wasi>` (default `wasm32-unknown-unknown`)
- `--wasm-module-name <name>` (default derived from root namespace)
- `--emit-mlir` (write post-lowering MLIR for inspection)
- `--emit-llvm-ir` (write `.ll` text)
- `--emit-wat` (optional textual disassembly if tool is available)
- `--optimize-lowered-serdes` (existing semantics-preserving pass pipeline)

Outputs under `--out-dir`:

- `<module>.wasm` (required)
- `<module>.manifest.json` (required)
- `<module>.mlir` (optional)
- `<module>.ll` (optional)
- `<module>.wat` (optional)

### 4.2 ABI Contract (Wasm exports)

Exports are C-ABI-like and pointer/length based over linear memory.

Per type `T` (mangled symbol suffix):

- `dsdl_serialize_<T>(in_ptr, in_len, out_ptr, out_cap, out_written_ptr) -> i32`
- `dsdl_deserialize_<T>(in_ptr, in_len, out_ptr, out_cap, out_written_ptr) -> i32`

Common exports:

- `dsdl_manifest_ptr() -> i32`
- `dsdl_manifest_len() -> i32`
- `dsdl_last_error_code() -> i32`

Status codes:

- `0` success
- non-zero deterministic error enum (bounds, invalid union tag, truncated payload, etc.)

### 4.3 Canonical Value Buffer (CVB)

To avoid high-level language type generation, host input/output uses a canonical packed representation:

- Scalars: fixed little-endian widths.
- Fixed arrays: contiguous elements.
- Variable arrays: `u32 length` + contiguous payload.
- Unions: `u8/u16 tag` + active payload.
- Composites: declaration order layout with required alignment/padding rules.

The exact layout for each type is described in `<module>.manifest.json`.

## 5. Compiler Architecture Work

### 5.1 Backend wiring

- Add `WasmEmitter`:
  - `include/llvmdsdl/CodeGen/WasmEmitter.h`
  - `lib/CodeGen/WasmEmitter.cpp`
- Wire through `tools/dsdlc/main.cpp`.
- Add `lib/CodeGen/CMakeLists.txt` entry.

### 5.2 Lowering path

Reuse existing front-end + semantic + lowering-facts stack, then add:

1. `lower-dsdl-serialization` (existing)
2. `convert-dsdl-to-llvm` (new conversion path or extension)
3. LLVM optimization pipeline tuned for Wasm-safe transformations
4. Emit LLVM IR module with target triple set to Wasm target
5. Emit object + link to `.wasm` (or direct backend emission, depending on toolchain path)

### 5.3 Runtime integration

- Reuse `runtime/dsdl_runtime.h` semantics as the truth source.
- Add Wasm-oriented runtime C/C++ unit:
  - no libc-heavy dependencies
  - no dynamic allocation requirement for core SerDes path
  - explicit bounds checks on every memory touch
- Compile runtime into the same Wasm module as generated type functions.

### 5.4 Manifest generation

Emit deterministic JSON with:

- module name/version
- root namespace hash
- per-type:
  - full type name/version
  - DSDL extent + bit length facts
  - export symbol names
  - CVB layout schema (field offsets/tags/array rules)

## 6. Build and Toolchain Integration

### 6.1 Required tools

- LLVM/MLIR build with WebAssembly backend enabled
- linker supporting Wasm output (`wasm-ld`)
- optional disassembler (`wasm2wat` or equivalent)
- runtime executor for tests (`wasmtime` preferred)

### 6.2 CMake integration

Add options:

- `LLVMDSDL_ENABLE_WASM_BACKEND` (default `ON` if toolchain supports it)
- `LLVMDSDL_WASM_RUNTIME_EXECUTOR` (`wasmtime` path, optional)

Add build/demo targets:

- `generate-uavcan-wasm`
- `run-wasm-serdes-smoke` (if executor available)

## 7. Testing Strategy

### 7.1 Lit

- CLI help/validation for `dsdlc wasm`.
- deterministic file naming.
- manifest contains expected exports for fixture types.

### 7.2 Integration

- Fixture corpus round-trip through Wasm serialize/deserialize.
- Determinism: repeated generation yields byte-identical `.wasm` and manifest.
- Differential parity: Wasm output equality vs existing C backend for representative fixtures.
- Negative tests: truncated input, invalid tags, array bounds overflow.

### 7.3 Performance

- Microbench for scalar/array/union/composite workloads.
- Compare:
  - Wasm module in `wasmtime`
  - native C backend baseline
- Capture throughput + p50/p99 latency for fixed fixture set.

## 8. Delivery Workstreams

### Workstream A: MVP compiler path (`dsdlc wasm`)

Deliver:

- CLI command + minimal options
- `.wasm` + manifest generation
- per-type serialize/deserialize exports
- fixture smoke tests

Exit criteria:

- one-command generation succeeds for fixture and `uavcan` trees
- round-trip tests pass in executor

### Workstream B: ABI hardening + manifest contract

Deliver:

- finalized status/error enum
- manifest schema v1
- strict bounds checking and deterministic failure behavior

Exit criteria:

- ABI documented and frozen for v1
- invalid input corpus returns deterministic non-zero codes

### Workstream C: Differential parity

Deliver:

- C vs Wasm byte-equality tests for representative type corpus
- optimized and non-optimized lane coverage

Exit criteria:

- parity suite green in RelWithDebInfo full lane

### Workstream D: Performance and optimization

Deliver:

- benchmark harness and report artifact
- tuned pass pipeline for Wasm backend

Exit criteria:

- measurable improvement from baseline and no correctness regressions

### Workstream E: Documentation and demo

Deliver:

- README + DESIGN + DEMO updates
- 5-minute demo script with generated Wasm and runtime call trace

Exit criteria:

- reproducible demo on macOS and Linux

### Workstream F: WASI Component Model Packaging (Phase 2)

Deliver:

- Define a WIT interface for exported SerDes operations and error model.
- Add canonical ABI lift/lower adapter generation/integration.
- Package generated core Wasm into a component-model artifact.
- Add component-runtime execution lanes (component-aware runtime, e.g. Wasmtime component support).
- Document host-consumption flow for component bindings.

Exit criteria:

- Generated output includes a component-model package artifact.
- Component API is callable through typed host bindings without manual pointer/length plumbing.
- Component lanes pass deterministic smoke and round-trip tests.

## 9. Risks and Mitigations

- Risk: ABI too hard to consume without generated bindings.
  - Mitigation: provide stable manifest schema + tiny runtime helper library (non-generated).
- Risk: Wasm runtime overhead vs native.
  - Mitigation: optimize hot SerDes loops, minimize memory copies, benchmark continuously.
- Risk: Toolchain variability across environments.
  - Mitigation: version-pinned CI images and explicit capability checks in CMake.
- Risk: CVB format churn.
  - Mitigation: schema versioning from day one; additive evolution only.

## 10. Acceptance Criteria

Project is "Wasm backend complete" when:

1. `dsdlc wasm` is a documented first-class command.
2. DSDL trees compile to deterministic `.wasm` + manifest artifacts.
3. Wasm serialize/deserialize round-trip correctness is validated in CI.
4. Differential parity vs C backend is validated for representative fixtures.
5. No TS/ECMAScript code generation is required for consumers to use Wasm output.

Project reaches "Wasm component packaging complete" when:

1. WIT-defined component interfaces are versioned and documented.
2. Component-model artifacts are generated reproducibly in CI.
3. Component-runtime tests pass for representative fixture and `uavcan` samples.

## 11. Implementation Sequence (Practical)

1. Add CLI + emitter skeleton + manifest writer.
2. Implement minimal type set (scalars, fixed arrays).
3. Add variable arrays + composites.
4. Add unions + services.
5. Add differential parity tests.
6. Tune optimization pipeline and publish benchmark results.
7. Finalize docs and demo flow.
8. Phase 2: add WASI Component Model packaging (WIT + canonical ABI adapters + component-runtime tests).
