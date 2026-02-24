# llvm-dsdl

![](llvm-dsdl.png)

`llvm-dsdl` is an out-of-tree LLVM/MLIR-based DSDL compiler for Cyphal data
types.

Canonical project docs:

- `DEMO.md`: 5-minute demo flow (quick + scale-up paths).
- `DESIGN.md`: architecture snapshot.
- `CONTRIBUTING.md` section 15: release readiness runbook.
- `tools/dsdld/README.md`: `dsdld` usage (VS Code + Neovim setup).
- `editors/vscode/dsdld-client/README.md`: extension install/debug/settings.
- `docs/LSP_LINT_RULE_AUTHORING.md`: lint rule implementation workflow.
- `docs/LSP_AI_OPERATOR_GUIDE.md`: AI safety and operator controls.
- `docs/CONVERGENCE_SCORECARD.md`: generated MLIR-convergence score snapshot.
- `docs/PARITY_MATRIX.md`: generated parity matrix coverage snapshot.
- `docs/MALFORMED_INPUT_CONTRACT_MATRIX.md`: generated malformed-input contract matrix snapshot.
- `LICENSE.md`: project license (MIT).
- `THIRD_PARTY_NOTICES.md`: third-party licensing notices.

It currently provides:

- A DSDL frontend (`.dsdl` discovery, parse, semantic analysis).
- A custom MLIR DSDL dialect and lowering pipeline hooks.
- C11 code generation (`dsdlc --target-language c`) with:
  - Per-type headers mirroring namespace directories.
  - Declarations/wrappers in headers plus per-definition `.c` SerDes implementations generated via MLIR/EmitC lowering.
  - A local header-only runtime (`dsdl_runtime.h`) with bit-level primitives.
- C++23 code generation (`dsdlc --target-language cpp`) with:
  - Per-type headers (`.hpp`) mirroring namespace directories.
  - Header-only inline SerDes for both `std` and allocator-oriented `pmr` profiles.
  - A C++ runtime wrapper (`dsdl_runtime.hpp`) over the core bit-level runtime.
  - MLIR schema/plan metadata validation before emission (fail-fast on malformed IR facts).
- Rust code generation (`dsdlc --target-language rust`) with:
  - A generated crate layout (`Cargo.toml`, `src/lib.rs`, `src/**`).
  - Per-type Rust data types and inline SerDes methods.
  - A local Rust runtime module (`src/dsdl_runtime.rs`) with bit-level primitives.
  - `std` and `no-std-alloc` profiles plus runtime specialization
    (`portable|fast`).
  - Rust memory-mode contract options (`max-inline|inline-then-pool`) with
    configurable inline threshold metadata.
  - MLIR schema/plan metadata validation before emission (matching C++ structural checks).
- Go code generation (`dsdlc --target-language go`) with:
  - A generated module layout (`go.mod`, `uavcan/**`, `dsdlruntime/**`).
  - Per-type Go data types and inline SerDes methods.
  - A local Go runtime module (`dsdlruntime/dsdl_runtime.go`) with bit-level primitives.
  - Deterministic output and full-`uavcan` generation/build gates.
- TypeScript code generation (`dsdlc --target-language ts`) with:
  - A generated package/module layout (`package.json`, `index.ts`, namespace `*.ts` files).
  - Per-type TypeScript interface/type declarations, DSDL metadata constants, and generated runtime SerDes entrypoints.
  - A generated TypeScript runtime helper module (`dsdl_runtime.ts`) for bit-level read/write primitives.
  - Runtime specialization (`portable|fast`) for generated runtime helper implementation strategy.
  - MLIR schema/plan metadata validation before emission.
- Python 3.10 code generation (`dsdlc --target-language python`) with:
  - A generated package/module layout (package root + namespace `*.py` files).
  - Per-type Python dataclasses, DSDL metadata constants, and generated runtime-backed SerDes methods.
  - A generated pure-Python runtime helper (`_dsdl_runtime.py`) and runtime loader (`_runtime_loader.py`).
  - Optional C accelerator loading (`_dsdl_runtime_accel`) with automatic fallback to pure runtime.
  - MLIR schema/plan metadata validation before emission.
- Spec-conformant frontend semantics (single mode; no compatibility fallback mode).

## Ambitions

Support code generation from a common MLIR for
- C17
- C++20 std
- C++20 pmr
- Rust std
- Rust no-std
- typescript
- Python
- wasm

Tools
- dsdlc -> code generator
- dsdl-opt -> out-of tree LLVM plugin
- libdsdlc -> dynamic DSDL serdes for each language supported.
- dsdld -> DSDL language server

## Repository Layout

- `include/llvmdsdl`: public C++ headers.
- `lib`: frontend, semantics, IR, lowering, transforms, codegen.
- `tools/dsdlc`: CLI driver (`ast`, `mlir`, `c`, `cpp`, `rust`, `go`, `ts`, `python`).
- `tools/dsdl-opt`: MLIR pass driver for the DSDL dialect.
- `runtime/dsdl_runtime.h`: generated C runtime support header.
- `test/unit`: unit tests.
- `test/lit`: lit/FileCheck tests (enabled when `llvm-lit`/`lit` is available).
- `submodules/public_regulated_data_types`: DSDL corpus submodule.

## Prerequisites

- CMake `>= 3.24` (`>= 3.25` recommended for full preset support)
- Ninja (recommended)
- C++20 compiler
- LLVM + MLIR with CMake package config files (`LLVMConfig.cmake`, `MLIRConfig.cmake`)

Known-good local setup:

- LLVM/MLIR `21.1.8`
- C11/C++20 toolchain

## Semantic Mode

`dsdlc` runs in one semantic mode: spec-conformant Cyphal DSDL analysis.

- Non-conformant definitions are rejected with diagnostics.
- There is no compatibility fallback mode.

## Backend Semantic Boundaries (Maintainer Note)

Semantics shared across non-C backends (C++/Rust/Go/TypeScript/Python) are
generated from lowered MLIR contracts through shared codegen layers:

- `RuntimeLoweredPlan*` for ordered runtime field/section planning
- `RuntimeHelperBindings*` for lowered helper-symbol lookup/resolution
- `ScriptedBodyPlan*` for scripted-backend (TypeScript/Python) helper planning
- `NativeEmitterTraversal*` for native-backend (C++/Rust/Go) lowered-step traversal
- `NativeHelperContract*` for native-backend section/field helper-contract validation
- `CodegenDiagnosticText*` for cross-backend diagnostic text parity
- `NamingPolicy*` for shared identifier keyword/sanitize/case projection
- `ConstantLiteralRender*` for shared literal rendering across backends
- `StorageTypeTokens*` for shared scalar storage token mapping (C/C++/Rust/Go)
- `DefinitionIndex*` and `DefinitionPathProjection*` for shared type lookup and
  deterministic type/file projection
- `DefinitionDependencies*` for shared native-backend composite dependency
  collection
- `CompositeImportGraph*` for shared scripted-backend composite import collection/projection
- `CHeaderRender*` for shared C metadata macro and service wrapper rendering
- `HelperBindingNaming*` for shared lowered helper-binding identifier projection
- `LoweredFactsLookup*` for shared lowered section-facts lookup

Low-level runtime primitives remain hand-written by design in generated/runtime
helper modules (`dsdl_runtime.h/.hpp/.rs/.go/.ts`, `_dsdl_runtime.py`, and the
optional Python accelerator).

## Maintenance Utility Targets

These are manual utility targets and are not part of normal generation workflows.

```bash
# Verify formatting.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-format -j1

# Bulk rewrite formatting.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target format-source -j1

# Static analysis helpers.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-clang-tidy -j1
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-iwyu -j1

# Convergence scorecard and regression gate.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target convergence-report -j1

# Parity matrix coverage report and regression gate.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target parity-matrix-report -j1

# Malformed-input contract matrix report and regression gate.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target malformed-contract-report -j1

# LLVM source-based coverage (configure with coverage enabled first).
cmake -S . -B build/coverage -G "Ninja Multi-Config" \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DLLVMDSDL_ENABLE_LLVM_COVERAGE=ON
cmake --build build/coverage --config RelWithDebInfo --target coverage-report -j1
```

Coverage outputs:

- `build/coverage/coverage/RelWithDebInfo/summary.txt`
- `build/coverage/coverage/RelWithDebInfo/coverage.lcov`
- `build/coverage/coverage/RelWithDebInfo/html/index.html`

## Release Readiness

Use:

- `CONTRIBUTING.md` section `15. Release Checklist`

for the preflight + gate + artifact checklist used for release candidates.

## Quick Start

From repository root:

```bash
git submodule update --init --recursive

LLVM_PREFIX="$(brew --prefix llvm)"   # macOS/Homebrew example
cmake -S . -B build -G "Ninja Multi-Config" \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"

cmake --build build --config RelWithDebInfo -j
ctest --test-dir build --build-config RelWithDebInfo --output-on-failure
```

If `llvm-lit`, `lit` (from PATH), or Python `lit` is not installed, CMake will
skip lit tests and print a warning. Unit tests still run via `ctest`.

## Preset-Driven Automation

`CMakePresets.json` uses `Ninja Multi-Config` and defines a matrix on:

- environment: `dev-llvm-env`, `dev-homebrew`, `ci`
- build config: `Debug`, `RelWithDebInfo`, `Release`

List available presets:

```bash
cmake --list-presets=all
```

Canonical workflow entrypoints:

```bash
cmake --workflow --preset matrix-dev-llvm-env
cmake --workflow --preset matrix-dev-homebrew
cmake --workflow --preset matrix-ci
```

Each matrix workflow runs:

1. configure once for the selected environment
2. `Debug` build + smoke tests (exclude `integration`)
3. `RelWithDebInfo` build + full test suite
4. (`matrix-ci` only) required Python accelerator gate tests (`python-accel-required`)
5. `Release` build + smoke tests

`Release` matrix builds also produce a self-contained tool bundle by invoking
`bundle-tools-self-contained`.

Build/test presets are matrix-aligned and explicit about configuration. Examples:

```bash
cmake --build --preset build-dev-homebrew-relwithdebinfo
ctest --preset test-dev-homebrew-full-relwithdebinfo

cmake --build --preset build-dev-homebrew-release
ctest --preset test-dev-homebrew-smoke-release

# CI accelerator-required gate lane (RelWithDebInfo).
ctest --preset test-ci-python-accel-required-relwithdebinfo
```

### Preset Migration Table

| Old command | New command |
| --- | --- |
| `cmake --workflow --preset dev` | `cmake --workflow --preset matrix-dev-llvm-env` |
| `cmake --workflow --preset dev-homebrew` | `cmake --workflow --preset matrix-dev-homebrew` |
| `cmake --workflow --preset dev-llvm-env` | `cmake --workflow --preset matrix-dev-llvm-env` |
| `cmake --workflow --preset full` | `cmake --workflow --preset matrix-dev-llvm-env` |
| `cmake --workflow --preset ci` | `cmake --workflow --preset matrix-ci` |
| `cmake --build --preset build-dev-homebrew` | `cmake --build --preset build-dev-homebrew-relwithdebinfo` |
| `ctest --preset test-dev-homebrew` | `ctest --preset test-dev-homebrew-smoke-relwithdebinfo` |
| `ctest --preset test-dev-full` | `ctest --preset test-dev-llvm-env-full-relwithdebinfo` |

### Self-Contained Tool Bundles

Self-contained bundles are now tied to `Release` matrix presets (no separate
`dev-homebrew-self-contained` preset/workflow).

Expected output:

- `<build-dir>/self-contained-tools`
- `<build-dir>/self-contained-tools/MANIFEST.txt`

Platform behavior:

- macOS: uses `otool` + `install_name_tool` path rewriting.
- Linux: uses `patchelf` for RPATH and dependency rewriting and includes `ldd`
  manifest output.

Linux prerequisite:

```bash
# one example (Debian/Ubuntu)
sudo apt-get install -y patchelf
```

## CMake Generation Targets

When the `uavcan` namespace root exists (auto-detected from either
`public_regulated_dsdl/uavcan` or `submodules/public_regulated_data_types/uavcan`),
CMake provides first-class generation targets per language/profile:

- `generate-uavcan-c`
- `generate-uavcan-cpp-std`
- `generate-uavcan-cpp-pmr`
- `generate-uavcan-cpp-both`
- `generate-uavcan-rust-std`
- `generate-uavcan-rust-no-std-alloc`
- `generate-uavcan-rust-runtime-fast`
- `generate-uavcan-rust-no-std-runtime-fast`
- `generate-uavcan-go`
- `generate-uavcan-ts`
- `generate-uavcan-python`
- `stage-uavcan-python-runtime-accelerator` (optional staging; skips if accel unavailable)
- `stage-uavcan-python-runtime-accelerator-required` (fails if accel unavailable)
- `stage-uavcan-python-wheel` (optional wheel build/staging; includes accel if staged/available)
- `stage-uavcan-python-wheel-accel-required` (fails if accel is unavailable)
- `generate-uavcan-all` (aggregate)
- `generate-demo-2026-02-16` (demo bundle with logs + `DEMO.md`)

Run after configure/build:

```bash
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-c
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-cpp-std
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-cpp-pmr
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-rust-std
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-rust-no-std-alloc
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-rust-runtime-fast
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-rust-no-std-runtime-fast
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-go
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-ts
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-python
cmake --build --preset build-dev-homebrew-relwithdebinfo --target stage-uavcan-python-runtime-accelerator
cmake --build --preset build-dev-homebrew-relwithdebinfo --target stage-uavcan-python-wheel
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-uavcan-all
cmake --build --preset build-dev-homebrew-relwithdebinfo --target generate-demo-2026-02-16
```

Generated output paths are under `<build-dir>/generated/uavcan`:

- `<build-dir>/generated/uavcan/c`
- `<build-dir>/generated/uavcan/cpp-std`
- `<build-dir>/generated/uavcan/cpp-pmr`
- `<build-dir>/generated/uavcan/cpp-both`
- `<build-dir>/generated/uavcan/rust-std`
- `<build-dir>/generated/uavcan/rust-no-std-alloc`
- `<build-dir>/generated/uavcan/rust-runtime-fast`
- `<build-dir>/generated/uavcan/rust-no-std-runtime-fast`
- `<build-dir>/generated/uavcan/go`
- `<build-dir>/generated/uavcan/ts`
- `<build-dir>/generated/uavcan/python`
- `<build-dir>/generated/uavcan/python-wheel/<config>` (wheel staging output + `MANIFEST.txt`)
- `<build-dir>/demo-2026-02-16` (demo artifact bundle)

Notes:

- `generate-uavcan-c` emits both per-type headers and per-definition `.c`
  implementation translation units under namespace folders
  (for example `uavcan/node/Heartbeat_1_0.c`) and compile-check all generated
  `.c` files with `-std=c11 -Wall -Wextra -Werror`.

## CLI Usage

Using matrix presets, build `RelWithDebInfo` first and set:

```bash
DSDLC=./build/matrix/dev-homebrew/tools/dsdlc/RelWithDebInfo/dsdlc
```

### AST dump

```bash
"${DSDLC}" --target-language ast \
  submodules/public_regulated_data_types/uavcan
```

### MLIR output

```bash
"${DSDLC}" --target-language mlir \
  submodules/public_regulated_data_types/uavcan
```

### C header generation

```bash
"${DSDLC}" --target-language c \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-out
```

Optional:

- `--optimize-lowered-serdes`: enable optional semantics-preserving MLIR
  optimization on lowered SerDes IR before backend emission.
- No additional C mode flags are required: `dsdlc --target-language c` always emits headers and
  per-definition implementation translation units.

### C++23 header generation (`std`/`pmr`)

Generate both profiles:

```bash
"${DSDLC}" --target-language cpp \
  submodules/public_regulated_data_types/uavcan \
  --cpp-profile both \
  --outdir build/uavcan-cpp-out
```

Generate only one profile:

```bash
"${DSDLC}" --target-language cpp \
  submodules/public_regulated_data_types/uavcan \
  --cpp-profile std \
  --outdir build/uavcan-cpp-std-out
```

```bash
"${DSDLC}" --target-language cpp \
  submodules/public_regulated_data_types/uavcan \
  --cpp-profile pmr \
  --outdir build/uavcan-cpp-pmr-out
```

Profile behavior:

- `std`: generated types use `std::array` for fixed arrays and `std::vector`
  for variable-length arrays.
- `pmr`: same wire behavior, but variable-length arrays use `std::pmr::vector`
  and allocator handoff is available through
  `llvmdsdl::cpp::MemoryResource*` on generated types/SerDes entry points.
- `both`: emits two trees under `<out>/std` and `<out>/pmr`.

Naming style:

- Generated C++ types are always inside C++ namespaces derived from DSDL
  namespaces.
- Example: `uavcan__node__Heartbeat` is emitted as `uavcan::node::Heartbeat`.
- When multiple versions of the same type coexist in one namespace, version
  suffixes are added to keep names unique (e.g. `uavcan::file::Path_1_0`,
  `uavcan::file::Path_2_0`).

### Rust crate generation (`std`/`no-std-alloc` + runtime specialization + memory mode)

```bash
"${DSDLC}" --target-language rust \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-rust-out \
  --rust-crate-name uavcan_dsdl_generated \
  --rust-profile std
```

No-std+alloc profile:

```bash
"${DSDLC}" --target-language rust \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-rust-no-std-out \
  --rust-crate-name uavcan_dsdl_generated_no_std \
  --rust-profile no-std-alloc
```

Runtime-specialized std profile:

```bash
"${DSDLC}" --target-language rust \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-rust-fast-out \
  --rust-crate-name uavcan_dsdl_generated_fast \
  --rust-profile std \
  --rust-runtime-specialization fast
```

No-std+alloc profile with explicit memory mode contracts:

```bash
"${DSDLC}" --target-language rust \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-rust-no-std-inline-out \
  --rust-crate-name uavcan_dsdl_generated_no_std_inline \
  --rust-profile no-std-alloc \
  --rust-memory-mode max-inline
```

```bash
"${DSDLC}" --target-language rust \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-rust-no-std-pool-out \
  --rust-crate-name uavcan_dsdl_generated_no_std_pool \
  --rust-profile no-std-alloc \
  --rust-memory-mode inline-then-pool \
  --rust-inline-threshold-bytes 512
```

Current behavior:

- `--rust-profile std` is the default and recommended first path.
- `--rust-profile no-std-alloc` emits a crate configured for `no_std` with
  `alloc` (Cargo default features are empty; `std` remains an opt-in feature).
- `--rust-runtime-specialization portable` is the default profile behavior.
- `--rust-runtime-specialization fast` enables a runtime-optimized bit-copy path
  via Cargo feature `runtime-fast` while keeping generated semantic type files
  unchanged.
- `--rust-memory-mode max-inline` is the default in both `std` and
  `no-std-alloc` profiles.
- `--rust-memory-mode inline-then-pool` enables a contract where
  variable-length payloads remain inline up to
  `--rust-inline-threshold-bytes` and use per-type pools above that threshold.
- `--rust-inline-threshold-bytes` must be a positive integer (default `256`).
- Runtime-specialization integration gates include C/Rust parity verification for
  both `std + fast` and `no-std-alloc + fast` profile combinations.
- Generated Rust API uses `DsdlVec` aliasing in `dsdl_runtime` so profile
  changes do not alter lowered wire semantics.
- `DsdlVec` is now backed by `VarArray<T>` in the runtime module, which preserves
  Vec-compatible generated API usage while carrying memory-contract metadata for
  `max-inline` and `inline-then-pool` evolution.
- Generated Rust array fields now emit per-field pool-class constants, install
  memory contracts during decode, and route allocation planning through
  `reserve_with_pool(...)` in `inline-then-pool` mode while preserving the
  existing serialize/deserialize method signatures.
- Generated `Cargo.toml` now records the selected Rust profile, runtime
  specialization, memory mode, and inline-threshold value under
  `[package.metadata.llvmdsdl]`.
- Allocation-failure taxonomy contract (stabilized in Workstream A):
  - malformed/truncated wire input remains distinct from allocation errors;
  - max-inline mode does not depend on pool allocation paths;
  - inline-then-pool mode surfaces deterministic allocation-failure diagnostics
    with type-class context and stable runtime codes:
    - `DSDL_RUNTIME_ERROR_ALLOCATION_OUT_OF_MEMORY` (`13`)
    - `DSDL_RUNTIME_ERROR_ALLOCATION_POOL_UNAVAILABLE` (`14`)
    - `DSDL_RUNTIME_ERROR_ALLOCATION_INVALID_REQUEST` (`15`)
  - runtime troubleshooting text is available via
    `allocation_error_hint(AllocationErrorKind)`.
- Generated Rust types expose both:
- `deserialize(&mut self, &[u8]) -> Result<usize, i8>` (ergonomic path), and
- `deserialize_with_consumed(&mut self, &[u8]) -> (i8, usize)` (C-like parity path that reports consumed bytes on error too).

Rust runtime memory-mode benchmark (artifact-first, optional threshold gating):

```bash
# Runs generated Rust benchmark code for max-inline and inline-then-pool modes.
ctest --test-dir build/matrix/dev-homebrew -C RelWithDebInfo \
  -R llvmdsdl-fixtures-rust-runtime-bench --output-on-failure

# Artifacts:
# - build/matrix/dev-homebrew/test/integration/fixtures-rust-runtime-bench-out/rust-runtime-bench.json
# - build/matrix/dev-homebrew/test/integration/fixtures-rust-runtime-bench-out/rust-runtime-bench.txt
#
# Per-mode raw outputs:
# - .../fixtures-rust-runtime-bench-out/max-inline/rust-runtime-bench-max-inline.json
# - .../fixtures-rust-runtime-bench-out/inline-then-pool/rust-runtime-bench-inline-then-pool.json
```

```bash
# Optional threshold gates (off by default).
cmake --preset dev-homebrew \
  -DLLVMDSDL_RUST_RUNTIME_BENCH_ENABLE_THRESHOLDS=ON \
  -DLLVMDSDL_RUST_RUNTIME_BENCH_THRESHOLDS_JSON=$PWD/test/integration/rust_runtime_bench_thresholds.json
ctest --test-dir build/matrix/dev-homebrew -C RelWithDebInfo \
  -R llvmdsdl-fixtures-rust-runtime-bench --output-on-failure
```

Embedded deployment guidance from the benchmark lane:

- safety-critical deterministic profile: use `max-inline`.
- memory-constrained profile with bounded pools: use `inline-then-pool` and tune
  `--rust-inline-threshold-bytes` from measured payload families.
- host-throughput profile: keep `max-inline` unless local benchmark results show
  `inline-then-pool` parity for your representative data.

### TypeScript module generation (non-C-like target)

```bash
"${DSDLC}" --target-language ts \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-ts-out \
  --ts-module uavcan_dsdl_generated_ts
```

Runtime-specialized fast profile:

```bash
"${DSDLC}" --target-language ts \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-ts-fast-out \
  --ts-module uavcan_dsdl_generated_ts_fast \
  --ts-runtime-specialization fast
```

Current behavior:

- Emits `package.json`, `index.ts`, one namespace-mirrored `*.ts` file per DSDL
  definition, and shared runtime module `dsdl_runtime.ts`.
- `index.ts` exports each generated definition module under a collision-safe
  namespace alias.
- Generates interface/type declarations, DSDL metadata constants
  (`DSDL_FULL_NAME`, version major/minor), and runtime-backed
  `serialize*`/`deserialize*` helpers.
- `--ts-runtime-specialization portable` is the default profile behavior.
- `--ts-runtime-specialization fast` enables byte-aligned fast paths in
  `dsdl_runtime.ts` while preserving semantic generated type files.
- Runtime-specialization integration gates include generation, typecheck,
  C<->TS parity, and semantic-diff checks between portable and fast outputs.
- Uses lowered-contract validation (`collectLoweredFactsFromMlir`) plus shared
  runtime planning/binding layers (`RuntimeLoweredPlan`, `RuntimeHelperBindings`,
  `ScriptedBodyPlan`) for runtime section emission.
- Emits lowered-helper bindings (`mlir_*`) and helper-driven call sites for
  scalar normalization/sign-extension, array prefix/length validation, union
  tag normalization/validation, delimiter checks, and section capacity checks.
- Keeps low-level bit/float/buffer primitives in generated `dsdl_runtime.ts`
  by design.
- Hard-fails generation if required lowered runtime planning metadata is
  missing/inconsistent.
- Integration gates verify no fallback runtime stub signatures in generated
  fixture and full-`uavcan` TypeScript output.
- Integration coverage includes:
  - full-`uavcan` generation/determinism/typecheck/consumer-smoke/index-contract/runtime-execution gates
  - runtime smoke lanes across scalar/array/union/composite/delimited/service/padding/truncated-decode families
  - C<->TS parity lanes including invariant-based random+direct checks, signed-narrow cast-mode checks, and optimized-lowering variants

### Python module generation (`python3.10` + optional accelerator)

```bash
"${DSDLC}" --target-language python \
  submodules/public_regulated_data_types/uavcan \
  --outdir build/uavcan-python-out \
  --py-package uavcan_dsdl_generated_py
```

Current behavior:

- Emits one namespace-mirrored `*.py` file per DSDL definition plus package `__init__.py` files.
- Generates per-type dataclasses with:
  - `serialize(self) -> bytes`
  - `deserialize(cls, data: bytes | bytearray | memoryview) -> Type`
- Emits generated runtime modules in the package root:
  - `_dsdl_runtime.py` (pure Python runtime)
  - `_runtime_loader.py` (selects `auto|pure|accel` via `LLVMDSDL_PY_RUNTIME_MODE`)
- Emits packaging metadata:
  - `pyproject.toml` (minimal local installability)
  - `py.typed` (typing marker)
- Uses lowered-contract validation (`collectLoweredFactsFromMlir`) plus shared
  runtime planning/binding layers (`RuntimeLoweredPlan`, `RuntimeHelperBindings`,
  `ScriptedBodyPlan`) for SerDes emission.
- Emits lowered-helper bindings (`mlir_*`) and helper-driven call sites for
  scalar normalization/sign-extension, array prefix/length validation, union
  tag normalization/validation, delimiter checks, and section capacity checks.
- Keeps low-level bit/float/buffer primitives in `_dsdl_runtime.py`
  (and optional accelerator implementation) by design.
- Optional C accelerator module support (`_dsdl_runtime_accel`) is available via CMake option
  `LLVMDSDL_ENABLE_PYTHON_ACCELERATOR=ON`.
- Explicit malformed-input contract:
  - `portable` pure runtime: tolerant out-of-range reads (missing bits are zero-extended).
  - `fast` pure runtime: byte-aligned out-of-range `extract_bits`/`copy_bits` raises `ValueError`.
  - `accel` runtime: follows accelerator helper behavior (currently tolerant `extract_bits`,
    strict range checks in `copy_bits`).

Pure-only install/run workflow:

```bash
OUT_PY="build/uavcan-python-out"
"${DSDLC}" --target-language python \
  submodules/public_regulated_data_types/uavcan \
  --outdir "${OUT_PY}" \
  --py-package uavcan_dsdl_generated_py

python3 -m venv "${OUT_PY}/.venv"
"${OUT_PY}/.venv/bin/pip" install -e "${OUT_PY}"

LLVMDSDL_PY_RUNTIME_MODE=pure \
  "${OUT_PY}/.venv/bin/python" -c \
  "from uavcan_dsdl_generated_py.uavcan.node.heartbeat_1_0 import Heartbeat_1_0; print(Heartbeat_1_0)"
```

Accel-enabled install/run workflow:

```bash
# Configure once with accelerator enabled.
cmake -S . -B build/matrix/dev-homebrew -G "Ninja Multi-Config" \
  -DLLVMDSDL_ENABLE_PYTHON_ACCELERATOR=ON

# Generate package and stage accelerator beside the generated package.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target generate-uavcan-python
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target stage-uavcan-python-runtime-accelerator-required

OUT_PY="build/matrix/dev-homebrew/generated/uavcan/python"
python3 -m venv "${OUT_PY}/.venv"
"${OUT_PY}/.venv/bin/pip" install -e "${OUT_PY}"

LLVMDSDL_PY_RUNTIME_MODE=accel \
  "${OUT_PY}/.venv/bin/python" -c \
  "from uavcan_dsdl_generated_py import _runtime_loader as rl; print(rl.BACKEND)"
```

Optional wheel build/staging workflow:

```bash
# Build pure-or-accel wheel opportunistically.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target stage-uavcan-python-wheel

# Require accelerator presence and stage a wheel with accelerator payload.
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target stage-uavcan-python-wheel-accel-required

# Artifacts:
# - build/matrix/dev-homebrew/generated/uavcan/python-wheel/RelWithDebInfo/*.whl
# - build/matrix/dev-homebrew/generated/uavcan/python-wheel/RelWithDebInfo/MANIFEST.txt
```

Python runtime benchmark (artifact-first, optional threshold gating):

```bash
# Run benchmark integration lane (always emits JSON/text artifacts).
ctest --test-dir build/matrix/dev-homebrew -C RelWithDebInfo \
  -R llvmdsdl-uavcan-python-runtime-bench --output-on-failure

# Artifacts:
# - build/matrix/dev-homebrew/test/integration/uavcan-python-runtime-bench-out/python-runtime-bench.json
# - build/matrix/dev-homebrew/test/integration/uavcan-python-runtime-bench-out/python-runtime-bench.txt

# Optional threshold gates (off by default).
cmake --preset dev-homebrew \
  -DLLVMDSDL_PY_RUNTIME_BENCH_ENABLE_THRESHOLDS=ON \
  -DLLVMDSDL_PY_RUNTIME_BENCH_THRESHOLDS_JSON=$PWD/test/integration/python_runtime_bench_thresholds.json
ctest --test-dir build/matrix/dev-homebrew -C RelWithDebInfo \
  -R llvmdsdl-uavcan-python-runtime-bench --output-on-failure
```

Python parity and validation taxonomy:

- Fixture generation hardening:
  - `llvmdsdl-fixtures-python-generation-hardening`
- Runtime smoke:
  - `llvmdsdl-fixtures-python-runtime-smoke`
  - `llvmdsdl-fixtures-python-runtime-smoke-optimized`
  - `llvmdsdl-fixtures-python-runtime-smoke-fast`
- Backend-selection behavior:
  - `llvmdsdl-fixtures-python-runtime-backend-selection`
  - `llvmdsdl-fixtures-python-runtime-backend-selection-accel-required`
  - `llvmdsdl-fixtures-python-malformed-input-contract`
  - `llvmdsdl-fixtures-python-malformed-decode-fuzz-parity`
  - `llvmdsdl-fixtures-python-malformed-decode-fuzz-parity-accel-required`
  - `llvmdsdl-fixtures-python-runtime-parity`
  - `llvmdsdl-fixtures-python-runtime-parity-accel-required`
- C<->Python differential parity families:
  - `llvmdsdl-fixtures-c-python-*-parity`
  - `llvmdsdl-fixtures-c-python-*-parity-optimized`
  - `llvmdsdl-signed-narrow-c-python-parity`
  - `llvmdsdl-signed-narrow-c-python-parity-optimized`
  - `llvmdsdl-signed-narrow-c-python-parity-runtime-fast`
- Full-tree `uavcan` lanes:
  - `llvmdsdl-uavcan-python-generation`
  - `llvmdsdl-uavcan-python-generation-runtime-fast`
  - `llvmdsdl-uavcan-python-runtime-specialization-diff`
  - `llvmdsdl-uavcan-python-determinism`
  - `llvmdsdl-uavcan-python-runtime-execution`
  - `llvmdsdl-uavcan-python-runtime-execution-runtime-fast`
  - `llvmdsdl-uavcan-python-runtime-execution-optimized`
  - `llvmdsdl-uavcan-python-runtime-bench`

Python troubleshooting matrix:

| Symptom | Typical cause | Resolution |
| --- | --- | --- |
| `LLVMDSDL_PY_RUNTIME_MODE=accel` fails with `_dsdl_runtime_accel` import error | Accelerator not built or not staged beside generated package | Configure with `-DLLVMDSDL_ENABLE_PYTHON_ACCELERATOR=ON`, build, then run `stage-uavcan-python-runtime-accelerator-required` |
| `LLVMDSDL_PY_RUNTIME_MODE=auto` reports `pure` backend unexpectedly | Auto mode falls back when accelerator is unavailable | Check `_runtime_loader.py` backend printout and stage accelerator if accel is required |
| `pip install -e <out-dir>` fails | Missing generated `pyproject.toml` due to stale output or wrong `--outdir` | Regenerate with `dsdlc --target-language python` and confirm `<out-dir>/pyproject.toml` exists |
| Specialization diff lane fails | Unexpected semantic drift between `portable` and `fast` runtime specializations | Run `llvmdsdl-uavcan-python-runtime-specialization-diff` and inspect generated `_dsdl_runtime.py` differences only |
| Bench lane fails when thresholds enabled | Missing or overly strict thresholds file | Start from `test/integration/python_runtime_bench_thresholds.json`, calibrate on reference hardware, then enable thresholds |

## Reproducible Full `uavcan` Generation Check

```bash
OUT="build/uavcan-out"
mkdir -p "${OUT}"

"${DSDLC}" --target-language c \
  submodules/public_regulated_data_types/uavcan \
  --outdir "${OUT}"

find submodules/public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find "${OUT}" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

The two counts should match.

## Status Snapshot

Current milestone supports generating all types under:

- `submodules/public_regulated_data_types/uavcan`

with no `dsdl_runtime_stub_*` references in generated headers, C++23 generation
(`std` + `pmr` profiles), Rust generation in `std`, `no-std-alloc`, and
runtime-specialized (`fast`) modes, plus TypeScript generation with compile
(`tsc --noEmit`), determinism, consumer-smoke, index-contract, runtime
execution, runtime-specialized (`portable|fast`) generation/typecheck/parity/
semantic-diff gates, and fixture/runtime parity validation lanes (including
signed-narrow parity, optimized parity, variable-array/bigint/union/composite/
service/delimited families), and Python generation with runtime smoke,
specialization (`portable|fast`), backend selection (`auto|pure|accel`),
packaging metadata emission (`pyproject.toml`, `py.typed`), accelerator staging
targets, full-tree generation/determinism, and runtime benchmark harness
coverage, plus Rust runtime memory-mode benchmark artifacts (`max-inline` vs
`inline-then-pool`) for small/medium/large payload families.

## Codegen Throughput Benchmarking

For large-scale throughput regression testing, use the complex benchmark corpus:

- root namespace: `test/benchmark/complex/civildrone`
- lookup namespace: `test/benchmark/complex/uavcan`

Utility targets:

- `benchmark-codegen-record`
- `benchmark-codegen-init-thresholds`
- `benchmark-codegen-check-dev-ab`
- `benchmark-codegen-check-ci-oom`

Optional per-language CI-oom CTest lanes (enabled with
`-DLLVMDSDL_ENABLE_BENCHMARK_TESTS=ON` after thresholds are calibrated):

- `llvmdsdl-codegen-benchmark-ci-oom-c`
- `llvmdsdl-codegen-benchmark-ci-oom-cpp`
- `llvmdsdl-codegen-benchmark-ci-oom-rust`
- `llvmdsdl-codegen-benchmark-ci-oom-go`
- `llvmdsdl-codegen-benchmark-ci-oom-ts`
- `llvmdsdl-codegen-benchmark-ci-oom-python`

Typical flow:

```bash
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target benchmark-codegen-record -j1
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target benchmark-codegen-init-thresholds -j1
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target benchmark-codegen-check-ci-oom -j1
```

The committed threshold file (`test/benchmark/complex_codegen_thresholds.json`)
is a template until calibrated. Generate thresholds from a record run and update
the file for your CI/hardware class.
