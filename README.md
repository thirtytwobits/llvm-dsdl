# llvm-dsdl

`llvm-dsdl` is an out-of-tree LLVM/MLIR-based DSDL compiler for Cyphal data
types.

It currently provides:

- A DSDL frontend (`.dsdl` discovery, parse, semantic analysis).
- A custom MLIR DSDL dialect and lowering pipeline hooks.
- C11 code generation (`dsdlc c`) with:
  - Per-type headers mirroring namespace directories.
  - Declarations/wrappers in headers plus per-definition `.c` SerDes implementations generated via MLIR/EmitC lowering.
  - A local header-only runtime (`dsdl_runtime.h`) with bit-level primitives.
- C++23 code generation (`dsdlc cpp`) with:
  - Per-type headers (`.hpp`) mirroring namespace directories.
  - Header-only inline SerDes for both `std` and allocator-oriented `pmr` profiles.
  - A C++ runtime wrapper (`dsdl_runtime.hpp`) over the core bit-level runtime.
  - MLIR schema/plan metadata validation before emission (fail-fast on malformed IR facts).
- Rust code generation (`dsdlc rust`) with:
  - A generated crate layout (`Cargo.toml`, `src/lib.rs`, `src/**`).
  - Per-type Rust data types and inline SerDes methods.
  - A local Rust runtime module (`src/dsdl_runtime.rs`) with bit-level primitives.
  - `std` profile enabled now, with an explicit reserved seam for future
    `no_std + alloc`.
  - MLIR schema/plan metadata validation before emission (matching C++ structural checks).
- Strict-mode-first semantics (`--strict` is default).

## Repository Layout

- `include/llvmdsdl`: public C++ headers.
- `lib`: frontend, semantics, IR, lowering, transforms, codegen.
- `tools/dsdlc`: CLI driver (`ast`, `mlir`, `c`, `cpp`, `rust`).
- `tools/dsdl-opt`: MLIR pass driver for the DSDL dialect.
- `runtime/dsdl_runtime.h`: generated C runtime support header.
- `test/unit`: unit tests.
- `test/lit`: lit/FileCheck tests (enabled when `llvm-lit`/`lit` is available).
- `public_regulated_data_types`: DSDL corpus submodule.

## Prerequisites

- CMake `>= 3.24`
- Ninja (recommended)
- C++20 compiler
- LLVM + MLIR with CMake package config files (`LLVMConfig.cmake`, `MLIRConfig.cmake`)

Known-good local setup:

- LLVM/MLIR `20.1.8`
- C11/C++20 toolchain

## Quick Start

From repository root:

```bash
git submodule update --init --recursive

LLVM_PREFIX="$(brew --prefix llvm)"   # macOS/Homebrew example
cmake -S . -B build -G Ninja \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"

cmake --build build -j
ctest --test-dir build --output-on-failure
```

If `llvm-lit` or Python `lit` is not installed, CMake will skip lit tests and
print a warning. Unit tests still run via `ctest`.

## Preset-Driven Automation

`CMakePresets.json` is configured with `configure`, `build`, `test`, and
`workflow` presets so contributors can run the common pipelines with one command.

List available presets:

```bash
cmake --list-presets=all
```

Fast dev workflow (configure + build + fast tests):

```bash
cmake --workflow --preset dev
```

Full verification workflow (includes strict `uavcan` C, C++, and Rust integration tests):

```bash
cmake --workflow --preset full
```

The full test set now also includes `llvmdsdl-uavcan-mlir-lowering`, which
validates full-`uavcan` MLIR lowering and `convert-dsdl-to-emitc` pass
execution under `dsdl-opt`.

It also includes `llvmdsdl-uavcan-cpp-c-parity`, a cross-backend integration
gate that compares generated C and generated C++ (`std` profile) SerDes
behavior over randomized inputs for representative types.

It also includes `llvmdsdl-uavcan-cpp-pmr-c-parity`, which runs the same
cross-backend parity harness against the generated C++ `pmr` profile to verify
allocator-oriented API paths preserve the same wire behavior.

With Rust tooling available, the suite also enables
`llvmdsdl-uavcan-rust-cargo-check` to compile-check the full generated `uavcan`
Rust crate using `cargo check`.

When Rust tooling is available (`cargo` + `rustc`), `ctest` also enables
`llvmdsdl-uavcan-c-rust-parity`, which links generated C implementation units
into a Rust harness and checks deserialize/serialize return codes, consumed and
produced sizes, and output bytes for representative `uavcan` types.

Demo artifact workflow (creates `build/<preset>/demo-2026-02-16/` with
generated outputs, full-`uavcan` MLIR + lowered MLIR snapshots, selected test
logs, and `DEMO.md`):

```bash
cmake --workflow --preset demo
```

On macOS with Homebrew LLVM:

```bash
cmake --workflow --preset demo-homebrew
```

Differential parity workflow note:

- If sibling repositories `../nunavut` and `../pydsdl` are present and Python 3
  is available, `ctest` auto-enables `llvmdsdl-differential-parity`.
- This test generates reference C with Nunavut for representative `uavcan`
  types, then compares deserialize/serialize behavior against `llvm-dsdl`.
- The current differential gate enforces byte parity for non-float-focused
  cases and return-code/size parity for float-involved cases.

Run only the differential parity test:

```bash
ctest --test-dir build -R llvmdsdl-differential-parity --output-on-failure
```

Run only the C/Rust parity test (when `cargo` and `rustc` are available):

```bash
ctest --test-dir build -R llvmdsdl-uavcan-c-rust-parity --output-on-failure
```

Run only the C/C++ PMR parity test:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-cpp-pmr-c-parity --output-on-failure
```

Run only the generated-Rust compile gate:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-rust-cargo-check --output-on-failure
```

Run only strict `uavcan` integration validation:

```bash
cmake --workflow --preset uavcan
```

macOS Homebrew LLVM workflow:

```bash
cmake --workflow --preset dev-homebrew
```

Environment-driven LLVM workflow:

```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --workflow --preset dev-llvm-env
```

## CMake Generation Targets

When the `uavcan` namespace root exists (auto-detected from either
`public_regulated_dsdl/uavcan` or `public_regulated_data_types/uavcan`),
CMake provides first-class generation targets per language/profile:

- `generate-uavcan-c`
- `generate-uavcan-cpp-std`
- `generate-uavcan-cpp-pmr`
- `generate-uavcan-cpp-both`
- `generate-uavcan-rust-std`
- `generate-uavcan-all` (aggregate)
- `generate-demo-2026-02-16` (demo bundle with logs + `DEMO.md`)

Run after configure/build:

```bash
cmake --build --preset build-dev-homebrew --target generate-uavcan-c
cmake --build --preset build-dev-homebrew --target generate-uavcan-cpp-std
cmake --build --preset build-dev-homebrew --target generate-uavcan-cpp-pmr
cmake --build --preset build-dev-homebrew --target generate-uavcan-rust-std
cmake --build --preset build-dev-homebrew --target generate-uavcan-all
cmake --build --preset build-dev-homebrew --target generate-demo-2026-02-16
```

Generated output paths are under `<build-dir>/generated/uavcan`:

- `<build-dir>/generated/uavcan/c`
- `<build-dir>/generated/uavcan/cpp-std`
- `<build-dir>/generated/uavcan/cpp-pmr`
- `<build-dir>/generated/uavcan/cpp-both`
- `<build-dir>/generated/uavcan/rust-std`
- `<build-dir>/demo-2026-02-16` (demo artifact bundle)

Notes:

- `generate-uavcan-c` emits both per-type headers and per-definition `.c`
  implementation translation units under namespace folders
  (for example `uavcan/node/Heartbeat_1_0.c`) and compile-check all generated
  `.c` files with `-std=c11 -Wall -Wextra -Werror`.

## CLI Usage

### AST dump

```bash
./build/tools/dsdlc/dsdlc ast \
  --root-namespace-dir public_regulated_data_types/uavcan
```

### MLIR output

```bash
./build/tools/dsdlc/dsdlc mlir \
  --root-namespace-dir public_regulated_data_types/uavcan
```

### C header generation (strict by default)

```bash
./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --out-dir build/uavcan-out
```

Optional:

- `--compat-mode`: relax strictness for compatibility behavior.
- No additional C mode flags are required: `dsdlc c` always emits headers and
  per-definition implementation translation units.

### C++23 header generation (`std`/`pmr`)

Generate both profiles:

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --cpp-profile both \
  --out-dir build/uavcan-cpp-out
```

Generate only one profile:

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --cpp-profile std \
  --out-dir build/uavcan-cpp-std-out
```

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --cpp-profile pmr \
  --out-dir build/uavcan-cpp-pmr-out
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

### Rust crate generation (`std` profile)

```bash
./build/tools/dsdlc/dsdlc rust \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --out-dir build/uavcan-rust-out \
  --rust-crate-name uavcan_dsdl_generated \
  --rust-profile std
```

Current behavior:

- `--rust-profile std` is the default and recommended first path.
- `--rust-profile no-std-alloc` is reserved and currently returns a clear
  "not implemented yet" error.
- Generated Rust API uses `DsdlVec` aliasing in `dsdl_runtime` so we can switch
  allocation strategy in the planned embedded-focused backend.
- Generated Rust types expose both:
- `deserialize(&mut self, &[u8]) -> Result<usize, i8>` (ergonomic path), and
- `deserialize_with_consumed(&mut self, &[u8]) -> (i8, usize)` (C-like parity path that reports consumed bytes on error too).

## Reproducible Full `uavcan` Generation Check

```bash
OUT="build/uavcan-out-strict"
mkdir -p "${OUT}"

./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --out-dir "${OUT}"

find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find "${OUT}" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

The two counts should match.

## Status Snapshot

Current milestone supports generating all types under:

- `public_regulated_data_types/uavcan`

with strict mode enabled and no `dsdl_runtime_stub_*` references in generated
headers, strict C++23 generation (`std` + `pmr` profiles), plus strict Rust
crate generation in `std` mode.
