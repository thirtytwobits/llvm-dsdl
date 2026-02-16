# llvm-dsdl

`llvm-dsdl` is an out-of-tree LLVM/MLIR-based DSDL compiler for Cyphal data
types.

It currently provides:

- A DSDL frontend (`.dsdl` discovery, parse, semantic analysis).
- A custom MLIR DSDL dialect and lowering pipeline hooks.
- C11 code generation (`dsdlc c`) with:
  - Per-type headers mirroring namespace directories.
  - Header-only, inline serialization/deserialization bodies.
  - A local header-only runtime (`dsdl_runtime.h`) with bit-level primitives.
- Strict-mode-first semantics (`--strict` is default).

## Repository Layout

- `include/llvmdsdl`: public C++ headers.
- `lib`: frontend, semantics, IR, lowering, transforms, codegen.
- `tools/dsdlc`: CLI driver (`ast`, `mlir`, `c`).
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
- `--emit-impl-tu`: emit `generated_impl.c` from the MLIR/EmitC pipeline.
- `--emit-runtime-header-only`: keep header-only runtime/codegen path enabled.

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
headers.
