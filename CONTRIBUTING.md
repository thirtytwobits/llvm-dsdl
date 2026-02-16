# Contributing to llvm-dsdl

This document is a reproducibility-first guide for contributors. It is designed
to let a new contributor go from a clean checkout to a verified build and strict
`uavcan` generation with deterministic checks.

## Table of Contents

1. Scope
2. Prerequisites
3. Clean Checkout
4. Configure (CMake + LLVM/MLIR)
5. Build
6. Test
7. Reproduce Strict `uavcan` Generation
8. Validate Generated Output
9. Common Workflows
10. Troubleshooting
11. Commit and PR Expectations

## 1. Scope

This guide covers:

- Building `dsdlc` and `dsdl-opt`.
- Running unit tests and lit tests (when available).
- Generating C11 output from `public_regulated_data_types/uavcan`.
- Verifying generation completeness and compile sanity.

## 2. Prerequisites

Minimum:

- CMake `>= 3.24`
- Ninja (recommended generator)
- A C++20 compiler
- LLVM + MLIR development packages discoverable by CMake
  (`LLVMConfig.cmake`, `MLIRConfig.cmake`)
- Git

Optional but recommended:

- Python 3 with `lit` module, or `llvm-lit` in `PATH` (for lit tests)

Known-good local toolchain:

- LLVM/MLIR `20.1.8`
- C11/C++20-capable compiler

## 3. Clean Checkout

From a new directory:

```bash
git clone <your-fork-or-origin-url> llvm-dsdl
cd llvm-dsdl
git submodule update --init --recursive
```

`public_regulated_data_types` is required for the strict full-tree generation
checks in this document.

## 4. Configure (CMake + LLVM/MLIR)

### 4.1 macOS/Homebrew example

```bash
LLVM_PREFIX="$(brew --prefix llvm)"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
```

### 4.2 Generic Linux example

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

### 4.3 Expected configure behavior

- CMake should report LLVM and MLIR config paths.
- If lit tooling is missing, you will see:
  - `Skipping lit tests because llvm-lit/lit is unavailable`
- This is not fatal; unit tests still run.

## 5. Build

```bash
cmake --build build -j
```

Expected artifacts:

- `build/tools/dsdlc/dsdlc`
- `build/tools/dsdl-opt/dsdl-opt`
- static libs under `build/lib/...`

## 6. Test

### 6.1 Unit tests

```bash
ctest --test-dir build --output-on-failure
```

### 6.2 lit tests (if lit is available)

When lit is configured, `ctest` includes `llvmdsdl-lit`.

You can also run lit directly:

```bash
llvm-lit -sv build/test/lit
```

or with Python module:

```bash
python3 -m lit.main -sv build/test/lit
```

## 7. Reproduce Strict `uavcan` Generation

This is the primary end-to-end check for current project status.

```bash
OUT="build/uavcan-out-strict-verify"
mkdir -p "${OUT}"

./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --out-dir "${OUT}"
```

Expected result:

- Exit code `0`
- Per-type headers generated under `${OUT}/uavcan/...`
- `${OUT}/dsdl_runtime.h` emitted

### 7.1 Optional impl translation unit

```bash
OUT_IMPL="build/uavcan-out-with-impl"
mkdir -p "${OUT_IMPL}"

./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --emit-impl-tu \
  --out-dir "${OUT_IMPL}"
```

Expected extra artifact:

- `${OUT_IMPL}/generated_impl.c`

## 8. Validate Generated Output

### 8.1 Header count parity

```bash
find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-out-strict-verify -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

Expected:

- Counts are equal.

### 8.2 Ensure stubs are gone

```bash
rg -n "dsdl_runtime_stub_" build/uavcan-out-strict-verify
```

Expected:

- No matches.

### 8.3 Compile-check all generated headers as C11, warning-clean

```bash
outdir="build/uavcan-out-strict-verify"
scratch="$(mktemp -d)"
rc=0

for h in $(find "${outdir}" -name '*.h' ! -name 'dsdl_runtime.h' | sort); do
  rel="${h#${outdir}/}"
  cat > "${scratch}/tu.c" <<EOF
#include "${rel}"
int main(void) { return 0; }
EOF
  cc -std=c11 -Wall -Wextra -Werror -I"${outdir}" "${scratch}/tu.c" -c -o "${scratch}/tu.o" || { echo "FAILED:${rel}"; rc=1; break; }
done

exit ${rc}
```

Expected:

- Exit code `0`

## 9. Common Workflows

### 9.1 Reconfigure after dependency changes

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=... \
  -DMLIR_DIR=...
```

### 9.2 Fast rebuild of one target

```bash
cmake --build build --target dsdlc -j
```

### 9.3 Inspect CLI options

```bash
./build/tools/dsdlc/dsdlc
./build/tools/dsdl-opt/dsdl-opt --help
```

## 10. Troubleshooting

### 10.1 `Could not find LLVMConfig.cmake` / `MLIRConfig.cmake`

- Pass explicit `-DLLVM_DIR` and `-DMLIR_DIR`.
- Confirm paths contain the corresponding config files.

### 10.2 lit tests not running

- Install `llvm-lit` or Python `lit`.
- Re-run CMake configure to pick it up.

### 10.3 `dsdlc` strict generation fails

Checklist:

- Submodule initialized:
  - `git submodule update --init --recursive`
- Command points at the root namespace:
  - `--root-namespace-dir public_regulated_data_types/uavcan`
- Use strict explicitly while diagnosing:
  - `--strict`

### 10.4 Header compile-check failures

- Ensure include root is the generation root:
  - `-I build/uavcan-out-strict-verify`
- Confirm `dsdl_runtime.h` exists in output root.

## 11. Commit and PR Expectations

Please include in PR description:

- Exact configure command used.
- Exact build/test commands run.
- Strict generation command used.
- Validation outputs:
  - DSDL/header count parity
  - stub scan result
  - C11 compile-check result

For behavior changes affecting wire semantics, include:

- A focused unit/integration test.
- A short rationale referencing affected DSDL rules.

For substantial codegen/frontend changes, also include:

- One representative generated header diff snippet (before/after).
- Any compatibility notes (`--strict` vs `--compat-mode`).

