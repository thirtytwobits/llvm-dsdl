# Contributing to llvm-dsdl

This document is a reproducibility-first guide for contributors. It is designed
to let a new contributor go from a clean checkout to a verified build and strict
`uavcan` generation with deterministic checks.

## Table of Contents

1. Scope
2. Prerequisites
3. Clean Checkout
4. Recommended: Preset-Driven Workflows
5. Configure (Manual CMake)
6. Build
7. Test
8. Reproduce Strict `uavcan` C Generation
9. Reproduce Strict `uavcan` C++ Generation (`std` + `pmr`)
10. Reproduce Strict `uavcan` Rust Generation (`std`)
11. Validate Generated Output
12. Common Development Tasks
13. Troubleshooting
14. Commit and PR Expectations

## 1. Scope

This guide covers:

- Building `dsdlc` and `dsdl-opt`.
- Running unit tests and lit tests (when available).
- Running integration validation for strict `uavcan` generation.
- Generating C11 output from `public_regulated_data_types/uavcan`.
- Generating C++23 output (`std`/`pmr` profiles) from
  `public_regulated_data_types/uavcan`.
- Generating Rust crate output (currently `std` profile) from
  `public_regulated_data_types/uavcan`.
- Verifying generation completeness and compile sanity.

## 2. Prerequisites

Minimum:

- CMake `>= 3.24`
- Ninja (recommended generator)
- A C++20 compiler
- LLVM + MLIR development packages discoverable by CMake
  (`LLVMConfig.cmake`, `MLIRConfig.cmake`)
- Git

For workflow presets (`cmake --workflow`), use:

- CMake `>= 3.25`

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

`public_regulated_data_types` is required for strict full-tree generation
checks and integration validation.

## 4. Recommended: Preset-Driven Workflows

This repository ships with `/Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl/CMakePresets.json` including:

- configure presets
- build presets
- test presets
- workflow presets (configure + build + test chains)

List available presets:

```bash
cmake --list-presets=all
```

### 4.1 Fast dev workflow

```bash
cmake --workflow --preset dev
```

Runs:

1. `configure` preset `dev`
2. `build` preset `build-dev`
3. `test` preset `test-dev` (fast suite; excludes integration-labeled tests)

### 4.2 Full verification workflow

```bash
cmake --workflow --preset full
```

Runs all tests, including integration validation that:

- generates all strict `uavcan` headers,
- generates strict `uavcan` C++ headers for `std` and `pmr` profiles,
- generates strict `uavcan` Rust crate output (`std` profile),
- checks count parity (`.dsdl` count == generated header count),
- checks for stub references,
- compile-checks all generated headers as C11 with `-Wall -Wextra -Werror`.

### 4.3 Run only strict `uavcan` integration validation

```bash
cmake --workflow --preset uavcan
```

### 4.4 macOS Homebrew LLVM workflow

```bash
cmake --workflow --preset dev-homebrew
```

Uses:

- `/opt/homebrew/opt/llvm/lib/cmake/llvm`
- `/opt/homebrew/opt/llvm/lib/cmake/mlir`

### 4.5 Environment-driven LLVM/MLIR workflow

```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --workflow --preset dev-llvm-env
```

## 5. Configure (Manual CMake)

Use this when you want explicit control instead of presets.

### 5.1 macOS/Homebrew example

```bash
LLVM_PREFIX="$(brew --prefix llvm)"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
```

### 5.2 Generic Linux example

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

### 5.3 Expected configure behavior

- CMake should report LLVM and MLIR config paths.
- If lit tooling is missing, you may see:
  - `Skipping lit tests because llvm-lit/lit is unavailable`
- This is not fatal; unit tests still run.

## 6. Build

Manual build:

```bash
cmake --build build -j
```

Preset build (example):

```bash
cmake --build --preset build-dev
```

Expected artifacts:

- `build/.../tools/dsdlc/dsdlc`
- `build/.../tools/dsdl-opt/dsdl-opt`
- static libs under `build/.../lib/...`

## 7. Test

### 7.1 Manual test invocation

```bash
ctest --test-dir build --output-on-failure
```

### 7.2 Preset test invocation

Fast test set:

```bash
ctest --preset test-dev
```

Full test set:

```bash
ctest --preset test-dev-full
```

Strict `uavcan` integration only:

```bash
ctest --preset test-uavcan
```

### 7.3 lit tests (if lit is available)

When lit is configured, `ctest` includes `llvmdsdl-lit`.

You can also run lit directly:

```bash
llvm-lit -sv build/test/lit
```

or with Python module:

```bash
python3 -m lit.main -sv build/test/lit
```

## 8. Reproduce Strict `uavcan` C Generation

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
- Per-definition implementation translation units generated by default under
  `${OUT}/<namespace>/`.
  Example: `${OUT}/uavcan/node/Heartbeat_1_0.c`

## 9. Reproduce Strict `uavcan` C++ Generation (`std` + `pmr`)

```bash
OUT_CPP="build/uavcan-cpp-out-strict-verify"
mkdir -p "${OUT_CPP}"

./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --cpp-profile both \
  --out-dir "${OUT_CPP}"
```

Expected result:

- Exit code `0`
- Two generated trees:
  - `${OUT_CPP}/std/...`
  - `${OUT_CPP}/pmr/...`
- Each tree contains:
  - `dsdl_runtime.h`
  - `dsdl_runtime.hpp`
  - one generated `.hpp` per input `.dsdl` definition.

Single-profile generation:

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

## 10. Reproduce Strict `uavcan` Rust Generation (`std`)

```bash
OUT_RUST="build/uavcan-rust-out-strict-verify"
mkdir -p "${OUT_RUST}"

./build/tools/dsdlc/dsdlc rust \
  --root-namespace-dir public_regulated_data_types/uavcan \
  --strict \
  --out-dir "${OUT_RUST}" \
  --rust-crate-name uavcan_dsdl_generated \
  --rust-profile std
```

Expected result:

- Exit code `0`
- Generated crate files:
  - `${OUT_RUST}/Cargo.toml`
  - `${OUT_RUST}/src/lib.rs`
  - `${OUT_RUST}/src/dsdl_runtime.rs`
- One generated Rust type file per input `.dsdl` definition.

Reserved future profile:

- `--rust-profile no-std-alloc` currently returns a not-implemented error by
  design. This is the carve-out seam for embedded/allocator-focused work.

## 11. Validate Generated Output

### 11.1 Header count parity

```bash
find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-out-strict-verify -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

Expected:

- Counts are equal.

### 11.2 Ensure stubs are gone

```bash
rg -n "dsdl_runtime_stub_" build/uavcan-out-strict-verify
```

Expected:

- No matches.

### 11.3 Compile-check all generated headers as C11, warning-clean

```bash
outdir="build/uavcan-out-strict-verify"
scratch="$(mktemp -d)"
rc=0

for h in $(find "${outdir}" -name '*.h' ! -name 'dsdl_runtime.h' | sort); do
  rel="${h#${outdir}/}"
  cat > "${scratch}/tu.c" <<EOF2
#include "${rel}"
int main(void) { return 0; }
EOF2
  cc -std=c11 -Wall -Wextra -Werror -I"${outdir}" "${scratch}/tu.c" -c -o "${scratch}/tu.o" || { echo "FAILED:${rel}"; rc=1; break; }
done

exit ${rc}
```

Expected:

- Exit code `0`

### 11.4 C++ type-file parity check

```bash
find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-cpp-out-strict-verify/std -name '*.hpp' ! -name 'dsdl_runtime.hpp' | wc -l
find build/uavcan-cpp-out-strict-verify/pmr -name '*.hpp' ! -name 'dsdl_runtime.hpp' | wc -l
```

Expected:

- All three counts are equal.

### 11.5 Compile-check generated C++ headers as C++23, warning-clean

```bash
for profile in std pmr; do
  outdir="build/uavcan-cpp-out-strict-verify/${profile}"
  scratch="$(mktemp -d)"
  rc=0

  for h in $(find "${outdir}" -name '*.hpp' ! -name 'dsdl_runtime.hpp' | sort); do
    rel="${h#${outdir}/}"
    cat > "${scratch}/tu.cpp" <<EOF2
#include "${rel}"
int main() { return 0; }
EOF2
    c++ -std=c++23 -Wall -Wextra -Werror -I"${outdir}" "${scratch}/tu.cpp" -c -o "${scratch}/tu.o" || { echo "FAILED:${profile}:${rel}"; rc=1; break; }
  done

  if [ "${rc}" -ne 0 ]; then
    exit 1
  fi
done
```

Expected:

- Exit code `0`

### 11.6 Rust type-file parity check

```bash
find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-rust-out-strict-verify/src -name '*.rs' \
  ! -name 'lib.rs' ! -name 'mod.rs' ! -name 'dsdl_runtime.rs' | wc -l
```

Expected:

- Counts are equal.

## 12. Common Development Tasks

### 12.1 Reconfigure after dependency changes

```bash
cmake --preset dev
```

or manually:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=... \
  -DMLIR_DIR=...
```

### 12.2 Fast rebuild of one target

```bash
cmake --build build --target dsdlc -j
```

### 12.3 Inspect CLI options

```bash
./build/tools/dsdlc/dsdlc
./build/tools/dsdl-opt/dsdl-opt --help
```

### 12.4 Run complete automation in one command

```bash
cmake --workflow --preset full
```

## 13. Troubleshooting

### 13.1 `Could not find LLVMConfig.cmake` / `MLIRConfig.cmake`

- Use `dev-homebrew` or `dev-llvm-env` workflow presets.
- Or pass explicit `-DLLVM_DIR` and `-DMLIR_DIR` manually.
- Confirm paths contain the corresponding config files.

### 13.2 lit tests not running

- Install `llvm-lit` or Python `lit`.
- Re-run configure so CMake can detect lit tooling.

### 13.3 `dsdlc` strict generation fails

Checklist:

- Submodule initialized:
  - `git submodule update --init --recursive`
- Command points at the root namespace:
  - `--root-namespace-dir public_regulated_data_types/uavcan`
- Use strict explicitly while diagnosing:
  - `--strict`

### 13.4 Header compile-check failures

- Ensure include root is the generation root:
  - `-I build/uavcan-out-strict-verify`
- Confirm `dsdl_runtime.h` exists in output root.

### 13.5 Workflow preset not found

- Ensure your CMake supports workflow presets (`>= 3.25`).
- Check `cmake --list-presets=all` to confirm preset names.

### 13.6 C++ profile selection

- Use `--cpp-profile std` for `std::array` + `std::vector` generated output.
- Use `--cpp-profile pmr` for allocator-handoff surface on generated types and
  SerDes entry points (`std::pmr::vector`-based variable-length fields).
- Use `--cpp-profile both` to emit both trees in one run.

### 13.7 Rust profile selection

- Use `--rust-profile std` for current production path.
- `--rust-profile no-std-alloc` is intentionally not implemented yet.

## 14. Commit and PR Expectations

Please include in PR description:

- Exact preset/workflow or manual configure command used.
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
