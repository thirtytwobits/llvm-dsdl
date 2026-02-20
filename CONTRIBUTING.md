# Contributing to llvm-dsdl

This document is a reproducibility-first guide for contributors. It is designed
to let a new contributor go from a clean checkout to a verified build and
`uavcan` generation with deterministic checks.

## Table of Contents

1. Scope
2. Prerequisites
3. Clean Checkout
4. Recommended: Preset-Driven Workflows
5. Configure (Manual CMake)
6. Build
7. Test
8. Reproduce `uavcan` C Generation
9. Reproduce `uavcan` C++ Generation (`std` + `pmr`)
10. Reproduce `uavcan` Rust Generation (`std`)
11. Validate Generated Output
12. Common Development Tasks
13. Troubleshooting
14. Commit and PR Expectations
15. Release Checklist

## 1. Scope

This guide covers:

- Building `dsdlc` and `dsdl-opt`.
- Running unit tests and lit tests (when available).
- Running integration validation for `uavcan` generation.
- Generating C11 output from `submodules/public_regulated_data_types/uavcan`.
- Generating C++23 output (`std`/`pmr` profiles) from
  `submodules/public_regulated_data_types/uavcan`.
- Generating Rust crate output (currently `std` profile) from
  `submodules/public_regulated_data_types/uavcan`.
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

`submodules/public_regulated_data_types` is required for full-tree generation
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

### 4.1 Canonical matrix workflows

```bash
cmake --workflow --preset matrix-dev-llvm-env
cmake --workflow --preset matrix-dev-homebrew
cmake --workflow --preset matrix-ci
```

Each matrix workflow runs:

1. configure once for one environment preset
2. `Debug` build + smoke tests (exclude `integration`)
3. `RelWithDebInfo` build + full test set
4. `Release` build + smoke tests

Release builds also invoke `bundle-tools-self-contained`.

### 4.2 Environment-driven LLVM/MLIR usage

```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --workflow --preset matrix-dev-llvm-env
```

## 5. Configure (Manual CMake)

Use this when you want explicit control instead of presets.

### 5.1 macOS/Homebrew example

```bash
LLVM_PREFIX="$(brew --prefix llvm)"

cmake -S . -B build -G "Ninja Multi-Config" \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
```

### 5.2 Generic Linux example

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
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
cmake --build build --config RelWithDebInfo -j
```

Preset build (example):

```bash
cmake --build --preset build-dev-llvm-env-relwithdebinfo
```

Expected artifacts:

- `build/.../tools/dsdlc/dsdlc`
- `build/.../tools/dsdl-opt/dsdl-opt`
- static libs under `build/.../lib/...`

## 7. Test

### 7.1 Manual test invocation

```bash
ctest --test-dir build --build-config RelWithDebInfo --output-on-failure
```

### 7.2 Preset test invocation

Fast test set:

```bash
ctest --preset test-dev-llvm-env-smoke-relwithdebinfo
```

Full test set:

```bash
ctest --preset test-dev-llvm-env-full-relwithdebinfo
```

### 7.3 lit tests (if lit is available)

When lit is configured, `ctest` includes `llvmdsdl-lit`.

You can also run lit directly:

```bash
lit -sv build/test/lit/RelWithDebInfo
```

or with Python module:

```bash
python3 -m lit.main -sv build/test/lit/RelWithDebInfo
```

## 8. Reproduce `uavcan` C Generation

This is the primary end-to-end check for current project status.

```bash
DSDLC=./build/matrix/dev-llvm-env/tools/dsdlc/RelWithDebInfo/dsdlc
OUT="build/uavcan-out-verify"
mkdir -p "${OUT}"

"${DSDLC}" c \
  --root-namespace-dir submodules/public_regulated_data_types/uavcan \
  --out-dir "${OUT}"
```

Expected result:

- Exit code `0`
- Per-type headers generated under `${OUT}/uavcan/...`
- `${OUT}/dsdl_runtime.h` emitted
- Per-definition implementation translation units generated by default under
  `${OUT}/<namespace>/`.
  Example: `${OUT}/uavcan/node/Heartbeat_1_0.c`

## 9. Reproduce `uavcan` C++ Generation (`std` + `pmr`)

```bash
OUT_CPP="build/uavcan-cpp-out-verify"
mkdir -p "${OUT_CPP}"

"${DSDLC}" cpp \
  --root-namespace-dir submodules/public_regulated_data_types/uavcan \
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
"${DSDLC}" cpp \
  --root-namespace-dir submodules/public_regulated_data_types/uavcan \
  --cpp-profile std \
  --out-dir build/uavcan-cpp-std-out
```

```bash
"${DSDLC}" cpp \
  --root-namespace-dir submodules/public_regulated_data_types/uavcan \
  --cpp-profile pmr \
  --out-dir build/uavcan-cpp-pmr-out
```

## 10. Reproduce `uavcan` Rust Generation (`std`)

```bash
OUT_RUST="build/uavcan-rust-out-verify"
mkdir -p "${OUT_RUST}"

"${DSDLC}" rust \
  --root-namespace-dir submodules/public_regulated_data_types/uavcan \
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

Additional profile:

- `--rust-profile no-std-alloc` is supported for `no_std + alloc` targets.

## 11. Validate Generated Output

### 11.1 Header count parity

```bash
find submodules/public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-out-verify -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

Expected:

- Counts are equal.

### 11.2 Ensure stubs are gone

```bash
rg -n "dsdl_runtime_stub_" build/uavcan-out-verify
```

Expected:

- No matches.

### 11.3 Compile-check all generated headers as C11, warning-clean

```bash
outdir="build/uavcan-out-verify"
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
find submodules/public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-cpp-out-verify/std -name '*.hpp' ! -name 'dsdl_runtime.hpp' | wc -l
find build/uavcan-cpp-out-verify/pmr -name '*.hpp' ! -name 'dsdl_runtime.hpp' | wc -l
```

Expected:

- All three counts are equal.

### 11.5 Compile-check generated C++ headers as C++23, warning-clean

```bash
for profile in std pmr; do
  outdir="build/uavcan-cpp-out-verify/${profile}"
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
find submodules/public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find build/uavcan-rust-out-verify/src -name '*.rs' \
  ! -name 'lib.rs' ! -name 'mod.rs' ! -name 'dsdl_runtime.rs' | wc -l
```

Expected:

- Counts are equal.

## 12. Common Development Tasks

### 12.1 Reconfigure after dependency changes

```bash
cmake --preset dev-llvm-env
```

or manually:

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
  -DLLVM_DIR=... \
  -DMLIR_DIR=...
```

### 12.2 Fast rebuild of one target

```bash
cmake --build build --config RelWithDebInfo --target dsdlc -j
```

### 12.3 Inspect CLI options

```bash
./build/tools/dsdlc/RelWithDebInfo/dsdlc
./build/tools/dsdl-opt/RelWithDebInfo/dsdl-opt --help
```

### 12.4 Run complete automation in one command

```bash
cmake --workflow --preset matrix-dev-llvm-env
```

## 13. Troubleshooting

### 13.1 `Could not find LLVMConfig.cmake` / `MLIRConfig.cmake`

- Use `matrix-dev-homebrew` or `matrix-dev-llvm-env` workflow presets.
- Or pass explicit `-DLLVM_DIR` and `-DMLIR_DIR` manually.
- Confirm paths contain the corresponding config files.

### 13.2 lit tests not running

- Install `llvm-lit` or Python `lit`.
- Re-run configure so CMake can detect lit tooling.

### 13.3 `dsdlc` generation fails

Checklist:

- Submodule initialized:
  - `git submodule update --init --recursive`
- Command points at the root namespace:
  - `--root-namespace-dir submodules/public_regulated_data_types/uavcan`

### 13.4 Header compile-check failures

- Ensure include root is the generation root:
  - `-I build/uavcan-out-verify`
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
- Use `--rust-profile no-std-alloc` for `no_std + alloc` targets.

## 14. Commit and PR Expectations

Please include in PR description:

- Exact preset/workflow or manual configure command used.
- Exact build/test commands run.
- Generation command used.
- Validation outputs:
  - DSDL/header count parity
  - stub scan result
  - C11 compile-check result

For behavior changes affecting wire semantics, include:

- A focused unit/integration test.
- A short rationale referencing affected DSDL rules.

For substantial codegen/frontend changes, also include:

- One representative generated header diff snippet (before/after).
- Any semantic behavior notes relevant to the change.

## 15. Release Checklist

Date baseline: February 20, 2026.

Use this checklist before cutting a release tag or publishing demo/release
artifacts.

### 15.1 Toolchain baseline

Required:

1. CMake `>= 3.24` (`>= 3.25` recommended for presets).
2. Ninja.
3. C/C++ toolchain with C++20 support.
4. LLVM + MLIR with `LLVMConfig.cmake` and `MLIRConfig.cmake`.

Known-good baseline used by current team workflows:

1. LLVM/MLIR `21.1.8`.
2. macOS/Homebrew LLVM path:
   - `LLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm`
   - `MLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir`

Optional but strongly recommended:

1. `lit` for lit test execution.
2. `clang-format` for format checks.
3. `clang-tidy` for static diagnostics.
4. `include-what-you-use` for include hygiene checks.
5. `cargo`/`rustc` for Rust integration gates.
6. `go` for Go integration gates.
7. `tsc` for TypeScript typecheck/consumer gates.
8. `patchelf` for Linux Release self-contained bundle rewriting.

### 15.2 Configure + build preflight

```bash
cd /path/to/llvm-dsdl
git submodule update --init --recursive

cmake --workflow --preset matrix-dev-homebrew
```

If you are not on macOS/Homebrew LLVM, use:

```bash
cmake --workflow --preset matrix-dev-llvm-env
```

### 15.3 Required verification gates

Run these before release:

```bash
cmake --workflow --preset matrix-dev-homebrew
cmake --workflow --preset matrix-dev-llvm-env
cmake --workflow --preset matrix-ci
```

Run format/include checks:

```bash
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-format -j1
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-iwyu -j1
```

Optional but recommended static checks:

```bash
cmake --build build/matrix/dev-homebrew --config RelWithDebInfo --target check-clang-tidy -j1
```

### 15.4 Demo readiness gates

Verify the canonical demo flow is runnable:

```bash
cmake --build --preset build-dev-homebrew-relwithdebinfo --target dsdlc dsdl-opt -j
bash -lc 'source /dev/null; DSDLC=build/matrix/dev-homebrew/tools/dsdlc/RelWithDebInfo/dsdlc DSDLOPT=build/matrix/dev-homebrew/tools/dsdl-opt/RelWithDebInfo/dsdl-opt ROOT_NS=test/lit/fixtures/vendor OUT=build/matrix/dev-homebrew/demo-smoke; rm -rf "$OUT"; mkdir -p "$OUT"; "$DSDLC" mlir --root-namespace-dir "$ROOT_NS" > "$OUT/module.mlir"; "$DSDLOPT" --pass-pipeline=builtin.module\\(lower-dsdl-serialization,convert-dsdl-to-emitc\\) "$OUT/module.mlir" > "$OUT/module.emitc.mlir"; "$DSDLC" c --root-namespace-dir "$ROOT_NS" --out-dir "$OUT/c"; "$DSDLC" cpp --root-namespace-dir "$ROOT_NS" --cpp-profile both --out-dir "$OUT/cpp"; "$DSDLC" rust --root-namespace-dir "$ROOT_NS" --rust-profile std --rust-crate-name demo_vendor_generated --out-dir "$OUT/rust"; "$DSDLC" go --root-namespace-dir "$ROOT_NS" --go-module demo/vendor/generated --out-dir "$OUT/go"; "$DSDLC" ts --root-namespace-dir "$ROOT_NS" --ts-module demo_vendor_generated_ts --out-dir "$OUT/ts"'
```

### 15.5 Release artifact expectations

Expectations for a release-ready repository/build state:

1. `dsdlc --help` is detailed and up to date.
2. `dsdlc` run summary reports:
   - generated file count
   - output root
   - elapsed runtime
3. `DEMO.md` quick path and scale-up path both work.
4. `DESIGN.md` reflects current architecture and backend set.
5. Generated outputs are deterministic for integration determinism lanes.
6. Lowered contract validation remains hard-fail on malformed/missing metadata.

### 15.6 Final sign-off

Release sign-off requires all of the following:

1. Required gates in Sections 15.3 and 15.4 pass.
2. No unresolved high-severity regressions in open TODO/fix lists.
3. Canonical docs are in sync:
   - `README.md`
   - `DEMO.md`
   - `DESIGN.md`
