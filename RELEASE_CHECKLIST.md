# Release Checklist

Date baseline: February 20, 2026.

Use this checklist before cutting a release tag or publishing demo/release
artifacts.

## 1. Toolchain Baseline

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

## 2. Configure + Build Preflight

```bash
cd /path/to/llvm-dsdl
git submodule update --init --recursive

cmake --workflow --preset dev-homebrew
```

If you are not on macOS/Homebrew LLVM, use the preset appropriate for your
environment.

## 3. Required Verification Gates

Run these before release:

```bash
cmake --workflow --preset full
cmake --workflow --preset optimized-homebrew
ctest --test-dir build/dev-homebrew -L ts --output-on-failure
ctest --test-dir build/dev-homebrew -L ts-runtime-specialization --output-on-failure
```

Run format/include checks:

```bash
cmake --build build/dev-homebrew --target check-format -j1
cmake --build build/dev-homebrew --target check-iwyu -j1
```

Optional but recommended static checks:

```bash
cmake --build build/dev-homebrew --target check-clang-tidy -j1
```

## 4. Demo Readiness Gates

Verify the canonical demo flow is runnable:

```bash
cmake --build build/dev-homebrew --target dsdlc dsdl-opt -j
bash -lc 'source /dev/null; DSDLC=build/dev-homebrew/tools/dsdlc/dsdlc DSDLOPT=build/dev-homebrew/tools/dsdl-opt/dsdl-opt ROOT_NS=test/lit/fixtures/vendor OUT=build/dev-homebrew/demo-smoke; rm -rf "$OUT"; mkdir -p "$OUT"; "$DSDLC" mlir --root-namespace-dir "$ROOT_NS" > "$OUT/module.mlir"; "$DSDLOPT" --pass-pipeline=builtin.module\\(lower-dsdl-serialization,convert-dsdl-to-emitc\\) "$OUT/module.mlir" > "$OUT/module.emitc.mlir"; "$DSDLC" c --root-namespace-dir "$ROOT_NS" --out-dir "$OUT/c"; "$DSDLC" cpp --root-namespace-dir "$ROOT_NS" --cpp-profile both --out-dir "$OUT/cpp"; "$DSDLC" rust --root-namespace-dir "$ROOT_NS" --rust-profile std --rust-crate-name demo_vendor_generated --out-dir "$OUT/rust"; "$DSDLC" go --root-namespace-dir "$ROOT_NS" --go-module demo/vendor/generated --out-dir "$OUT/go"; "$DSDLC" ts --root-namespace-dir "$ROOT_NS" --ts-module demo_vendor_generated_ts --out-dir "$OUT/ts"'
```

## 5. Release Artifact Expectations

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

## 6. Final Sign-Off

Release sign-off requires all of the following:

1. Required gates in Sections 3 and 4 pass.
2. No unresolved high-severity regressions in open TODO/fix lists.
3. Canonical docs are in sync:
   - `README.md`
   - `DEMO.md`
   - `DESIGN.md`
   - `CANONICAL_PLAN.md`
