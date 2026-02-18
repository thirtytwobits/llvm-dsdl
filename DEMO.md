# LLVM/MLIR DSDL Codegen Demo (5 Minutes, CLI-Only)

This is a fast live demo for showing the technique:

- start from platform-agnostic IDL (`.dsdl`)
- compile into MLIR
- lower with MLIR passes
- emit production-style code for multiple languages from one pipeline

The live demo itself is CLI-first. The section below is a one-time CMake prep so the demo binaries exist.

## 0) One-Time Binary Prep (CMake)

If `dsdlc` and `dsdl-opt` are not built yet, run:

```bash
cd /Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl

git submodule update --init --recursive

LLVM_PREFIX="$(brew --prefix llvm)"

cmake -S . -B build/dev-homebrew -G Ninja \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"

cmake --build build/dev-homebrew --target dsdlc dsdl-opt -j

build/dev-homebrew/tools/dsdlc/dsdlc --help >/dev/null
build/dev-homebrew/tools/dsdl-opt/dsdl-opt --help >/dev/null
```

## 1) Demo Setup (30 seconds)

Run from repo root:

```bash
cd /Users/thirtytwobits/workspace/github/thirtytwobits/llvm-dsdl

DSDLC="${DSDLC:-build/dev-homebrew/tools/dsdlc/dsdlc}"
DSDLOPT="${DSDLOPT:-build/dev-homebrew/tools/dsdl-opt/dsdl-opt}"
ROOT_NS="test/lit/fixtures/vendor"   # fast 6-type fixture for live demo
OUT="build/dev-homebrew/demo-live"

test -x "$DSDLC" && test -x "$DSDLOPT"
find "$ROOT_NS" -name '*.dsdl' | wc -l
```

Expected: `6`.

## 2) One-Pass Live Generation (90 seconds)

```bash
set -euo pipefail
rm -rf "$OUT"
mkdir -p "$OUT"

"$DSDLC" ast  --root-namespace-dir "$ROOT_NS" > "$OUT/ast.txt"
"$DSDLC" mlir --root-namespace-dir "$ROOT_NS" > "$OUT/module.mlir"

"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization\) \
  "$OUT/module.mlir" > "$OUT/module.lowered.mlir"

"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization,convert-dsdl-to-emitc\) \
  "$OUT/module.mlir" > "$OUT/module.emitc.mlir"

"$DSDLC" c \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --out-dir "$OUT/c"

"$DSDLC" cpp \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --cpp-profile both \
  --out-dir "$OUT/cpp"

"$DSDLC" rust \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --rust-profile std \
  --rust-crate-name demo_vendor_generated \
  --out-dir "$OUT/rust"

"$DSDLC" go \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --go-module demo/vendor/generated \
  --out-dir "$OUT/go"
```

## 3) Show “Proof” in 60 Seconds

### 2.1 IR stages

```bash
wc -l "$OUT/module.mlir" "$OUT/module.lowered.mlir" "$OUT/module.emitc.mlir"
rg -n "dsdl\\.serialization_plan" "$OUT/module.mlir" | head
rg -n "llvmdsdl\\.lowered_contract_producer|lowered_capacity_check_helper" "$OUT/module.lowered.mlir" | head
rg -n "emitc\\.verbatim|vendor_Type_1_0__serialize_ir_" "$OUT/module.emitc.mlir" | head
```

### 2.2 Multi-language output from same IDL

```bash
find "$OUT/c"   -maxdepth 3 -type f | sort
find "$OUT/cpp" -maxdepth 4 -type f | sort
find "$OUT/rust/src/vendor" -maxdepth 1 -type f | sort
find "$OUT/go/vendor" -maxdepth 1 -type f | sort
```

### 2.3 One type, four languages

```bash
sed -n '1,60p'  "$OUT/c/vendor/Type_1_0.h"
sed -n '1,80p'  "$OUT/cpp/std/vendor/Type_1_0.hpp"
sed -n '1,80p'  "$OUT/rust/src/vendor/type__1_0.rs"
sed -n '1,80p'  "$OUT/go/vendor/type__1_0.go"
```

## 4) Five-Minute Talk Track

Use this pacing:

1. **0:00-0:30**: "We start with a platform-agnostic IDL corpus (`.dsdl`)."
2. **0:30-1:30**: "Frontend + semantics produce structured IR (`ast`, `module.mlir`)."
3. **1:30-2:15**: "MLIR passes lower serialization plans and can convert toward EmitC-compatible representation."
4. **2:15-3:45**: "From the same semantic source, we emit C, C++ (`std`/`pmr`), Rust, and Go."
5. **3:45-5:00**: "The value is compiler architecture: shared analysis, explicit contracts, target-specific emitters, and repeatable transformations."

## 5) Kitchen Tour (90 seconds)

The quick source walk:

1. Entry points
```bash
ls tools/dsdlc tools/dsdl-opt
```

2. Frontend + semantics (parse, resolve, validate)
```bash
ls include/llvmdsdl/Frontend include/llvmdsdl/Semantics
ls lib/Frontend lib/Semantics
```

3. DSDL dialect + lowering + passes
```bash
ls include/llvmdsdl/IR include/llvmdsdl/Lowering include/llvmdsdl/Transforms
ls lib/IR lib/Lowering lib/Transforms
```

4. Language backends
```bash
ls include/llvmdsdl/CodeGen
ls lib/CodeGen
```

5. Runtime layers
```bash
ls runtime runtime/cpp runtime/rust runtime/go
```

6. Validation corpus and tests
```bash
find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
ls test/lit test/integration test/unit
```

## 6) Optional “Scale-Up” (Full `uavcan`)

If you want to show real corpus scale (not just the 6-type fixture):

```bash
ROOT_NS_FULL="public_regulated_data_types/uavcan"
OUT_FULL="build/dev-homebrew/demo-uavcan"
rm -rf "$OUT_FULL"
mkdir -p "$OUT_FULL"

"$DSDLC" mlir --root-namespace-dir "$ROOT_NS_FULL" > "$OUT_FULL/module.mlir"
"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization,convert-dsdl-to-emitc\) \
  "$OUT_FULL/module.mlir" > "$OUT_FULL/module.emitc.mlir"

"$DSDLC" c    --root-namespace-dir "$ROOT_NS_FULL" --strict --out-dir "$OUT_FULL/c"
"$DSDLC" cpp  --root-namespace-dir "$ROOT_NS_FULL" --strict --cpp-profile both --out-dir "$OUT_FULL/cpp"
"$DSDLC" rust --root-namespace-dir "$ROOT_NS_FULL" --strict --rust-profile std --rust-crate-name uavcan_dsdl_generated --out-dir "$OUT_FULL/rust"
"$DSDLC" go   --root-namespace-dir "$ROOT_NS_FULL" --strict --go-module demo/uavcan/generated --out-dir "$OUT_FULL/go"
```

Quick count checks:

```bash
find "$ROOT_NS_FULL" -name '*.dsdl' | wc -l
find "$OUT_FULL/c" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

## 7) Message to Land with the Room

Use this close:

"This is not just a code generator script. It is a compiler pipeline: typed frontend, analyzable MLIR dialect, explicit lowering contracts, and multi-target emission. That architecture is what makes new backends, static analysis, and optimization realistic instead of ad hoc."

## 8) Caveat: Remaining Project Work

As of **February 18, 2026**, this demo shows real end-to-end value, but the plan is not yet fully finished.

Remaining work (from `MLIR_MAX_LOWERING_PLAN.md`) is primarily:

1. Finish tightening the lowered contract so every SerDes-relevant decision is unambiguous and fully validated (Phase 0/1 closeout).
2. Complete full convergence so C++/Rust/Go body shaping is entirely driven by the shared lowered body plan, with no backend-specific semantic branching (Phase 2 finish).
3. Eliminate any remaining semantic fallback patterns and keep hard-fail behavior when required lowered facts are missing (Phase 3 finish).
4. Keep parity gates fully normalized across all language pairs and fixtures (including directed vectors and inventory/pass-line invariants) as coverage expands (Phase 4 finish).
5. Add optimization-pass readiness and prove parity is unchanged with optimization-enabled lowering pipelines (Phase 5).
6. Update architecture documentation (`DESIGN.md`) to reflect final intentional backend-specific behavior only.

When all work in `MLIR_MAX_LOWERING_PLAN.md` is complete, we should have:

1. A versioned, backend-agnostic lowered SerDes contract as the single source of truth.
2. C, C++ (`std`/`pmr`), Rust, and Go emitters acting as thin rendering/adaptation layers over one shared lowered program.
3. No backend semantic fallback drift for core SerDes families (scalar, alignment, arrays, union tags, delimiters, service sections).
4. Strong cross-language equivalence guarantees enforced by parity suites with consistent summary/inventory/count invariants.
5. Optional optimization passes that improve lowered code shape without changing wire semantics.
6. A stable foundation to add new runtime profiles and future language targets without re-implementing semantics per backend.
