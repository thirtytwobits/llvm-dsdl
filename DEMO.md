# LLVM/MLIR Multi-Language Codegen Demo (5 Minutes, CLI-First)

This demo is for the technique:

1. Start with a platform-agnostic IDL corpus.
2. Parse/analyze once.
3. Lower through MLIR once.
4. Emit multiple target languages from one compiler pipeline.

The quick path below is designed for a live room demo in under five minutes.

## 0) Demo Setup (One-Time Build Prep)

Run once to prepare binaries used by the live demo:

```bash
cd /path/to/llvm-dsdl

git submodule update --init --recursive

LLVM_PREFIX="$(brew --prefix llvm)"   # macOS/Homebrew example

cmake -S . -B build/dev-homebrew -G Ninja \
  -DLLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"

cmake --build build/dev-homebrew --target dsdlc dsdl-opt -j

build/dev-homebrew/tools/dsdlc/dsdlc --help >/dev/null
build/dev-homebrew/tools/dsdl-opt/dsdl-opt --help >/dev/null
```

## 1) Quick Demo Path (Under 5 Minutes)

### 1.1 Session setup (30 seconds)

```bash
cd /path/to/llvm-dsdl

DSDLC="${DSDLC:-build/dev-homebrew/tools/dsdlc/dsdlc}"
DSDLOPT="${DSDLOPT:-build/dev-homebrew/tools/dsdl-opt/dsdl-opt}"
ROOT_NS="test/lit/fixtures/vendor"
OUT="build/dev-homebrew/demo-live"

test -x "$DSDLC"
test -x "$DSDLOPT"
find "$ROOT_NS" -name '*.dsdl' | wc -l
```

Expected fixture count: `6`.

### 1.2 Single-pass compile/lower/emit (90 seconds)

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
  --out-dir "$OUT/c"

"$DSDLC" cpp \
  --root-namespace-dir "$ROOT_NS" \
  --cpp-profile both \
  --out-dir "$OUT/cpp"

"$DSDLC" rust \
  --root-namespace-dir "$ROOT_NS" \
  --rust-profile std \
  --rust-crate-name demo_vendor_generated \
  --out-dir "$OUT/rust"

"$DSDLC" rust \
  --root-namespace-dir "$ROOT_NS" \
  --rust-profile std \
  --rust-runtime-specialization fast \
  --rust-crate-name demo_vendor_generated_fast \
  --out-dir "$OUT/rust-fast"

"$DSDLC" go \
  --root-namespace-dir "$ROOT_NS" \
  --go-module demo/vendor/generated \
  --out-dir "$OUT/go"

"$DSDLC" ts \
  --root-namespace-dir "$ROOT_NS" \
  --ts-module demo_vendor_generated_ts \
  --out-dir "$OUT/ts"
```

### 1.3 Show proof on screen (60 seconds)

IR stage proof:

```bash
wc -l "$OUT/module.mlir" "$OUT/module.lowered.mlir" "$OUT/module.emitc.mlir"
rg -n "dsdl\\.serialization_plan" "$OUT/module.mlir" | head
rg -n "llvmdsdl\\.lowered_contract_producer|lowered_capacity_check_helper" "$OUT/module.lowered.mlir" | head
rg -n "emitc\\.verbatim|_serialize_ir_" "$OUT/module.emitc.mlir" | head
```

Multi-language output proof:

```bash
find "$OUT/c" -maxdepth 3 -type f | sort
find "$OUT/cpp" -maxdepth 4 -type f | sort
find "$OUT/rust/src/vendor" -maxdepth 1 -type f | sort
find "$OUT/go/vendor" -maxdepth 1 -type f | sort
find "$OUT/ts/vendor" -maxdepth 1 -type f | sort
```

One-type, many-targets proof:

```bash
sed -n '1,60p' "$OUT/c/vendor/Type_1_0.h"
sed -n '1,80p' "$OUT/cpp/std/vendor/Type_1_0.hpp"
sed -n '1,80p' "$OUT/rust/src/vendor/type__1_0.rs"
sed -n '1,80p' "$OUT/go/vendor/type__1_0.go"
sed -n '1,80p' "$OUT/ts/vendor/type__1_0.ts"
```

## 2) Five-Minute Talk Track

1. `0:00-0:30`: "IDL in, single frontend + semantic pass."
2. `0:30-1:30`: "MLIR module captures canonical representation."
3. `1:30-2:15`: "Lowering + conversion passes create explicit, inspectable transformation steps."
4. `2:15-3:45`: "Same semantic source emits C/C++/Rust/Go/TypeScript."
5. `3:45-5:00`: "Value is architecture: shared analysis and lowering, target-specific rendering."

## 3) Kitchen Tour (90 Seconds)

Entry points:

```bash
ls tools/dsdlc tools/dsdl-opt
```

Frontend and semantics:

```bash
ls include/llvmdsdl/Frontend include/llvmdsdl/Semantics
ls lib/Frontend lib/Semantics
```

Dialect, lowering, transforms:

```bash
ls include/llvmdsdl/IR include/llvmdsdl/Lowering include/llvmdsdl/Transforms
ls lib/IR lib/Lowering lib/Transforms
```

Backends and shared codegen planning:

```bash
ls include/llvmdsdl/CodeGen
ls lib/CodeGen
```

Runtime layers:

```bash
ls runtime runtime/cpp runtime/rust runtime/go
ls "$OUT/ts/dsdl_runtime.ts"
```

Tests and corpus:

```bash
find submodules/public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
ls test/lit test/integration test/unit
```

## 4) Optional Scale-Up Path (Full `uavcan`)

Use this when you want to show "same technique at corpus scale":

```bash
ROOT_NS_FULL="submodules/public_regulated_data_types/uavcan"
OUT_FULL="build/dev-homebrew/demo-uavcan"

rm -rf "$OUT_FULL"
mkdir -p "$OUT_FULL"

"$DSDLC" mlir --root-namespace-dir "$ROOT_NS_FULL" > "$OUT_FULL/module.mlir"
"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization,convert-dsdl-to-emitc\) \
  "$OUT_FULL/module.mlir" > "$OUT_FULL/module.emitc.mlir"

"$DSDLC" c \
  --root-namespace-dir "$ROOT_NS_FULL" \
  --out-dir "$OUT_FULL/c"
"$DSDLC" cpp \
  --root-namespace-dir "$ROOT_NS_FULL" \
  --cpp-profile both \
  --out-dir "$OUT_FULL/cpp"
"$DSDLC" rust \
  --root-namespace-dir "$ROOT_NS_FULL" \
  --rust-profile std \
  --rust-crate-name uavcan_dsdl_generated \
  --out-dir "$OUT_FULL/rust"
"$DSDLC" go \
  --root-namespace-dir "$ROOT_NS_FULL" \
  --go-module demo/uavcan/generated \
  --out-dir "$OUT_FULL/go"
"$DSDLC" ts \
  --root-namespace-dir "$ROOT_NS_FULL" \
  --ts-module demo_uavcan_generated_ts \
  --out-dir "$OUT_FULL/ts"
```

Scale counts:

```bash
find "$ROOT_NS_FULL" -name '*.dsdl' | wc -l
find "$OUT_FULL/c" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

## 5) Caveat: Remaining Work

As of **February 20, 2026**, the MLIR-max lowering program for the in-scope
backend set is complete (C/C++/Rust/Go/TypeScript), including lowered-contract
validation, runtime specialization lanes, and cross-language parity gates.

Work that remains is mostly product polish and expansion, not core viability:

1. Continue reducing backend-local rendering code so new targets can be added
   with lower risk.
2. Keep the release checklist and integration gates current as toolchains and
   dependencies evolve.
3. Add next-backend onboarding guidance and apply the same lowered-contract +
   parity model to future targets.

What this project delivers now:

1. A compiler-shaped pipeline (frontend -> semantics -> MLIR -> lowering ->
   emitters), not isolated per-language generators.
2. One platform-agnostic IDL source producing multiple target language outputs
   through shared lowering contracts.
3. A repeatable validation model (unit, integration, parity, determinism,
   and optimization-enabled checks) that supports confidence at scale.
