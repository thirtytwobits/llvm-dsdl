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

"$DSDLC" --target-language ast  "$ROOT_NS" > "$OUT/ast.txt"
"$DSDLC" --target-language mlir "$ROOT_NS" > "$OUT/module.mlir"

"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization\) \
  "$OUT/module.mlir" > "$OUT/module.lowered.mlir"

"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization,convert-dsdl-to-emitc\) \
  "$OUT/module.mlir" > "$OUT/module.emitc.mlir"

"$DSDLC" --target-language c \
  "$ROOT_NS" \
  --outdir "$OUT/c"

"$DSDLC" --target-language cpp \
  "$ROOT_NS" \
  --cpp-profile both \
  --outdir "$OUT/cpp"

"$DSDLC" --target-language rust \
  "$ROOT_NS" \
  --rust-profile std \
  --rust-crate-name demo_vendor_generated \
  --outdir "$OUT/rust"

"$DSDLC" --target-language rust \
  "$ROOT_NS" \
  --rust-profile std \
  --rust-runtime-specialization fast \
  --rust-crate-name demo_vendor_generated_fast \
  --outdir "$OUT/rust-fast"

"$DSDLC" --target-language go \
  "$ROOT_NS" \
  --go-module demo/vendor/generated \
  --outdir "$OUT/go"

"$DSDLC" --target-language ts \
  "$ROOT_NS" \
  --ts-module demo_vendor_generated_ts \
  --outdir "$OUT/ts"

"$DSDLC" --target-language python \
  "$ROOT_NS" \
  --py-package demo_vendor_generated_py \
  --outdir "$OUT/python"
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
find "$OUT/python/demo_vendor_generated_py/vendor" -maxdepth 1 -type f | sort
```

One-type, many-targets proof:

```bash
sed -n '1,60p' "$OUT/c/vendor/Type_1_0.h"
sed -n '1,80p' "$OUT/cpp/std/vendor/Type_1_0.hpp"
sed -n '1,80p' "$OUT/rust/src/vendor/type__1_0.rs"
sed -n '1,80p' "$OUT/go/vendor/type__1_0.go"
sed -n '1,80p' "$OUT/ts/vendor/type__1_0.ts"
sed -n '1,80p' "$OUT/python/demo_vendor_generated_py/vendor/type_1_0.py"
```

### 1.4 Python runtime/backend mini-demo (60 seconds)

Show runtime specialization and backend-selection model live:

```bash
"$DSDLC" --target-language python \
  "$ROOT_NS" \
  --py-package demo_vendor_generated_py_fast \
  --py-runtime-specialization fast \
  --outdir "$OUT/python-fast"

python3 -m venv "$OUT/python/.venv"
"$OUT/python/.venv/bin/pip" install -e "$OUT/python"

LLVMDSDL_PY_RUNTIME_MODE=pure \
  "$OUT/python/.venv/bin/python" - <<'PY'
from demo_vendor_generated_py import _runtime_loader as rl
from demo_vendor_generated_py.vendor.type_1_0 import Type_1_0
payload = Type_1_0(foo=1, bar=2).serialize()
roundtrip = Type_1_0.deserialize(payload)
print("backend=", rl.BACKEND, "payload_len=", len(payload), "bar=", roundtrip.bar)
PY

LLVMDSDL_PY_RUNTIME_MODE=auto \
  "$OUT/python/.venv/bin/python" - <<'PY'
from demo_vendor_generated_py import _runtime_loader as rl
print("auto backend=", rl.BACKEND)
PY

test -f "$OUT/python/pyproject.toml"
test -f "$OUT/python/demo_vendor_generated_py/py.typed"
```

## 2) Five-Minute Talk Track

1. `0:00-0:30`: "IDL in, single frontend + semantic pass."
2. `0:30-1:30`: "MLIR module captures canonical representation."
3. `1:30-2:15`: "Lowering + conversion passes create explicit, inspectable transformation steps."
4. `2:15-3:45`: "Same semantic source emits C/C++/Rust/Go/TypeScript/Python."
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
ls "$OUT/python/demo_vendor_generated_py/_dsdl_runtime.py"
ls "$OUT/python/demo_vendor_generated_py/_runtime_loader.py"
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

"$DSDLC" --target-language mlir "$ROOT_NS_FULL" > "$OUT_FULL/module.mlir"
"$DSDLOPT" --pass-pipeline=builtin.module\(lower-dsdl-serialization,convert-dsdl-to-emitc\) \
  "$OUT_FULL/module.mlir" > "$OUT_FULL/module.emitc.mlir"

"$DSDLC" --target-language c \
  "$ROOT_NS_FULL" \
  --outdir "$OUT_FULL/c"
"$DSDLC" --target-language cpp \
  "$ROOT_NS_FULL" \
  --cpp-profile both \
  --outdir "$OUT_FULL/cpp"
"$DSDLC" --target-language rust \
  "$ROOT_NS_FULL" \
  --rust-profile std \
  --rust-crate-name uavcan_dsdl_generated \
  --outdir "$OUT_FULL/rust"
"$DSDLC" --target-language go \
  "$ROOT_NS_FULL" \
  --go-module demo/uavcan/generated \
  --outdir "$OUT_FULL/go"
"$DSDLC" --target-language ts \
  "$ROOT_NS_FULL" \
  --ts-module demo_uavcan_generated_ts \
  --outdir "$OUT_FULL/ts"
"$DSDLC" --target-language python \
  "$ROOT_NS_FULL" \
  --py-package demo_uavcan_generated_py \
  --outdir "$OUT_FULL/python"
```

Scale counts:

```bash
find "$ROOT_NS_FULL" -name '*.dsdl' | wc -l
find "$OUT_FULL/c" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

## 5) Caveat: Remaining Work

As of **February 20, 2026**, the MLIR-max lowering program for the in-scope
backend set is complete (C/C++/Rust/Go/TypeScript/Python), including lowered-contract
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

## 6) Python Troubleshooting Matrix (Demo-Day Quick Reference)

| Symptom | Quick diagnosis | Fix |
| --- | --- | --- |
| `LLVMDSDL_PY_RUNTIME_MODE=accel` import error | accelerator module is missing beside generated package | build with `-DLLVMDSDL_ENABLE_PYTHON_ACCELERATOR=ON` and stage with `stage-uavcan-python-runtime-accelerator-required` |
| `auto backend=pure` when accel expected | fallback is active because accel was unavailable | verify staged `_dsdl_runtime_accel.*` and rerun |
| `pip install -e` fails in generated output | missing or stale `pyproject.toml` | regenerate with `dsdlc --target-language python` and reinstall from fresh output dir |
| specialization parity concern (`portable` vs `fast`) | helper implementation changed unexpectedly | run `llvmdsdl-uavcan-python-runtime-specialization-diff` and parity lanes before demo |
| benchmark gate failures | threshold config not calibrated for this host | run artifact-only benchmark first, calibrate thresholds, then enable gating |
