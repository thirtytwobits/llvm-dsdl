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

"$DSDLC" rust \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --rust-profile std \
  --rust-runtime-specialization fast \
  --rust-crate-name demo_vendor_generated_fast \
  --out-dir "$OUT/rust-fast"

"$DSDLC" go \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --go-module demo/vendor/generated \
  --out-dir "$OUT/go"

"$DSDLC" ts \
  --root-namespace-dir "$ROOT_NS" \
  --strict \
  --ts-module demo_vendor_generated_ts \
  --out-dir "$OUT/ts"
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
find "$OUT/ts/vendor" -maxdepth 1 -type f | sort
```

Optional quick TypeScript compile proof (if `tsc` is installed):

```bash
cat > "$OUT/ts/tsconfig.json" <<'JSON'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "Node",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true
  },
  "include": ["./**/*.ts"]
}
JSON

tsc -p "$OUT/ts/tsconfig.json" --pretty false
```

Optional quick TypeScript determinism proof:

```bash
OUT_TS_DET="$OUT/ts-determinism"
rm -rf "$OUT_TS_DET"
mkdir -p "$OUT_TS_DET"

"$DSDLC" ts --root-namespace-dir "$ROOT_NS" --strict --ts-module demo_vendor_generated_ts --out-dir "$OUT_TS_DET/run-a"
"$DSDLC" ts --root-namespace-dir "$ROOT_NS" --strict --ts-module demo_vendor_generated_ts --out-dir "$OUT_TS_DET/run-b"

diff -ru "$OUT_TS_DET/run-a" "$OUT_TS_DET/run-b"
```

Optional quick TypeScript consumer-smoke proof (imports from generated `index.ts`):

```bash
cat > "$OUT/ts/consumer_smoke.ts" <<'TS'
import { uavcan_node_heartbeat_1_0, uavcan_primitive_empty_1_0 } from "./index";
const heartbeatName: string = uavcan_node_heartbeat_1_0.DSDL_FULL_NAME;
const emptyName: string = uavcan_primitive_empty_1_0.DSDL_FULL_NAME;
export const smokeSummary = `${heartbeatName}:${emptyName}`;
TS

tsc -p "$OUT/ts/tsconfig.json" --pretty false
```

Optional quick TypeScript `index.ts` alias-contract proof:

```bash
INDEX_ALIAS_COUNT="$(rg -n '^export \* as [A-Za-z_][A-Za-z0-9_]* from "\\./.+";$' "$OUT/ts/index.ts" | wc -l | tr -d ' ')"
TYPE_FILE_COUNT="$(find "$OUT/ts" -name '*.ts' ! -name 'index.ts' | wc -l | tr -d ' ')"
echo "index aliases=$INDEX_ALIAS_COUNT type files=$TYPE_FILE_COUNT"
test "$INDEX_ALIAS_COUNT" = "$TYPE_FILE_COUNT"
```

Optional quick TypeScript runtime parity smoke proof (fixture-backed):

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-fixtures-c-ts-runtime-parity \
  --output-on-failure
```

Optional quick TypeScript variable-array C<->TS parity smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-fixtures-c-ts-variable-array-parity \
  --output-on-failure
```

Optional quick TypeScript fixed-array runtime smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-ts-runtime-fixed-array-smoke \
  --output-on-failure
```

Optional quick TypeScript variable-array runtime smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-ts-runtime-variable-array-smoke \
  --output-on-failure
```

Optional quick TypeScript bigint runtime smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-ts-runtime-bigint-smoke \
  --output-on-failure
```

Optional quick TypeScript bigint C<->TS parity smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-fixtures-c-ts-bigint-parity \
  --output-on-failure
```

Optional quick TypeScript union runtime smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-ts-runtime-union-smoke \
  --output-on-failure
```

Optional quick TypeScript union C<->TS parity smoke proof:

```bash
ctest --test-dir build/dev-homebrew \
  -R llvmdsdl-fixtures-c-ts-union-parity \
  --output-on-failure
```

Optional quick strict-vs-compat TypeScript proof (migration semantics):

```bash
COMPAT_ROOT="$OUT/compatdemo"
rm -rf "$COMPAT_ROOT"
mkdir -p "$COMPAT_ROOT"
cat > "$COMPAT_ROOT/CompatArray.1.0.dsdl" <<'DSDL'
uint8[0] fixed_bad
uint8[<=0] var_inc_bad
uint8[<1] var_exc_bad
@sealed
DSDL

# strict (spec-first) should fail
"$DSDLC" ts --root-namespace-dir "$COMPAT_ROOT" --strict --out-dir "$OUT/ts-compat-strict" || true

# compat should pass with deterministic compat warnings and clamped semantics
"$DSDLC" ts --root-namespace-dir "$COMPAT_ROOT" --compat-mode --out-dir "$OUT/ts-compat"
```

Optional one-command TypeScript completion ring (strict + compat + parity):

```bash
ctest --test-dir build/dev-homebrew -L ts --output-on-failure
```

### 2.3 One type, five languages

```bash
sed -n '1,60p'  "$OUT/c/vendor/Type_1_0.h"
sed -n '1,80p'  "$OUT/cpp/std/vendor/Type_1_0.hpp"
sed -n '1,80p'  "$OUT/rust/src/vendor/type__1_0.rs"
sed -n '1,80p'  "$OUT/go/vendor/type__1_0.go"
sed -n '1,80p'  "$OUT/ts/vendor/type__1_0.ts"
```

## 4) Five-Minute Talk Track

Use this pacing:

1. **0:00-0:30**: "We start with a platform-agnostic IDL corpus (`.dsdl`)."
2. **0:30-1:30**: "Frontend + semantics produce structured IR (`ast`, `module.mlir`)."
3. **1:30-2:15**: "MLIR passes lower serialization plans and can convert toward EmitC-compatible representation."
4. **2:15-3:45**: "From the same semantic source, we emit C, C++ (`std`/`pmr`), Rust, Go, and TypeScript."
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
ls "$OUT/ts/dsdl_runtime.ts"
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
"$DSDLC" ts   --root-namespace-dir "$ROOT_NS_FULL" --strict --ts-module demo_uavcan_generated_ts --out-dir "$OUT_FULL/ts"
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

As of **February 19, 2026**, Phases 0-5 in `MLIR_MAX_LOWERING_PLAN.md` are complete for the current backend scope (C/C++/Rust/Go), and TypeScript now includes strict+compat integration/parity coverage, including optimization-enabled parity gates.

Remaining work is now the **optional stretch**:

1. Continue maturing the new language-agnostic render-IR layer (`LoweredRenderIR`) so more backend code paths become pure rendering adapters.
2. Continue profile-knob expansion (embedded allocator/runtime specialization) without changing lowered wire semantics. (`--rust-profile no-std-alloc` and `--rust-runtime-specialization fast` are implemented with generation/cargo-check/parity and semantic-diff validation lanes.)
3. Continue converging all backends (including TypeScript) on thinner shared render-IR consumption so backend code remains syntax/runtime binding.

What we have now (already true):

1. A versioned, backend-agnostic lowered SerDes contract as the semantic source of truth.
2. C, C++ (`std`/`pmr`), Rust, Go, and TypeScript emitters converged on shared lowering products with strong parity/differential validation; TypeScript includes declaration generation, generated runtime helpers (`dsdl_runtime.ts`), compile/determinism/consumer/index-contract/runtime-execution gates, strict+compat generation/runtime gates, and C<->TS parity lanes (including signed-narrow and optimized variants).
3. Optional optimization passes with parity-preserving validation under optimization-enabled workflows.

What we should have when optional stretch work is done:

1. Even thinner backends built on shared render IR with backend code focused on syntax/runtime binding.
2. Profile flexibility (`std`/`pmr`/`no_std`/runtime-specialized embedded variants) without semantic drift.
3. A non-C-like target path where TypeScript remains parity-validated in strict and compat modes with no fallback runtime stubs, lowering risk for additional language targets.
