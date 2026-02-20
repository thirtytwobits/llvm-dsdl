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
  - `std` and `no-std-alloc` profiles plus runtime specialization
    (`portable|fast`).
  - MLIR schema/plan metadata validation before emission (matching C++ structural checks).
- Go code generation (`dsdlc go`) with:
  - A generated module layout (`go.mod`, `uavcan/**`, `dsdlruntime/**`).
  - Per-type Go data types and inline SerDes methods.
  - A local Go runtime module (`dsdlruntime/dsdl_runtime.go`) with bit-level primitives.
  - Deterministic output and full-`uavcan` generation/build gates.
- TypeScript code generation (`dsdlc ts`) with:
  - A generated package/module layout (`package.json`, `index.ts`, namespace `*.ts` files).
  - Per-type TypeScript interface/type declarations, DSDL metadata constants, and generated runtime SerDes entrypoints.
  - A generated TypeScript runtime helper module (`dsdl_runtime.ts`) for bit-level read/write primitives.
  - Runtime specialization (`portable|fast`) for generated runtime helper implementation strategy.
  - MLIR schema/plan metadata validation before emission.
- Spec-conformant frontend semantics (single mode; no compatibility fallback mode).

## Ambitions

Support code generation from a common MLIR for
- C17
- C++20 std
- C++20 pmr
- Rust std
- Rust no-std
- typescript
- wasm

Tools
- dsdlc -> code generator
- dsdl-opt -> out-of tree LLVM plugin
- libdsdlc -> dynamic DSDL serdes for each language supported.
- dsdld -> DSDL language server

## Repository Layout

- `include/llvmdsdl`: public C++ headers.
- `lib`: frontend, semantics, IR, lowering, transforms, codegen.
- `tools/dsdlc`: CLI driver (`ast`, `mlir`, `c`, `cpp`, `rust`, `go`, `ts`).
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

If `llvm-lit`, `lit` (from PATH), or Python `lit` is not installed, CMake will
skip lit tests and print a warning. Unit tests still run via `ctest`.

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

Full verification workflow (includes `uavcan` C, C++, and Rust integration tests):

```bash
cmake --workflow --preset full
```

The full test set now also includes `llvmdsdl-uavcan-mlir-lowering`, which
validates full-`uavcan` MLIR lowering and `convert-dsdl-to-emitc` pass
execution under `dsdl-opt`.

Optimization-enabled verification workflow (runs tests labeled `optimized`,
including signed-narrow parity, full `uavcan` parity, and differential parity
with `--optimize-lowered-serdes`):

```bash
cmake --workflow --preset optimized
```

On macOS with Homebrew LLVM:

```bash
cmake --workflow --preset optimized-homebrew
```

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

Rust no-std profile workflow (runs tests labeled `rust-no-std`):

```bash
cmake --workflow --preset rust-no-std
```

Rust runtime-specialization workflow (runs tests labeled
`rust-runtime-specialization`):

```bash
cmake --workflow --preset rust-runtime-specialization
```

This lane now includes runtime-fast generation/cargo checks, runtime
specialization semantic-diff checks, and C/Rust parity checks (including
`no-std-alloc + runtime-fast`).

TypeScript runtime-specialization workflow (runs tests labeled
`ts-runtime-specialization`):

```bash
cmake --workflow --preset ts-runtime-specialization
```

This lane includes runtime-fast generation, typecheck, C<->TS parity, and
portable-vs-fast semantic-diff checks.

TypeScript non-C-like target workflow (runs tests labeled `ts`):

```bash
cmake --workflow --preset ts
```

This lane runs the full TypeScript gate set, including generation/runtime
checks, full-`uavcan` generation/determinism/typecheck/
consumer/index-contract/runtime-execution gates, fixture/runtime semantic-family
smoke/parity lanes, and invariant-based C<->TS parity lanes (signed-narrow and
optimized variants included).

TypeScript completion workflow preset (same `ts` label, named for CI/demo use):

```bash
cmake --workflow --preset ts-complete
```

Homebrew LLVM variant:

```bash
cmake --workflow --preset ts-complete-homebrew
```

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

Run only optimization-enabled verification gates:

```bash
ctest --test-dir build -L optimized --output-on-failure
```

Run only the C/Rust parity test (when `cargo` and `rustc` are available):

```bash
ctest --test-dir build -R llvmdsdl-uavcan-c-rust-parity --output-on-failure
```

Run only Rust no-std profile gates:

```bash
ctest --test-dir build -L rust-no-std --output-on-failure
```

Run only Rust runtime-specialization gates:

```bash
ctest --test-dir build -L rust-runtime-specialization --output-on-failure
```

Run only TypeScript runtime-specialization gates:

```bash
ctest --test-dir build -L ts-runtime-specialization --output-on-failure
```

Run only the C/C++ PMR parity test:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-cpp-pmr-c-parity --output-on-failure
```

Run only the signed-narrow C/Go fixture parity test:

```bash
ctest --test-dir build -R llvmdsdl-signed-narrow-c-go-parity --output-on-failure
```

Run only Go runtime unit tests:

```bash
ctest --test-dir build -R llvmdsdl-go-runtime-unit-tests --output-on-failure
```

Run only TypeScript generation integration gate:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-ts-generation --output-on-failure
```

Run only TypeScript determinism integration gate:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-ts-determinism --output-on-failure
```

Run only TypeScript type-check integration gate (requires `tsc`):

```bash
ctest --test-dir build -R llvmdsdl-uavcan-ts-typecheck --output-on-failure
```

Run only TypeScript consumer-smoke integration gate (requires `tsc`):

```bash
ctest --test-dir build -R llvmdsdl-uavcan-ts-consumer-smoke --output-on-failure
```

Run only TypeScript index-contract integration gate:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-ts-index-contract --output-on-failure
```

Run only fixture C<->TypeScript runtime parity smoke:

```bash
ctest --test-dir build -R llvmdsdl-fixtures-c-ts-runtime-parity --output-on-failure
```

Run only TypeScript fixed-array runtime smoke:

```bash
ctest --test-dir build -R llvmdsdl-ts-runtime-fixed-array-smoke --output-on-failure
```

Run only TypeScript variable-array runtime smoke:

```bash
ctest --test-dir build -R llvmdsdl-ts-runtime-variable-array-smoke --output-on-failure
```

Run only fixture C<->TypeScript variable-array parity smoke:

```bash
ctest --test-dir build -R llvmdsdl-fixtures-c-ts-variable-array-parity --output-on-failure
```

Run only TypeScript bigint runtime smoke:

```bash
ctest --test-dir build -R llvmdsdl-ts-runtime-bigint-smoke --output-on-failure
```

Run only fixture C<->TypeScript bigint parity smoke:

```bash
ctest --test-dir build -R llvmdsdl-fixtures-c-ts-bigint-parity --output-on-failure
```

Run only TypeScript union runtime smoke:

```bash
ctest --test-dir build -R llvmdsdl-ts-runtime-union-smoke --output-on-failure
```

Run only fixture C<->TypeScript union parity smoke:

```bash
ctest --test-dir build -R llvmdsdl-fixtures-c-ts-union-parity --output-on-failure
```

Run the full Go differential ring workflow:

```bash
cmake --workflow --preset go-differential
```

Run the full Go differential ring workflow (Homebrew LLVM):

```bash
cmake --workflow --preset go-differential-homebrew
```

Run the full Go differential ring workflow (LLVM env vars):

```bash
cmake --workflow --preset go-differential-llvm-env
```

This runs:

- `llvmdsdl-go-runtime-unit-tests`
- `llvmdsdl-signed-narrow-c-go-parity`
- `llvmdsdl-uavcan-go-generation`
- `llvmdsdl-uavcan-go-determinism`
- `llvmdsdl-uavcan-go-build`
- `llvmdsdl-uavcan-c-go-parity`

`llvmdsdl-uavcan-c-go-parity` now enforces directed baseline coverage for every
parity case: at least one truncation vector and at least one serialize-buffer
vector per case (auto-augmented when not explicitly listed), with summary
markers validated by `RunCGoParity.cmake`.
It also enforces inventory parity between the C harness wrappers and executed
random parity cases (`DEFINE_ROUNDTRIP` count must match observed `cases`).
The harness also validates no duplicate case/vector names and emits an inventory
summary marker (`PASS parity inventory ...`) that the CMake gate verifies.
Finally, the gate checks line-level execution counts for random and directed
pass markers to ensure summary totals match actual executed vectors.
The parity harness runners also isolate each invocation into a unique
per-run work directory under the configured output root, which avoids race
conditions when multiple workflows/tests execute concurrently.
Successful runs clean up their per-run scratch directories automatically, while
still writing stable summary files under the test output root.
The runners also remove legacy flat output subdirectories (`c/`, `go/`,
`build/`, `harness/`, and Go caches) from older harness layouts when present.
The signed-narrow C/Go parity gate also enforces:
- explicit inventory marker parity (`PASS signed-narrow inventory ...`)
- random/direct pass-line execution counts matching summary totals
- archive existence checks before Go linking
- atomic summary-file replacement (`*.tmp-*` then rename)
The other parity families now enforce the same inventory/pass-line invariants:
- `llvmdsdl-uavcan-c-rust-parity`
- `llvmdsdl-signed-narrow-c-rust-parity`
- `llvmdsdl-uavcan-cpp-c-parity`
- `llvmdsdl-uavcan-cpp-pmr-c-parity`
- `llvmdsdl-signed-narrow-cpp-c-parity`
- `llvmdsdl-signed-narrow-cpp-pmr-c-parity`

Current `uavcan` C/Go parity coverage includes representative service/message
families for:

- `uavcan.node` (`Heartbeat`, `ExecuteCommand`, `GetInfo`, `ID`, `Mode`, `Version`, `Health`, `IOStatistics`)
- `uavcan.node.port` (`List`, `ID`, `ServiceID`, `SubjectID`, `ServiceIDList`, `SubjectIDList`)
- `uavcan.register` (`Value`, `Access`, `Name`, `List`)
- `uavcan.file` (`Path`, `Error`, `List`, `Read`, `Write`, `Modify`, `GetInfo`)
- `uavcan.internet.udp` (`OutgoingPacket`, `HandleIncomingPacket`)
- `uavcan.time` (`Synchronization`, `SynchronizedTimestamp`, `TimeSystem`, `TAIInfo`, `GetSynchronizationMasterInfo`)
- `uavcan.diagnostic` (`Record`, `Severity`)
- `uavcan.metatransport.can` (`Frame`, `DataClassic`, `DataFD`, `Error`, `RTR`, `Manifestation`, `ArbitrationID`)
- `uavcan.metatransport.serial` (`Fragment`)
- `uavcan.metatransport.ethernet` (`Frame`, `EtherType`)
- `uavcan.metatransport.udp` (`Endpoint`, `Frame`)
- `uavcan.primitive` (`Empty`, `String`, `Unstructured`)
- `uavcan.primitive.scalar` (`Bit`, `Integer*`, `Natural*`, `Real*`)
- `uavcan.primitive.array` (`Bit`, `Integer*`, `Natural*`, `Real*`)
- `uavcan.pnp` (`NodeIDAllocationData`)
- `uavcan.pnp.cluster` (`Entry`, `AppendEntries`, `RequestVote`, `Discovery`)
- `uavcan.si.unit` (`angle`, `length`, `velocity`, `acceleration`, `force`, `torque`, `temperature`, `voltage`)
- `uavcan.si.sample` (`angle`, `velocity`, `acceleration`, `force`, `torque`, `temperature`, `voltage`)

Run only the generated-Rust compile gate:

```bash
ctest --test-dir build -R llvmdsdl-uavcan-rust-cargo-check --output-on-failure
```

Run only `uavcan` integration validation:

```bash
cmake --workflow --preset uavcan
```

macOS Homebrew LLVM workflow:

```bash
cmake --workflow --preset dev-homebrew
```

Self-contained tool bundle workflow (macOS/Homebrew LLVM):

```bash
cmake --workflow --preset self-contained-tools-homebrew
```

This produces relocatable `dsdlc` and `dsdl-opt` binaries plus their non-system
runtime `.dylib` dependencies under:

- `<build-dir>/self-contained-tools` (for example
  `build/dev-homebrew-self-contained/self-contained-tools`)
- `<build-dir>/self-contained-tools/MANIFEST.txt` with rewritten runtime links
  (`@executable_path/...` / `@loader_path/...`)

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
- `generate-uavcan-rust-no-std-alloc`
- `generate-uavcan-rust-runtime-fast`
- `generate-uavcan-rust-no-std-runtime-fast`
- `generate-uavcan-ts`
- `generate-uavcan-all` (aggregate)
- `generate-demo-2026-02-16` (demo bundle with logs + `DEMO.md`)

Run after configure/build:

```bash
cmake --build --preset build-dev-homebrew --target generate-uavcan-c
cmake --build --preset build-dev-homebrew --target generate-uavcan-cpp-std
cmake --build --preset build-dev-homebrew --target generate-uavcan-cpp-pmr
cmake --build --preset build-dev-homebrew --target generate-uavcan-rust-std
cmake --build --preset build-dev-homebrew --target generate-uavcan-rust-no-std-alloc
cmake --build --preset build-dev-homebrew --target generate-uavcan-rust-runtime-fast
cmake --build --preset build-dev-homebrew --target generate-uavcan-rust-no-std-runtime-fast
cmake --build --preset build-dev-homebrew --target generate-uavcan-ts
cmake --build --preset build-dev-homebrew --target generate-uavcan-all
cmake --build --preset build-dev-homebrew --target generate-demo-2026-02-16
```

Generated output paths are under `<build-dir>/generated/uavcan`:

- `<build-dir>/generated/uavcan/c`
- `<build-dir>/generated/uavcan/cpp-std`
- `<build-dir>/generated/uavcan/cpp-pmr`
- `<build-dir>/generated/uavcan/cpp-both`
- `<build-dir>/generated/uavcan/rust-std`
- `<build-dir>/generated/uavcan/rust-no-std-alloc`
- `<build-dir>/generated/uavcan/rust-runtime-fast`
- `<build-dir>/generated/uavcan/rust-no-std-runtime-fast`
- `<build-dir>/generated/uavcan/ts`
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

### C header generation

```bash
./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-out
```

Optional:

- `--optimize-lowered-serdes`: enable optional semantics-preserving MLIR
  optimization on lowered SerDes IR before backend emission.
- No additional C mode flags are required: `dsdlc c` always emits headers and
  per-definition implementation translation units.

### C++23 header generation (`std`/`pmr`)

Generate both profiles:

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --cpp-profile both \
  --out-dir build/uavcan-cpp-out
```

Generate only one profile:

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --cpp-profile std \
  --out-dir build/uavcan-cpp-std-out
```

```bash
./build/tools/dsdlc/dsdlc cpp \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
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

### Rust crate generation (`std`/`no-std-alloc` + runtime specialization)

```bash
./build/tools/dsdlc/dsdlc rust \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-rust-out \
  --rust-crate-name uavcan_dsdl_generated \
  --rust-profile std
```

No-std+alloc profile:

```bash
./build/tools/dsdlc/dsdlc rust \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-rust-no-std-out \
  --rust-crate-name uavcan_dsdl_generated_no_std \
  --rust-profile no-std-alloc
```

Runtime-specialized std profile:

```bash
./build/tools/dsdlc/dsdlc rust \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-rust-fast-out \
  --rust-crate-name uavcan_dsdl_generated_fast \
  --rust-profile std \
  --rust-runtime-specialization fast
```

Current behavior:

- `--rust-profile std` is the default and recommended first path.
- `--rust-profile no-std-alloc` emits a crate configured for `no_std` with
  `alloc` (Cargo default features are empty; `std` remains an opt-in feature).
- `--rust-runtime-specialization portable` is the default profile behavior.
- `--rust-runtime-specialization fast` enables a runtime-optimized bit-copy path
  via Cargo feature `runtime-fast` while keeping generated semantic type files
  unchanged.
- Runtime-specialization integration gates include C/Rust parity verification for
  both `std + fast` and `no-std-alloc + fast` profile combinations.
- Generated Rust API uses `DsdlVec` aliasing in `dsdl_runtime` so profile
  changes do not alter lowered wire semantics.
- Generated Rust types expose both:
- `deserialize(&mut self, &[u8]) -> Result<usize, i8>` (ergonomic path), and
- `deserialize_with_consumed(&mut self, &[u8]) -> (i8, usize)` (C-like parity path that reports consumed bytes on error too).

### TypeScript module generation (non-C-like target)

```bash
./build/tools/dsdlc/dsdlc ts \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-ts-out \
  --ts-module uavcan_dsdl_generated_ts
```

Runtime-specialized fast profile:

```bash
./build/tools/dsdlc/dsdlc ts \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir build/uavcan-ts-fast-out \
  --ts-module uavcan_dsdl_generated_ts_fast \
  --ts-runtime-specialization fast
```

Current behavior:

- Emits `package.json`, `index.ts`, one namespace-mirrored `*.ts` file per DSDL
  definition, and shared runtime module `dsdl_runtime.ts`.
- `index.ts` exports each generated definition module under a collision-safe
  namespace alias.
- Generates interface/type declarations, DSDL metadata constants
  (`DSDL_FULL_NAME`, version major/minor), and runtime-backed
  `serialize*`/`deserialize*` helpers.
- `--ts-runtime-specialization portable` is the default profile behavior.
- `--ts-runtime-specialization fast` enables byte-aligned fast paths in
  `dsdl_runtime.ts` while preserving semantic generated type files.
- Runtime-specialization integration gates include generation, typecheck,
  C<->TS parity, and semantic-diff checks between portable and fast outputs.
- Uses lowered-contract validation (`collectLoweredFactsFromMlir`) and shared
  lowered render-order planning for runtime section emission.
- Hard-fails generation if required lowered runtime planning metadata is
  missing/inconsistent.
- Integration gates verify no fallback runtime stub signatures in generated
  fixture and full-`uavcan` TypeScript output.
- Integration coverage includes:
  - full-`uavcan` generation/determinism/typecheck/consumer-smoke/index-contract/runtime-execution gates
  - runtime smoke lanes across scalar/array/union/composite/delimited/service/padding/truncated-decode families
  - C<->TS parity lanes including invariant-based random+direct checks, signed-narrow cast-mode checks, and optimized-lowering variants

## Reproducible Full `uavcan` Generation Check

```bash
OUT="build/uavcan-out"
mkdir -p "${OUT}"

./build/tools/dsdlc/dsdlc c \
  --root-namespace-dir public_regulated_data_types/uavcan \ \
  --out-dir "${OUT}"

find public_regulated_data_types/uavcan -name '*.dsdl' | wc -l
find "${OUT}" -name '*.h' ! -name 'dsdl_runtime.h' | wc -l
```

The two counts should match.

## Status Snapshot

Current milestone supports generating all types under:

- `public_regulated_data_types/uavcan`

with no `dsdl_runtime_stub_*` references in generated headers, C++23 generation
(`std` + `pmr` profiles), Rust generation in `std`, `no-std-alloc`, and
runtime-specialized (`fast`) modes, plus TypeScript generation with compile
(`tsc --noEmit`), determinism, consumer-smoke, index-contract, runtime
execution, runtime-specialized (`portable|fast`) generation/typecheck/parity/
semantic-diff gates, and fixture/runtime parity validation lanes (including
signed-narrow parity, optimized parity, variable-array/bigint/union/composite/
service/delimited families).
