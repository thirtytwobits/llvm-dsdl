# llvm-dsdl Design

## 1. Program Design

### 1.1 Goals

`llvm-dsdl` is designed to be a strict, reproducible DSDL compiler with:

- Spec-first frontend behavior (strict by default).
- A typed semantic model that can support multiple backends.
- An LLVM/MLIR foundation for long-term compiler evolution.
- Multi-language code generation with shared wire semantics.

The project currently supports:

- C (`dsdlc c`)
- C++23 (`dsdlc cpp`, with `std` and `pmr` profiles)
- Rust (`dsdlc rust`, currently `std` profile)

for the `uavcan` namespace under regulated data types.

### 1.2 High-Level Architecture

```mermaid
flowchart LR
  A["DSDL files"] --> B["Frontend: discovery + lexer + parser"]
  B --> C["AST"]
  C --> D["Semantic analyzer + evaluator"]
  D --> E["Semantic model"]
  E --> F["MLIR DSDL lowering"]
  F --> G["MLIR module"]
  G --> H["C path: EmitC lowering + C output (per-definition impl TUs)"]
  E --> I["C/C++/Rust header emitters"]
  I --> J["Generated language artifacts"]
```

### 1.3 Pipeline Stages

#### Frontend

The frontend discovers `.dsdl` files, validates namespace and version/file naming
conventions, tokenizes/parses, and builds AST with source locations.

Primary modules:

- `include/llvmdsdl/Frontend/*`
- `lib/Frontend/*`

#### Semantics

Semantic analysis resolves type references, constants, directives, array bounds,
union constraints, and bit-length/extent information. This stage produces a typed,
resolved `SemanticModule` used by code generators.

Primary modules:

- `include/llvmdsdl/Semantics/*`
- `lib/Semantics/*`

#### IR / MLIR

The project defines a custom DSDL MLIR dialect and lowering hooks. This creates a
compiler-grade intermediate representation suitable for validation/transforms and
future target backends.

Primary modules:

- `include/llvmdsdl/IR/*`
- `lib/IR/*`
- `lib/Lowering/*`
- `lib/Transforms/*`

#### Code Generation

Current generators are in `lib/CodeGen`:

- `CEmitter.cpp`
  - Emits per-type C headers and runtime header.
  - Emits per-definition C implementation translation units through EmitC lowering.
- `CppEmitter.cpp`
  - Emits namespace-based C++23 headers.
  - Supports `std` and `pmr` profiles.
- `RustEmitter.cpp`
  - Emits crate/module layout and Rust SerDes/runtime integration.

#### Runtime Layer

Runtime helpers encapsulate wire-level bit operations and numeric conversions:

- `runtime/dsdl_runtime.h` (C core)
- `runtime/cpp/dsdl_runtime.hpp` (C++ wrapper)
- `runtime/rust/dsdl_runtime.rs` (Rust runtime)

### 1.4 Tooling and Validation

- `dsdlc` is the main CLI frontend/driver.
- `dsdl-opt` supports MLIR pass experimentation.
- Integration tests verify strict full-tree generation and compile checks.
- CMake workflow presets provide reproducible configure/build/test automation.

---

## 2. Comparison: LLVM vs Nunavut+pydsdl vs Native Non-LLVM

This section compares three implementation approaches:

1. **llvm-dsdl (LLVM/MLIR-based)**  
2. **pydsdl + nunavut (Python reference ecosystem)**  
3. **Native non-LLVM implementation (hypothetical C++ implementation without MLIR/LLVM)**

### 2.1 Feature Matrix

| Capability | llvm-dsdl (LLVM/MLIR) | pydsdl + nunavut | Native Non-LLVM (hypothetical) |
|---|---|---|---|
| Core language/runtime implementation | C++ | Python (+ templates) | C++ |
| DSDL parser + semantics | Yes | Yes (mature reference behavior) | Yes (must build from scratch) |
| Canonical compiler IR | Yes (custom MLIR dialect) | No compiler IR layer | Optional custom IR (must design/maintain) |
| Pass manager and rewrite infra | Yes (MLIR passes/patterns) | No | Must implement custom pass infra |
| Built-in verifier hooks | Yes (dialect/op verifiers) | Limited (library-level checks) | Must implement custom verifier framework |
| Optimization framework | Yes (MLIR + LLVM ecosystem) | Minimal | Custom optimizer required |
| Multi-target codegen scaling | Strong long-term fit | Template-dependent and target-specific | Medium; backend-specific effort |
| Debug/inspection infrastructure | Strong (IR dump/pass pipelines) | Python-level debugging | Custom tooling required |
| Build/dependency complexity | High | Low/medium | Medium |
| Startup and iteration speed | Fast runtime binaries, slower compile/build | Fast scripting iteration | Medium |
| Ecosystem interoperability | Excellent with compiler/toolchain stack | Excellent with existing OpenCyphal workflows | Depends on design |
| Best fit | Long-term compiler platform | Proven generator stack and quick adoption | Lightweight custom stack |

### 2.2 What LLVM/MLIR Is Doing for Us Here

In this project, LLVM/MLIR provides:

- A **structured intermediate representation** for DSDL concepts.
- **Pass composition** for transformation, canonicalization, and validation.
- A path to **incremental lowering** from DSDL-level semantics to target-level code.
- Existing conversion/translation infrastructure (e.g., EmitC path for C impl output).
- Better long-term maintainability for advanced compiler features than ad-hoc
  backend-specific code paths.

Without LLVM, the project would need to build and maintain its own:

- IR and verifier system.
- Pass manager and transformation pipeline.
- Canonicalization/rewrite engine.
- Lowering framework for multiple backends.

### 2.3 Trade-Off Summary

#### llvm-dsdl (LLVM/MLIR)

Pros:

- Compiler-grade architecture with strong extensibility.
- Better fit for multiple language backends and deeper optimization.
- Strong validation and transformation model.

Cons:

- More complexity (build, dependencies, contributor onboarding).
- Higher initial implementation cost.

#### pydsdl + nunavut

Pros:

- Mature, field-proven ecosystem in OpenCyphal workflows.
- Very fast to get started.
- Strong compatibility expectations for existing users.

Cons:

- Not centered around compiler IR/passes.
- Harder to evolve into advanced optimization/transformation architecture.

#### Native Non-LLVM C++ (No MLIR)

Pros:

- Simpler dependency profile than LLVM.
- Potentially easier to keep minimal at small scope.

Cons:

- Significant custom infrastructure burden for anything beyond basic generation.
- Long-term scaling cost for multi-language + optimization + analysis features.

---

## 3. Future Expansion Enabled by LLVM/MLIR

Because the project uses LLVM/MLIR, future work can go beyond “template-based
codegen” into full compiler capabilities.

### 3.1 Near-Term Opportunities

- Move all language backends toward a shared MLIR-driven serialization plan.
- Add stronger IR verifiers for union/extent/array correctness invariants.
- Add target-specific lowering passes (C++/Rust) from common plan ops.
- Improve diagnostics with IR-level provenance and source mapping.

### 3.2 Mid-Term Opportunities

- Backend optimization passes:
  - dead-field elimination for constant/default paths,
  - loop simplification for fixed-size arrays,
  - inlining/specialization for nested composites.
- Alternate runtime strategies generated from profile-aware lowering
  (freestanding, PMR, `no_std + alloc`, etc.).
- Cross-language consistency checks from one canonical IR.

### 3.3 Long-Term Opportunities

- Additional target languages (e.g., Rust `no_std`, C++, Zig, others) from shared IR.
- Static analysis and linting passes directly on DSDL IR.
- Formal conformance tools using IR-level property checks.
- Advanced tooling (`dsdl-opt` pipelines, IR debugging, regression reduction).
- Optional lowerings to other MLIR/LLVM dialects for specialized environments.

---

## 4. Current State vs Target State

Current:

- Frontend + semantics are shared and mature enough for full `uavcan` generation.
- C++/Rust generation is implemented and tested for strict tree generation.
- C path already has an EmitC lowering route for implementation translation units.
- Current implementation emits one `.c` translation unit per DSDL definition
  (monolithic TU-only mode has been removed).

Target trajectory:

- Continue converging backend behavior onto MLIR-first lowering for deeper reuse
  and optimization.
- Keep language/profile APIs stable while strengthening wire-level conformance and
  differential validation.
