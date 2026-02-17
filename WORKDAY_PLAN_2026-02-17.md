# Workday Plan: MLIR-First Convergence (Tuesday, Feb 17, 2026)

Start time: 10:26 AM PST  
End goal: 8:00 PM PST

## 8:00 PM Goal

Ship a clear **“shared MLIR SerDes core v1”** milestone:

1. C++ and Rust SerDes generation use shared planning/render contracts for the majority of field-level behavior (array/scalar/composite/union paths).
2. Full validation remains green (`ctest` full suite + focused parity ring).
3. We can show concrete evidence of convergence (new shared modules + reduced backend-specific branching + updated design/docs).

## Definition Of Done (By 8:00 PM)

1. Shared CodeGen modules expanded to cover:
   - scalar statement shaping
   - composite/delimited statement shaping
   - remaining array/union control-flow glue where still duplicated
2. C++ and Rust emitters consume the shared contracts for those paths.
3. Tests:
   - unit tests for each new shared module
   - focused convergence ring green
   - full `ctest` green
4. Documentation updated with today’s status and what is still backend-specific.
5. Demo-ready command sequence captured for reproducibility.

## Time-Boxed Schedule

### 10:30 AM - 1:00 PM: Shared Statement Core Expansion

1. Add shared scalar/composite wire-plan modules.
2. Move duplicated C++/Rust scalar/composite prelude logic into shared code.
3. Add unit tests for new modules.

Gate @ 1:00 PM:
- New shared modules compile and unit tests pass.

### 1:00 PM - 3:30 PM: Rewire C++/Rust Emitters

1. Rewire C++ statement emission to use new shared plans.
2. Rewire Rust statement emission to use same plans.
3. Keep language-specific syntax emission thin.

Gate @ 3:30 PM:
- C++/Rust generation still succeeds for full `uavcan`.

### 3:30 PM - 5:30 PM: Validation + Parity Hardening

1. Run focused ring and fix regressions.
2. Add/adjust directed parity checks for any newly touched behavior.
3. Run full `ctest` and stabilize.

Gate @ 5:30 PM:
- Focused ring and full suite green.

### 5:30 PM - 7:00 PM: Docs + Demo Artifacts

1. Update `DESIGN.md` with today’s convergence delta.
2. Update `README.md` runbook commands (if changed).
3. Capture demo transcript/artifacts for tonight’s checkpoint.

Gate @ 7:00 PM:
- Docs reflect current architecture accurately.

### 7:00 PM - 8:00 PM: Freeze + Buffer

1. Final regression run.
2. Clean summary of delivered work + known gaps.
3. Freeze branch state for tomorrow’s continuation.

## Priority Stack (If Time Compresses)

1. Keep full suite green.
2. Deliver shared scalar/composite planning + rewire both C++/Rust.
3. Update `DESIGN.md` truthfully.
4. Stretch: additional directed parity vectors.

## Risks and Controls

1. Risk: regressions in cross-language parity.
   - Control: run focused ring after each major refactor slice.
2. Risk: over-refactor without measurable progress.
   - Control: require each slice to add tests + remove concrete duplicate paths.
3. Risk: late-day instability.
   - Control: stop feature work by 7:00 PM and reserve 1-hour stabilization buffer.

## Command Checklist

1. Build:
   - `cmake --build --preset build-dev-homebrew --target dsdlc llvmdsdl-unit-tests`
2. Focused ring:
   - `ctest --test-dir build/dev-homebrew -R "llvmdsdl-unit-tests|llvmdsdl-uavcan-mlir-lowering|llvmdsdl-uavcan-rust-generation|llvmdsdl-uavcan-cpp-generation|llvmdsdl-uavcan-cpp-c-parity|llvmdsdl-uavcan-cpp-pmr-c-parity|llvmdsdl-uavcan-c-rust-parity|llvmdsdl-signed-narrow-cpp-c-parity|llvmdsdl-signed-narrow-cpp-pmr-c-parity|llvmdsdl-signed-narrow-c-rust-parity" --output-on-failure`
3. Full suite:
   - `ctest --test-dir build/dev-homebrew --output-on-failure`

## Success Narrative For Tonight

“We now have a shared MLIR-driven SerDes planning core used by C++, Rust, and existing C paths where applicable, with parity gates still green and a clearer runway to additional target languages with less backend-specific code.”
