//===----------------------------------------------------------------------===//
///
/// @file
/// Public entry points and options for Rust backend emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_RUSTEMITTER_H
#define LLVMDSDL_CODEGEN_RUSTEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl
{

/// @file
/// @brief Rust backend emission entry points.

/// @brief Rust crate profile selection.
enum class RustProfile
{

    /// @brief `std` profile.
    Std,

    /// @brief `no_std` plus `alloc` profile.
    NoStdAlloc,
};

/// @brief Runtime implementation specialization for generated Rust helpers.
enum class RustRuntimeSpecialization
{

    /// @brief Portable baseline implementation.
    Portable,

    /// @brief Faster specialized implementation.
    Fast,
};

/// @brief Configuration options for Rust code generation.
struct RustEmitOptions final
{
    /// @brief Output directory root.
    std::string outDir;

    /// @brief Generated crate name.
    std::string crateName{"llvmdsdl_generated"};

    /// @brief Emits Cargo metadata when true.
    bool emitCargoToml{true};

    /// @brief Requested crate profile.
    RustProfile profile{RustProfile::Std};

    /// @brief Requested runtime specialization.
    RustRuntimeSpecialization runtimeSpecialization{RustRuntimeSpecialization::Portable};

    /// @brief Enables optional lowered-serdes optimization before emission.
    bool optimizeLoweredSerDes{false};
};

/// @brief Emits Rust artifacts from semantic and lowered MLIR inputs.
/// @param[in] semantic Resolved semantic module.
/// @param[in] module Lowered MLIR module.
/// @param[in] options Backend configuration.
/// @param[in,out] diagnostics Diagnostic sink.
/// @return Success or detailed failure.
llvm::Error emitRust(const SemanticModule&  semantic,
                     mlir::ModuleOp         module,
                     const RustEmitOptions& options,
                     DiagnosticEngine&      diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_RUSTEMITTER_H
