//===----------------------------------------------------------------------===//
///
/// @file
/// Public entry points and options for Go backend emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_GOEMITTER_H
#define LLVMDSDL_CODEGEN_GOEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl
{

/// @file
/// @brief Go backend emission entry points.

/// @brief Configuration options for Go code generation.
struct GoEmitOptions final
{
    /// @brief Output directory root.
    std::string outDir;

    /// @brief Generated Go module name.
    std::string moduleName{"llvmdsdl_generated"};

    /// @brief Emits `go.mod` when true.
    bool emitGoMod{true};

    /// @brief Enables optional lowered-serdes optimization before emission.
    bool optimizeLoweredSerDes{false};
};

/// @brief Emits Go artifacts from semantic and lowered MLIR inputs.
/// @param[in] semantic Resolved semantic module.
/// @param[in] module Lowered MLIR module.
/// @param[in] options Backend configuration.
/// @param[in,out] diagnostics Diagnostic sink.
/// @return Success or detailed failure.
llvm::Error emitGo(const SemanticModule& semantic,
                   mlir::ModuleOp        module,
                   const GoEmitOptions&  options,
                   DiagnosticEngine&     diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_GOEMITTER_H
