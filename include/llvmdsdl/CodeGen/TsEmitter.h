//===----------------------------------------------------------------------===//
///
/// @file
/// Public entry points and options for TypeScript backend emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_TS_EMITTER_H
#define LLVMDSDL_CODEGEN_TS_EMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl
{

/// @file
/// @brief TypeScript backend emission entry points.

/// @brief Configuration options for TypeScript code generation.
struct TsEmitOptions final
{
    /// @brief Output directory root.
    std::string outDir;

    /// @brief Generated npm/module name.
    std::string moduleName{"llvmdsdl_generated"};

    /// @brief Emits package metadata when true.
    bool emitPackageJson{true};

    /// @brief Enables optional lowered-serdes optimization before emission.
    bool optimizeLoweredSerDes{false};
};

/// @brief Emits TypeScript artifacts from semantic and lowered MLIR inputs.
/// @param[in] semantic Resolved semantic module.
/// @param[in] module Lowered MLIR module.
/// @param[in] options Backend configuration.
/// @param[in,out] diagnostics Diagnostic sink.
/// @return Success or detailed failure.
llvm::Error emitTs(const SemanticModule& semantic,
                   mlir::ModuleOp        module,
                   const TsEmitOptions&  options,
                   DiagnosticEngine&     diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_TS_EMITTER_H
