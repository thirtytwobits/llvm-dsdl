//===----------------------------------------------------------------------===//
///
/// @file
/// Public entry points and options for C backend emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_CEMITTER_H
#define LLVMDSDL_CODEGEN_CEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl
{

/// @file
/// @brief C backend emission entry points.

/// @brief Configuration options for C code generation.
struct CEmitOptions final
{
    /// @brief Output directory root for generated files.
    std::string outDir;

    /// @brief Emits C89-style top-of-block variable declarations when true.
    bool declareVariablesAtTop{true};

    /// @brief Enables optional lowered-serdes optimization before emission.
    bool optimizeLoweredSerDes{false};
};

/// @brief Emits C artifacts from semantic and lowered MLIR inputs.
/// @param[in] semantic Resolved semantic module.
/// @param[in] module Lowered MLIR module.
/// @param[in] options Backend configuration.
/// @param[in,out] diagnostics Diagnostic sink.
/// @return Success or detailed failure.
llvm::Error emitC(const SemanticModule& semantic,
                  mlir::ModuleOp        module,
                  const CEmitOptions&   options,
                  DiagnosticEngine&     diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_CEMITTER_H
