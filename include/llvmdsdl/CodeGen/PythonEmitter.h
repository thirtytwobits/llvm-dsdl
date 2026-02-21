//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Public entry points and options for Python backend emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_PYTHON_EMITTER_H
#define LLVMDSDL_CODEGEN_PYTHON_EMITTER_H

#include <string>

#include "llvm/Support/Error.h"

namespace mlir
{
class ModuleOp;
}  // namespace mlir

namespace llvmdsdl
{
class DiagnosticEngine;
struct SemanticModule;

/// @file
/// @brief Python backend emission entry points.

/// @brief Configuration options for Python code generation.
struct PythonEmitOptions final
{
    /// @brief Output directory root.
    std::string outDir;

    /// @brief Generated Python package name.
    std::string packageName{"dsdl_gen"};

    /// @brief Enables optional lowered-serdes optimization before emission.
    bool optimizeLoweredSerDes{false};
};

/// @brief Emits Python artifacts from semantic and lowered MLIR inputs.
/// @param[in] semantic Resolved semantic module.
/// @param[in] module Lowered MLIR module.
/// @param[in] options Backend configuration.
/// @param[in,out] diagnostics Diagnostic sink.
/// @return Success or detailed failure.
llvm::Error emitPython(const SemanticModule&    semantic,
                       mlir::ModuleOp           module,
                       const PythonEmitOptions& options,
                       DiagnosticEngine&        diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_PYTHON_EMITTER_H
