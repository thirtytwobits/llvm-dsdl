//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Entry points for lowering the semantic model into DSDL MLIR.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_LOWERING_LOWERTOMLIR_H
#define LLVMDSDL_LOWERING_LOWERTOMLIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir
{
class MLIRContext;
}

namespace llvmdsdl
{
class DiagnosticEngine;
struct SemanticModule;

/// @file
/// @brief Semantic-to-MLIR lowering entry points.

/// @brief Lowers semantic definitions into the DSDL MLIR dialect.
/// @param[in] module Semantic module to lower.
/// @param[in,out] context MLIR context owning produced operations.
/// @param[in,out] diagnostics Diagnostic sink for lowering failures.
/// @return Owning MLIR module reference.
mlir::OwningOpRef<mlir::ModuleOp> lowerToMLIR(const SemanticModule& module,
                                              mlir::MLIRContext&    context,
                                              DiagnosticEngine&     diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_LOWERING_LOWERTOMLIR_H
