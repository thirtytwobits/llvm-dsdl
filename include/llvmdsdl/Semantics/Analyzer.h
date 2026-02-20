//===----------------------------------------------------------------------===//
///
/// @file
/// Public semantic analysis entry points converting AST modules into semantic modules.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_SEMANTICS_ANALYZER_H
#define LLVMDSDL_SEMANTICS_ANALYZER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvm/Support/Error.h"

namespace llvmdsdl
{
class DiagnosticEngine;
struct ASTModule;

/// @file
/// @brief Semantic analysis entry points.

/// @brief Converts parsed AST into the resolved semantic model.
/// @param[in] module Parsed AST module.
/// @param[in,out] diagnostics Diagnostic sink for semantic issues.
/// @return Resolved semantic module on success.
llvm::Expected<SemanticModule> analyze(const ASTModule& module, DiagnosticEngine& diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_SEMANTICS_ANALYZER_H
