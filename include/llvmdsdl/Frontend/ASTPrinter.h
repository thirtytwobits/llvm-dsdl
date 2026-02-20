//===----------------------------------------------------------------------===//
///
/// @file
/// AST pretty-printing declarations for debugging and diagnostics.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_FRONTEND_AST_PRINTER_H
#define LLVMDSDL_FRONTEND_AST_PRINTER_H

#include <string>

namespace llvmdsdl
{
struct ASTModule;

/// @file
/// @brief AST pretty-printer entry points.

/// @brief Produces a human-readable representation of an AST module.
/// @param[in] module Parsed AST module.
/// @return Pretty-printed AST text.
std::string printAST(const ASTModule& module);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_FRONTEND_AST_PRINTER_H
