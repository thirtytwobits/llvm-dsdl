//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared constant-literal rendering helpers for code generation backends.
///
/// These APIs centralize backend-specific literal syntax for booleans, numeric
/// constants, and strings emitted from semantic constant values.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_CONSTANT_LITERAL_RENDER_H
#define LLVMDSDL_CODEGEN_CONSTANT_LITERAL_RENDER_H

#include <string>

#include "llvmdsdl/Semantics/Evaluator.h"

namespace llvmdsdl
{

/// @brief Target language for constant-literal rendering.
enum class ConstantLiteralLanguage
{
    /// @brief C literal syntax.
    C,

    /// @brief C++ literal syntax.
    Cpp,

    /// @brief Rust literal syntax.
    Rust,

    /// @brief Go literal syntax.
    Go,

    /// @brief TypeScript literal syntax.
    TypeScript,

    /// @brief Python literal syntax.
    Python,
};

/// @brief Renders a semantic constant value as source code for one language.
/// @param[in] language Target language syntax.
/// @param[in] value Semantic constant value.
/// @return Rendered constant expression.
std::string renderConstantLiteral(ConstantLiteralLanguage language, const Value& value);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_CONSTANT_LITERAL_RENDER_H
