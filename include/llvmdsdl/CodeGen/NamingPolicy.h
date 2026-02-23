//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared naming-policy helpers for backend code generation.
///
/// This interface centralizes language-specific identifier sanitization and
/// common case projections (snake/pascal/upper-snake) used by emitters.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_NAMING_POLICY_H
#define LLVMDSDL_CODEGEN_NAMING_POLICY_H

#include <string>

#include "llvm/ADT/StringRef.h"

namespace llvmdsdl
{

/// @brief Target language for identifier naming projection.
enum class CodegenNamingLanguage
{
    /// @brief C naming policy.
    C,

    /// @brief C++ naming policy.
    Cpp,

    /// @brief Rust naming policy.
    Rust,

    /// @brief Go naming policy.
    Go,

    /// @brief TypeScript naming policy.
    TypeScript,

    /// @brief Python naming policy.
    Python,
};

/// @brief Returns true when an identifier is a keyword in the target language.
/// @param[in] language Naming language.
/// @param[in] name Candidate identifier.
/// @return True when the identifier is reserved.
bool codegenIsKeyword(CodegenNamingLanguage language, llvm::StringRef name);

/// @brief Sanitizes one identifier for the target language.
/// @param[in] language Naming language.
/// @param[in] name Candidate identifier.
/// @return Language-safe identifier.
std::string codegenSanitizeIdentifier(CodegenNamingLanguage language, llvm::StringRef name);

/// @brief Projects text into snake_case and sanitizes for the target language.
/// @param[in] language Naming language.
/// @param[in] name Source text.
/// @return Language-safe snake_case identifier.
std::string codegenToSnakeCaseIdentifier(CodegenNamingLanguage language, llvm::StringRef name);

/// @brief Projects text into PascalCase and sanitizes for the target language.
/// @param[in] language Naming language.
/// @param[in] name Source text.
/// @return Language-safe PascalCase identifier.
std::string codegenToPascalCaseIdentifier(CodegenNamingLanguage language, llvm::StringRef name);

/// @brief Projects text into UPPER_SNAKE_CASE and sanitizes for the target language.
/// @param[in] language Naming language.
/// @param[in] name Source text.
/// @return Language-safe UPPER_SNAKE_CASE identifier.
std::string codegenToUpperSnakeCaseIdentifier(CodegenNamingLanguage language, llvm::StringRef name);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_NAMING_POLICY_H
