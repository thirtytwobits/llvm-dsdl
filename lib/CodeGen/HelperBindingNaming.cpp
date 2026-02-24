//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements shared naming helpers for lowered helper-binding symbols.
///
/// This utility preserves consistent helper binding naming across backends.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/HelperBindingNaming.h"

namespace llvmdsdl
{

std::string renderHelperBindingIdentifier(const CodegenNamingLanguage language, const llvm::StringRef helperSymbol)
{
    return "mlir_" + codegenSanitizeIdentifier(language, helperSymbol);
}

}  // namespace llvmdsdl
