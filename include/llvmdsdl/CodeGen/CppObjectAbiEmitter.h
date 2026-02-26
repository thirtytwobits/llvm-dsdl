//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Canonical profile-agnostic C++ ABI staging for object emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_CPPOBJECTABIEMITTER_H
#define LLVMDSDL_CODEGEN_CPPOBJECTABIEMITTER_H

#include "llvmdsdl/CodeGen/EmitCommon.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include "llvm/Support/Error.h"

#include <filesystem>
#include <string>
#include <vector>

namespace llvmdsdl
{
struct SemanticModule;

/// @brief Configuration options for canonical C++ ABI object staging.
struct CppObjectAbiEmitOptions final
{
    /// @brief Root directory for generated C++ object-stage artifacts.
    std::filesystem::path stageRoot;

    /// @brief Root directory of generated C stage artifacts consumed as wire-core implementation.
    std::filesystem::path cStageRoot;

    /// @brief Optional selected type keys for filtered emission.
    std::vector<std::string> selectedTypeKeys;

    /// @brief Shared write policy.
    EmitWritePolicy writePolicy;
};

/// @brief Generates canonical C++ ABI headers/sources, adapters, and C shim artifacts.
/// @param[in] semantic Semantic module closure.
/// @param[in] loweredFacts Lowered section facts including zero-overhead metadata.
/// @param[in] options C++ ABI stage options.
/// @param[out] outCppSources Generated C++ translation units for compilation.
/// @return Success or descriptive generation error.
llvm::Error emitCppObjectAbiStage(const SemanticModule&         semantic,
                                  const LoweredFactsMap&        loweredFacts,
                                  const CppObjectAbiEmitOptions& options,
                                  std::vector<std::filesystem::path>* outCppSources);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_CPPOBJECTABIEMITTER_H
