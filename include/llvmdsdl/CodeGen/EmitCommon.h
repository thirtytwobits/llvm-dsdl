//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared emission helpers for file-write policy and type-key selection.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_EMITCOMMON_H
#define LLVMDSDL_CODEGEN_EMITCOMMON_H

#include "llvmdsdl/Frontend/AST.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace llvmdsdl
{

/// @brief Output-file write policy shared by all code generators.
struct EmitWritePolicy final
{
    /// @brief Do not create or modify any files.
    bool dryRun{false};

    /// @brief Reject writes when destination file already exists.
    bool noOverwrite{false};

    /// @brief File mode applied after writing (POSIX-like bitmask).
    std::uint32_t fileMode{0444U};

    /// @brief Optional sink of absolute generated output paths.
    std::vector<std::string>* recordedOutputs{nullptr};
};

/// @brief Returns a canonical type key for one discovered definition.
/// @param[in] info Definition metadata.
/// @return Key in the form `fullName:major:minor`.
std::string definitionTypeKey(const DiscoveredDefinition& info);

/// @brief Builds a hash-set from a list of type keys.
/// @param[in] typeKeys Input key list.
/// @return Hash-set containing all keys.
std::unordered_set<std::string> makeTypeKeySet(const std::vector<std::string>& typeKeys);

/// @brief Indicates whether one definition should be emitted.
/// @param[in] info Definition metadata.
/// @param[in] selectedTypeKeys Optional selected key-set.
/// @return True if the definition is selected or if no selection is provided.
bool shouldEmitDefinition(const DiscoveredDefinition& info, const std::unordered_set<std::string>& selectedTypeKeys);

/// @brief Writes one generated file under a policy.
///
/// @details
/// When @ref EmitWritePolicy::dryRun is true, no filesystem mutation occurs.
/// In all modes, if @ref EmitWritePolicy::recordedOutputs is set, the resolved
/// absolute path is appended.
///
/// @param[in] path Destination file path.
/// @param[in] content File contents.
/// @param[in] policy Write policy.
/// @return Success or a descriptive I/O error.
llvm::Error writeGeneratedFile(const std::filesystem::path& path, llvm::StringRef content, const EmitWritePolicy& policy);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_EMITCOMMON_H
