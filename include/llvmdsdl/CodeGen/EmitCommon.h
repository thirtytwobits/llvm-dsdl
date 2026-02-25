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
#include <unordered_map>
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

    /// @brief Optional sink of generated-output path to required type keys.
    std::unordered_map<std::string, std::vector<std::string>>* recordedOutputRequiredTypeKeys{nullptr};
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
llvm::Error writeGeneratedFile(const std::filesystem::path&    path,
                               llvm::StringRef                 content,
                               const EmitWritePolicy&          policy,
                               const std::vector<std::string>& requiredTypeKeys = {});

/// @brief Renders one make-style depfile body.
///
/// @details
/// The output format is: `<escaped_target>: <escaped_dep_1> <escaped_dep_2> ...\n`.
/// Dependency inputs are sorted and de-duplicated for deterministic output.
///
/// @param[in] target Make-rule target path.
/// @param[in] deps Dependency path list.
/// @return Rendered depfile text with trailing newline.
std::string renderMakeDepfile(const std::string& target, const std::vector<std::string>& deps);

/// @brief Writes `<outputPath>.d` make depfile for one generated output path.
///
/// @details
/// Dependency paths and target path are normalized to absolute lexical paths.
/// The write path obeys @ref EmitWritePolicy semantics (`dryRun`,
/// `noOverwrite`, `fileMode`, and `recordedOutputs`).
///
/// @param[in] outputPath Generated output path the depfile describes.
/// @param[in] deps Dependency path list.
/// @param[in] policy Write policy.
/// @return Success or a descriptive I/O error.
llvm::Error writeDepfileForGeneratedOutput(const std::filesystem::path&    outputPath,
                                           const std::vector<std::string>& deps,
                                           const EmitWritePolicy&          policy);

/// @brief Writes `<outputPath>.d` depfile from pre-normalized sorted dependencies.
///
/// @details
/// This fast path assumes `normalizedSortedDedupDeps` are already absolute (or
/// otherwise final-form), sorted, and de-duplicated. No dependency
/// normalization, sorting, or de-duplication is performed in this overload.
///
/// @param[in] outputPath Generated output path the depfile describes.
/// @param[in] normalizedSortedDedupDeps Prepared dependency path list.
/// @param[in] policy Write policy.
/// @return Success or a descriptive I/O error.
llvm::Error writeDepfileForGeneratedOutputPrepared(const std::filesystem::path&    outputPath,
                                                   const std::vector<std::string>& normalizedSortedDedupDeps,
                                                   const EmitWritePolicy&          policy);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_EMITCOMMON_H
