//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Cached dependency planner for per-output depfile input resolution.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_FRONTEND_DEPFILEPLANNER_H
#define LLVMDSDL_FRONTEND_DEPFILEPLANNER_H

#include "llvmdsdl/Semantics/Model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvmdsdl
{

/// @brief Resolves and caches transitive DSDL input dependencies for depfile emission.
///
/// @details
/// The planner indexes one semantic module once, then memoizes:
/// 1) per-type transitive input paths; and
/// 2) per-required-type-set merged input paths.
///
/// Returned dependency vectors are normalized absolute paths, sorted, and de-duplicated.
/// Synthetic embedded-uavcan paths are excluded from emitted dependencies.
class DepfilePlanner final
{
public:
    /// @brief Build planner index from semantic definitions.
    explicit DepfilePlanner(const SemanticModule& semantic);

    /// @brief Returns transitive input dependencies for one type key.
    ///
    /// @param[in] typeKey Canonical type key (`full_name:major:minor`).
    /// @return Cached sorted+deduped dependency list.
    const std::vector<std::string>& depsForTypeKey(const std::string& typeKey);

    /// @brief Returns merged dependencies for required type keys.
    ///
    /// @param[in] requiredTypeKeys Required output type keys (order-insensitive).
    /// @return Cached sorted+deduped merged dependency list.
    const std::vector<std::string>& depsForRequiredTypeKeys(const std::vector<std::string>& requiredTypeKeys);

private:
    struct Node final
    {
        std::vector<std::size_t> dependencyIndexes;
        std::string              normalizedInputPath;
        bool                     emitAsInput{false};
    };

    std::vector<Node>                                   nodes_;
    std::unordered_map<std::string, std::size_t>        nodeIndexByTypeKey_;
    std::unordered_map<std::string, std::vector<std::string>> depsByTypeKey_;
    std::unordered_map<std::string, std::vector<std::string>> depsByRequiredSetSignature_;
    std::vector<std::uint32_t>                          visitEpochByNode_;
    std::uint32_t                                       nextVisitEpoch_{1U};
    std::vector<std::string>                            emptyDeps_;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_FRONTEND_DEPFILEPLANNER_H
