//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared semantic-definition lookup index for code generation backends.
///
/// The index provides deterministic lookup by lowered type key so emitters can
/// avoid re-implementing local lookup maps.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_DEFINITION_INDEX_H
#define LLVMDSDL_CODEGEN_DEFINITION_INDEX_H

#include <string>
#include <unordered_map>

#include "llvmdsdl/Semantics/Model.h"

namespace llvmdsdl
{

/// @brief Lookup index for semantic definitions.
class DefinitionIndex final
{
public:
    /// @brief Builds an index for a semantic module.
    /// @param[in] semantic Semantic module to index.
    explicit DefinitionIndex(const SemanticModule& semantic);

    /// @brief Finds a semantic definition by type reference.
    /// @param[in] ref Type reference.
    /// @return Matching definition, or `nullptr` when missing.
    const SemanticDefinition* find(const SemanticTypeRef& ref) const;

private:
    std::unordered_map<std::string, const SemanticDefinition*> byKey_;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_DEFINITION_INDEX_H
