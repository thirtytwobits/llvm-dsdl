//===----------------------------------------------------------------------===//
///
/// @file
/// Statement planning primitives for deterministic section body emission.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H
#define LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace llvmdsdl
{

/// @file
/// @brief Deterministic lowered statement ordering for section body emission.

/// @brief Planned field statement with lowered metadata attachments.
struct PlannedFieldStep final
{
    /// @brief Field being emitted.
    const SemanticField* field{nullptr};

    /// @brief Optional lowered array-prefix width override.
    std::optional<std::uint32_t> arrayLengthPrefixBits;

    /// @brief Optional lowered helper metadata for the field.
    const LoweredFieldFacts* fieldFacts{nullptr};
};

/// @brief Ordered statement plan for one section.
struct SectionStatementPlan final
{
    /// @brief Ordered fields for struct-like sections.
    std::vector<PlannedFieldStep> orderedFields;

    /// @brief Ordered branches for union sections.
    std::vector<PlannedFieldStep> unionBranches;
};

/// @brief Builds statement ordering from semantic and lowered facts.
/// @param[in] section Semantic section.
/// @param[in] sectionFacts Lowered section facts.
/// @return Section statement plan.
SectionStatementPlan buildSectionStatementPlan(const SemanticSection& section, const LoweredSectionFacts* sectionFacts);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H
