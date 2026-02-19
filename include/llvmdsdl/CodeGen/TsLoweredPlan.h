//===----------------------------------------------------------------------===//
///
/// @file
/// TypeScript-specific lowered planning declarations derived from shared lowered metadata.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H
#define LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace llvmdsdl
{

/// @file
/// @brief TypeScript ordered-field plans derived from lowered facts.

/// @brief Planned TypeScript field step in lowered execution order.
struct TsOrderedFieldStep final
{
    /// @brief Semantic field reference.
    const SemanticField* field{nullptr};

    /// @brief Optional lowered array-prefix width override.
    std::optional<std::uint32_t> arrayLengthPrefixBits;
};

/// @brief Builds deterministic TypeScript field ordering from lowered facts.
/// @param[in] section Semantic section to plan.
/// @param[in] sectionFacts Lowered section facts for the same section.
/// @return Ordered field steps or a contract-validation error.
llvm::Expected<std::vector<TsOrderedFieldStep>> buildTsOrderedFieldSteps(const SemanticSection&     section,
                                                                         const LoweredSectionFacts* sectionFacts);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H
