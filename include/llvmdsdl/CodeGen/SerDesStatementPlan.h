#ifndef LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H
#define LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace llvmdsdl {

struct PlannedFieldStep final {
  const SemanticField *field{nullptr};
  std::optional<std::uint32_t> arrayLengthPrefixBits;
  const LoweredFieldFacts *fieldFacts{nullptr};
};

struct SectionStatementPlan final {
  std::vector<PlannedFieldStep> orderedFields;
  std::vector<PlannedFieldStep> unionBranches;
};

SectionStatementPlan
buildSectionStatementPlan(const SemanticSection &section,
                          const LoweredSectionFacts *sectionFacts);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_SERDES_STATEMENT_PLAN_H
