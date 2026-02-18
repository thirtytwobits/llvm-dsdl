#include "llvmdsdl/CodeGen/LoweredBodyPlan.h"

namespace llvmdsdl {

LoweredBodyPlan buildLoweredBodyPlan(
    const SemanticSection &section, const LoweredSectionFacts *sectionFacts,
    const HelperBindingDirection direction) {
  LoweredBodyPlan out;
  out.statements = buildSectionStatementPlan(section, sectionFacts);
  out.helperBindings =
      buildSectionHelperBindingPlan(section, sectionFacts, direction);
  return out;
}

} // namespace llvmdsdl
