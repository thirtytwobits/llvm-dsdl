#ifndef LLVMDSDL_CODEGEN_LOWERED_BODY_PLAN_H
#define LLVMDSDL_CODEGEN_LOWERED_BODY_PLAN_H

#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/SerDesStatementPlan.h"

namespace llvmdsdl {

struct LoweredBodyPlan final {
  SectionStatementPlan statements;
  SectionHelperBindingPlan helperBindings;
};

LoweredBodyPlan buildLoweredBodyPlan(
    const SemanticSection &section, const LoweredSectionFacts *sectionFacts,
    HelperBindingDirection direction);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_LOWERED_BODY_PLAN_H
