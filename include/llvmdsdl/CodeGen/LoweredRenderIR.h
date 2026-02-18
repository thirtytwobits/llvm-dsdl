#ifndef LLVMDSDL_CODEGEN_LOWERED_RENDER_IR_H
#define LLVMDSDL_CODEGEN_LOWERED_RENDER_IR_H

#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/SerDesStatementPlan.h"

#include <vector>

namespace llvmdsdl {

enum class LoweredRenderStepKind {
  Field,
  Padding,
  UnionDispatch,
};

struct LoweredRenderStep final {
  LoweredRenderStepKind kind{LoweredRenderStepKind::Field};
  PlannedFieldStep fieldStep;
  std::vector<PlannedFieldStep> unionBranches;
};

struct LoweredBodyRenderIR final {
  std::vector<LoweredRenderStep> steps;
  SectionHelperBindingPlan helperBindings;
};

LoweredBodyRenderIR buildLoweredBodyRenderIR(
    const SemanticSection &section, const LoweredSectionFacts *sectionFacts,
    HelperBindingDirection direction);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_LOWERED_RENDER_IR_H
