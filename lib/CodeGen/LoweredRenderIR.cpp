#include "llvmdsdl/CodeGen/LoweredRenderIR.h"

#include <utility>

namespace llvmdsdl {

LoweredBodyRenderIR buildLoweredBodyRenderIR(
    const SemanticSection &section, const LoweredSectionFacts *sectionFacts,
    const HelperBindingDirection direction) {
  LoweredBodyRenderIR out;
  const auto statementPlan = buildSectionStatementPlan(section, sectionFacts);
  out.helperBindings =
      buildSectionHelperBindingPlan(section, sectionFacts, direction);

  if (section.isUnion) {
    LoweredRenderStep unionStep;
    unionStep.kind = LoweredRenderStepKind::UnionDispatch;
    unionStep.unionBranches = statementPlan.unionBranches;
    out.steps.push_back(std::move(unionStep));
    return out;
  }

  out.steps.reserve(statementPlan.orderedFields.size());
  for (const auto &fieldStep : statementPlan.orderedFields) {
    LoweredRenderStep step;
    if (fieldStep.field && fieldStep.field->isPadding) {
      step.kind = LoweredRenderStepKind::Padding;
    } else {
      step.kind = LoweredRenderStepKind::Field;
    }
    step.fieldStep = fieldStep;
    out.steps.push_back(std::move(step));
  }
  return out;
}

} // namespace llvmdsdl
