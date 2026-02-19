//===----------------------------------------------------------------------===//
///
/// @file
/// Builds TypeScript-specific lowering plans from render IR.
///
/// The planning utilities convert generic render steps into TypeScript execution primitives used by the TS emitter.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/TsLoweredPlan.h"

#include "llvmdsdl/CodeGen/LoweredRenderIR.h"

#include "llvm/Support/Error.h"

#include <set>

namespace llvmdsdl
{
namespace
{

std::size_t expectedStepCount(const SemanticSection& section)
{
    if (!section.isUnion)
    {
        return section.fields.size();
    }
    std::size_t count = 0;
    for (const auto& field : section.fields)
    {
        if (!field.isPadding)
        {
            ++count;
        }
    }
    return count;
}

}  // namespace

llvm::Expected<std::vector<TsOrderedFieldStep>> buildTsOrderedFieldSteps(const SemanticSection&           section,
                                                                         const LoweredSectionFacts* const sectionFacts)
{
    if (sectionFacts == nullptr)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "missing lowered section facts required for TypeScript runtime planning");
    }

    const auto renderIR = buildLoweredBodyRenderIR(section, sectionFacts, HelperBindingDirection::Serialize);
    std::vector<TsOrderedFieldStep> out;

    if (section.isUnion)
    {
        const auto* unionStep = renderIR.steps.empty() ? nullptr : &renderIR.steps.front();
        if (unionStep == nullptr || unionStep->kind != LoweredRenderStepKind::UnionDispatch)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered render IR missing union-dispatch step");
        }
        if (renderIR.steps.size() != 1U)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered union render IR unexpectedly contains non-dispatch steps");
        }
        out.reserve(unionStep->unionBranches.size());
        for (const auto& branch : unionStep->unionBranches)
        {
            if (branch.field == nullptr)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered union render IR contains null field branch");
            }
            const auto* const facts = findLoweredFieldFacts(sectionFacts, branch.field->name);
            if (facts == nullptr || !facts->stepIndex)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "missing lowered step index for union field '%s'",
                                               branch.field->name.c_str());
            }
            out.push_back(TsOrderedFieldStep{branch.field, branch.arrayLengthPrefixBits});
        }
    }
    else
    {
        out.reserve(renderIR.steps.size());
        for (const auto& step : renderIR.steps)
        {
            if (step.kind != LoweredRenderStepKind::Field && step.kind != LoweredRenderStepKind::Padding)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered struct render IR contains unsupported step kind");
            }
            if (step.fieldStep.field == nullptr)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered struct render IR contains null field step");
            }
            const auto* const facts = findLoweredFieldFacts(sectionFacts, step.fieldStep.field->name);
            if (facts == nullptr || !facts->stepIndex)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "missing lowered step index for field '%s'",
                                               step.fieldStep.field->name.c_str());
            }
            out.push_back(TsOrderedFieldStep{step.fieldStep.field, step.fieldStep.arrayLengthPrefixBits});
        }
    }

    if (out.size() != expectedStepCount(section))
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "lowered render IR step count does not match semantic field count");
    }
    std::set<const SemanticField*> uniqueness;
    for (const auto& step : out)
    {
        if (!uniqueness.insert(step.field).second)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered render IR contains duplicate field references");
        }
    }

    return out;
}

}  // namespace llvmdsdl
