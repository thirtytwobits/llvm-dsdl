//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Builds shared scripted operation plans from helper-bound body metadata.
///
/// This component classifies each field into operation categories consumed by
/// TS/Python emitters so orchestration decisions remain centralized.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/ScriptedOperationPlan.h"

#include <utility>

namespace llvmdsdl
{
namespace
{

ScriptedFieldCardinality classifyFieldCardinality(const RuntimeArrayKind arrayKind)
{
    switch (arrayKind)
    {
    case RuntimeArrayKind::None:
        return ScriptedFieldCardinality::Scalar;
    case RuntimeArrayKind::Fixed:
        return ScriptedFieldCardinality::FixedArray;
    case RuntimeArrayKind::Variable:
        return ScriptedFieldCardinality::VariableArray;
    }
    return ScriptedFieldCardinality::Scalar;
}

ScriptedFieldValueKind classifyFieldValueKind(const RuntimeFieldKind kind)
{
    switch (kind)
    {
    case RuntimeFieldKind::Padding:
        return ScriptedFieldValueKind::Padding;
    case RuntimeFieldKind::Bool:
        return ScriptedFieldValueKind::Bool;
    case RuntimeFieldKind::Unsigned:
        return ScriptedFieldValueKind::Unsigned;
    case RuntimeFieldKind::Signed:
        return ScriptedFieldValueKind::Signed;
    case RuntimeFieldKind::Float:
        return ScriptedFieldValueKind::Float;
    case RuntimeFieldKind::Composite:
        return ScriptedFieldValueKind::Composite;
    }
    return ScriptedFieldValueKind::Padding;
}

}  // namespace

ScriptedSectionOperationPlan buildScriptedSectionOperationPlan(const SemanticSection&           section,
                                                               const RuntimeSectionPlan&        runtimePlan,
                                                               const LoweredSectionFacts*       sectionFacts,
                                                               const RuntimeHelperNameResolver& helperNameResolver)
{
    const auto bodyPlan = buildScriptedSectionBodyPlan(section, runtimePlan, sectionFacts, helperNameResolver);

    ScriptedSectionOperationPlan out;
    out.isUnion        = runtimePlan.isUnion;
    out.unionTagBits   = runtimePlan.unionTagBits;
    out.maxBits        = runtimePlan.maxBits;
    out.sectionHelpers = bodyPlan.sectionHelpers;
    out.fields.reserve(bodyPlan.fields.size());
    for (const auto& bodyField : bodyPlan.fields)
    {
        ScriptedFieldOperationPlan operation;
        operation.body        = bodyField;
        operation.cardinality = classifyFieldCardinality(bodyField.field.arrayKind);
        operation.valueKind   = classifyFieldValueKind(bodyField.field.kind);
        out.fields.push_back(std::move(operation));
    }
    return out;
}

}  // namespace llvmdsdl
