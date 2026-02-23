//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared control-flow planning metadata for scripted runtime backends.
///
/// This plan precomputes section and field helper bindings from lowered facts
/// so scripted emitters can focus on syntax rendering and runtime call mapping.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_SCRIPTED_BODY_PLAN_H
#define LLVMDSDL_CODEGEN_SCRIPTED_BODY_PLAN_H

#include <optional>
#include <vector>

#include "llvmdsdl/CodeGen/RuntimeHelperBindings.h"
#include "llvmdsdl/CodeGen/RuntimeLoweredPlan.h"

namespace llvmdsdl
{
struct LoweredSectionFacts;
struct SemanticSection;

/// @brief Planned scripted-backend metadata for one runtime field.
struct ScriptedFieldBodyPlan final
{
    /// @brief Runtime field plan entry.
    RuntimeFieldPlan field;

    /// @brief Optional variable-array prefix override.
    std::optional<std::uint32_t> arrayPrefixOverride;

    /// @brief Lowered helper names for this field.
    RuntimeFieldHelperNames helpers;
};

/// @brief Planned scripted-backend metadata for one runtime section.
struct ScriptedSectionBodyPlan final
{
    /// @brief Section helper names.
    RuntimeSectionHelperNames sectionHelpers;

    /// @brief Ordered per-field body plans.
    std::vector<ScriptedFieldBodyPlan> fields;
};

/// @brief Builds scripted runtime body plan from runtime section and lowered facts.
/// @param[in] section Semantic section for field lookups.
/// @param[in] runtimePlan Runtime section plan in lowered order.
/// @param[in] sectionFacts Optional lowered helper facts.
/// @param[in] helperNameResolver Lowered symbol-to-emitted-name callback.
/// @return Scripted section body plan.
ScriptedSectionBodyPlan buildScriptedSectionBodyPlan(const SemanticSection&               section,
                                                     const RuntimeSectionPlan&            runtimePlan,
                                                     const LoweredSectionFacts*           sectionFacts,
                                                     const RuntimeHelperNameResolver& helperNameResolver);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_SCRIPTED_BODY_PLAN_H
