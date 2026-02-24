//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared scripted-backend operation planning metadata.
///
/// This plan extends helper-binding body planning with explicit operation
/// categories for scalar/array and field-kind orchestration in TS/Python
/// emitters.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_SCRIPTED_OPERATION_PLAN_H
#define LLVMDSDL_CODEGEN_SCRIPTED_OPERATION_PLAN_H

#include <cstdint>
#include <vector>

#include "llvmdsdl/CodeGen/ScriptedBodyPlan.h"

namespace llvmdsdl
{

/// @brief Field cardinality category for scripted operation rendering.
enum class ScriptedFieldCardinality
{
    /// @brief Scalar or composite single-value operation.
    Scalar,

    /// @brief Fixed-length array operation.
    FixedArray,

    /// @brief Variable-length array operation.
    VariableArray,
};

/// @brief Field value category for scripted operation rendering.
enum class ScriptedFieldValueKind
{
    /// @brief Padding-only operation.
    Padding,

    /// @brief Boolean value operation.
    Bool,

    /// @brief Unsigned integer value operation.
    Unsigned,

    /// @brief Signed integer value operation.
    Signed,

    /// @brief Floating-point value operation.
    Float,

    /// @brief Composite value operation.
    Composite,
};

/// @brief One scripted field operation.
struct ScriptedFieldOperationPlan final
{
    /// @brief Helper-bound field body metadata.
    ScriptedFieldBodyPlan body;

    /// @brief Field cardinality.
    ScriptedFieldCardinality cardinality{ScriptedFieldCardinality::Scalar};

    /// @brief Field value kind.
    ScriptedFieldValueKind valueKind{ScriptedFieldValueKind::Padding};
};

/// @brief One scripted section operation plan.
struct ScriptedSectionOperationPlan final
{
    /// @brief Union flag.
    bool isUnion{false};

    /// @brief Union tag width in bits.
    std::uint32_t unionTagBits{0};

    /// @brief Maximum section bit length.
    std::int64_t maxBits{0};

    /// @brief Section-level helper names.
    RuntimeSectionHelperNames sectionHelpers;

    /// @brief Ordered field operations.
    std::vector<ScriptedFieldOperationPlan> fields;
};

/// @brief Builds scripted operation plan from runtime section and lowered facts.
/// @param[in] section Semantic section for helper lookup.
/// @param[in] runtimePlan Runtime section plan in lowered order.
/// @param[in] sectionFacts Optional lowered helper facts.
/// @param[in] helperNameResolver Lowered symbol-to-emitted-name callback.
/// @return Scripted section operation plan.
ScriptedSectionOperationPlan buildScriptedSectionOperationPlan(const SemanticSection&           section,
                                                               const RuntimeSectionPlan&        runtimePlan,
                                                               const LoweredSectionFacts*       sectionFacts,
                                                               const RuntimeHelperNameResolver& helperNameResolver);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_SCRIPTED_OPERATION_PLAN_H
