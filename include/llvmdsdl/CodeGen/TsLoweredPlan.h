//===----------------------------------------------------------------------===//
///
/// @file
/// TypeScript-specific lowered planning declarations derived from shared lowered metadata.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H
#define LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Model.h"

namespace llvmdsdl
{
struct LoweredSectionFacts;

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

/// @brief TypeScript runtime field category.
enum class TsRuntimeFieldKind
{
    /// @brief Explicit padding field.
    Padding,

    /// @brief Boolean scalar.
    Bool,

    /// @brief Unsigned/byte/utf8 scalar.
    Unsigned,

    /// @brief Signed scalar.
    Signed,

    /// @brief Floating-point scalar.
    Float,

    /// @brief Nested composite field.
    Composite,
};

/// @brief TypeScript runtime array category.
enum class TsRuntimeArrayKind
{
    /// @brief Scalar/non-array field.
    None,

    /// @brief Fixed-length array field.
    Fixed,

    /// @brief Variable-length array field.
    Variable,
};

/// @brief TypeScript runtime plan entry for one field.
struct TsRuntimeFieldPlan final
{
    /// @brief Sanitized runtime field identifier.
    std::string fieldName;

    /// @brief Runtime field kind.
    TsRuntimeFieldKind kind{TsRuntimeFieldKind::Unsigned};

    /// @brief Cast behavior for numeric write paths.
    CastMode castMode{CastMode::Saturated};

    /// @brief Scalar/composite bit length for one element.
    std::int64_t bitLength{0};

    /// @brief Alignment in bits.
    std::int64_t alignmentBits{1};

    /// @brief True when bigint runtime APIs are required.
    bool useBigInt{false};

    /// @brief Optional composite type reference.
    std::optional<SemanticTypeRef> compositeType;

    /// @brief Composite sealed flag.
    bool compositeSealed{true};

    /// @brief Maximum payload bits for delimited composites.
    std::int64_t compositePayloadMaxBits{0};

    /// @brief Union option index for union sections.
    std::uint32_t unionOptionIndex{0};

    /// @brief Array category.
    TsRuntimeArrayKind arrayKind{TsRuntimeArrayKind::None};

    /// @brief Array capacity for fixed/variable arrays.
    std::int64_t arrayCapacity{0};

    /// @brief Prefix width for variable arrays.
    std::int64_t arrayLengthPrefixBits{0};
};

/// @brief TypeScript runtime plan for one semantic section.
struct TsRuntimeSectionPlan final
{
    /// @brief True for union sections.
    bool isUnion{false};

    /// @brief Union tag bit width for union sections.
    std::int64_t unionTagBits{0};

    /// @brief Ordered runtime field steps.
    std::vector<TsRuntimeFieldPlan> fields;

    /// @brief Maximum serialized bit length used for allocation.
    std::int64_t maxBits{0};
};

/// @brief Builds TypeScript runtime section plan from lowered facts.
/// @param[in] section Semantic section to plan.
/// @param[in] sectionFacts Lowered section facts for the same section.
/// @return Runtime section plan or a contract-validation error.
llvm::Expected<TsRuntimeSectionPlan> buildTsRuntimeSectionPlan(const SemanticSection&     section,
                                                               const LoweredSectionFacts* sectionFacts);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_TS_LOWERED_PLAN_H
