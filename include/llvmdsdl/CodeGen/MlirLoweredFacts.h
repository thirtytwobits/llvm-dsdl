//===----------------------------------------------------------------------===//
///
/// @file
/// Data structures and collection APIs for extracting lowered facts from MLIR modules.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H
#define LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace llvmdsdl
{

/// @file
/// @brief Extracted lowered-serdes metadata consumed by backend emitters.

/// @brief Lowered helper and ordering metadata for a single field.
struct LoweredFieldFacts final
{
    /// @brief Field execution index in lowered order.
    std::optional<std::int64_t> stepIndex;

    /// @brief Optional variable-array prefix width override.
    std::optional<std::uint32_t> arrayLengthPrefixBits;

    /// @brief Serialize array-length prefix helper symbol.
    std::string serArrayLengthPrefixHelper;

    /// @brief Deserialize array-length prefix helper symbol.
    std::string deserArrayLengthPrefixHelper;

    /// @brief Array-length validation helper symbol.
    std::string arrayLengthValidateHelper;

    /// @brief Delimiter validation helper symbol.
    std::string delimiterValidateHelper;

    /// @brief Serialize unsigned helper symbol.
    std::string serUnsignedHelper;

    /// @brief Deserialize unsigned helper symbol.
    std::string deserUnsignedHelper;

    /// @brief Serialize signed helper symbol.
    std::string serSignedHelper;

    /// @brief Deserialize signed helper symbol.
    std::string deserSignedHelper;

    /// @brief Serialize float helper symbol.
    std::string serFloatHelper;

    /// @brief Deserialize float helper symbol.
    std::string deserFloatHelper;
};

/// @brief Lowered metadata shared across all fields in one section.
struct LoweredSectionFacts final
{
    /// @brief Section capacity-check helper symbol.
    std::string capacityCheckHelper;

    /// @brief Optional union tag width.
    std::optional<std::uint32_t> unionTagBits;

    /// @brief Union-tag validation helper symbol.
    std::string unionTagValidateHelper;

    /// @brief Serialize union-tag helper symbol.
    std::string serUnionTagHelper;

    /// @brief Deserialize union-tag helper symbol.
    std::string deserUnionTagHelper;

    /// @brief Field facts keyed by field name.
    std::unordered_map<std::string, LoweredFieldFacts> fieldsByName;
};

/// @brief Per-definition lowered facts keyed by section name.
using LoweredDefinitionFacts = std::unordered_map<std::string, LoweredSectionFacts>;

/// @brief Top-level lowered facts keyed by canonical type key.
using LoweredFactsMap = std::unordered_map<std::string, LoweredDefinitionFacts>;

/// @brief Builds canonical lowered-facts key for a versioned type.
/// @param[in] name Fully qualified type name.
/// @param[in] major Major version.
/// @param[in] minor Minor version.
/// @return Canonical key string.
std::string loweredTypeKey(const std::string& name, std::uint32_t major, std::uint32_t minor);

/// @brief Collects lowered metadata from MLIR for all semantic definitions.
/// @param[in] semantic Semantic module used for validation/indexing.
/// @param[in] module MLIR module containing lowered serialization plans.
/// @param[in,out] diagnostics Diagnostic sink.
/// @param[in] backendLabel Backend label for diagnostics.
/// @param[out] outFacts Output lowered-facts map.
/// @param[in] optimizeLoweredSerDes Enables optional optimization pipeline.
/// @return True on success, false on extraction/validation failure.
bool collectLoweredFactsFromMlir(const SemanticModule& semantic,
                                 mlir::ModuleOp        module,
                                 DiagnosticEngine&     diagnostics,
                                 const std::string&    backendLabel,
                                 LoweredFactsMap*      outFacts,
                                 bool                  optimizeLoweredSerDes = false);

/// @brief Finds lowered field facts by field name.
/// @param[in] sectionFacts Section facts map.
/// @param[in] fieldName Field identifier.
/// @return Matching field facts or `nullptr` when absent.
const LoweredFieldFacts* findLoweredFieldFacts(const LoweredSectionFacts* sectionFacts, const std::string& fieldName);

/// @brief Resolves effective lowered array-prefix width for a field.
/// @param[in] sectionFacts Section facts map.
/// @param[in] fieldName Field identifier.
/// @return Prefix width when available.
std::optional<std::uint32_t> loweredFieldArrayPrefixBits(const LoweredSectionFacts* sectionFacts,
                                                         const std::string&         fieldName);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H
