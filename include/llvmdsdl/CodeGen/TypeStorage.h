//===----------------------------------------------------------------------===//
///
/// @file
/// Utility declarations for scalar storage widths and saturation range computation.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_TYPE_STORAGE_H
#define LLVMDSDL_CODEGEN_TYPE_STORAGE_H

#include <cstdint>
#include <optional>
#include <utility>

#include "llvmdsdl/Semantics/Model.h"

namespace llvmdsdl
{
enum class ArrayKind;

/// @file
/// @brief Scalar storage and saturation helper utilities for codegen.

/// @brief Returns storage width used by runtime helpers for a scalar width.
/// @param[in] bitLength Scalar bit width.
/// @return Storage width in bits.
std::uint32_t scalarStorageBits(std::uint32_t bitLength);

/// @brief Returns width suffix used in helper symbol names.
/// @param[in] bitLength Scalar bit width.
/// @return Symbol suffix string.
const char* scalarWidthSuffix(std::uint32_t bitLength);

/// @brief Returns true when the array kind is variable-length.
/// @param[in] kind Array kind.
/// @return True for variable-inclusive and variable-exclusive arrays.
bool isVariableArray(ArrayKind kind);

/// @brief Returns a scalar element view of an array field type.
/// @param[in] type Array field type.
/// @return Scalar element type.
SemanticFieldType arrayElementType(const SemanticFieldType& type);

/// @brief Computes unsigned saturation max for a bit width.
/// @param[in] bitLength Bit width.
/// @return Maximum saturating value when representable.
std::optional<std::uint64_t> resolveUnsignedSaturationMax(std::uint32_t bitLength);

/// @brief Computes signed saturation range for a bit width.
/// @param[in] bitLength Bit width.
/// @return Inclusive min/max pair when representable.
std::optional<std::pair<std::int64_t, std::int64_t>> resolveSignedSaturationRange(std::uint32_t bitLength);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_TYPE_STORAGE_H
