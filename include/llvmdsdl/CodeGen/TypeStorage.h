#ifndef LLVMDSDL_CODEGEN_TYPE_STORAGE_H
#define LLVMDSDL_CODEGEN_TYPE_STORAGE_H

#include "llvmdsdl/Semantics/Model.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace llvmdsdl {

std::uint32_t scalarStorageBits(std::uint32_t bitLength);
const char *scalarWidthSuffix(std::uint32_t bitLength);
bool isVariableArray(ArrayKind kind);
SemanticFieldType arrayElementType(const SemanticFieldType &type);
std::optional<std::uint64_t>
resolveUnsignedSaturationMax(std::uint32_t bitLength);
std::optional<std::pair<std::int64_t, std::int64_t>>
resolveSignedSaturationRange(std::uint32_t bitLength);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_TYPE_STORAGE_H
