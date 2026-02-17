#ifndef LLVMDSDL_CODEGEN_TYPE_STORAGE_H
#define LLVMDSDL_CODEGEN_TYPE_STORAGE_H

#include "llvmdsdl/Semantics/Model.h"

#include <cstdint>

namespace llvmdsdl {

std::uint32_t scalarStorageBits(std::uint32_t bitLength);
bool isVariableArray(ArrayKind kind);
SemanticFieldType arrayElementType(const SemanticFieldType &type);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_TYPE_STORAGE_H
