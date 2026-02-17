#include "llvmdsdl/CodeGen/TypeStorage.h"

namespace llvmdsdl {

std::uint32_t scalarStorageBits(const std::uint32_t bitLength) {
  if (bitLength <= 8) {
    return 8;
  }
  if (bitLength <= 16) {
    return 16;
  }
  if (bitLength <= 32) {
    return 32;
  }
  return 64;
}

bool isVariableArray(const ArrayKind kind) {
  return kind == ArrayKind::VariableInclusive ||
         kind == ArrayKind::VariableExclusive;
}

SemanticFieldType arrayElementType(const SemanticFieldType &type) {
  auto out = type;
  out.arrayKind = ArrayKind::None;
  out.arrayCapacity = 0;
  out.arrayLengthPrefixBits = 0;
  return out;
}

} // namespace llvmdsdl
