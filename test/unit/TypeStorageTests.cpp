#include "llvmdsdl/CodeGen/TypeStorage.h"

#include <iostream>

bool runTypeStorageTests() {
  if (llvmdsdl::scalarStorageBits(1) != 8 ||
      llvmdsdl::scalarStorageBits(8) != 8 ||
      llvmdsdl::scalarStorageBits(9) != 16 ||
      llvmdsdl::scalarStorageBits(16) != 16 ||
      llvmdsdl::scalarStorageBits(17) != 32 ||
      llvmdsdl::scalarStorageBits(33) != 64) {
    std::cerr << "scalar storage bit-width resolution mismatch\n";
    return false;
  }

  if (llvmdsdl::isVariableArray(llvmdsdl::ArrayKind::None) ||
      llvmdsdl::isVariableArray(llvmdsdl::ArrayKind::Fixed) ||
      !llvmdsdl::isVariableArray(llvmdsdl::ArrayKind::VariableInclusive) ||
      !llvmdsdl::isVariableArray(llvmdsdl::ArrayKind::VariableExclusive)) {
    std::cerr << "array variability classification mismatch\n";
    return false;
  }

  {
    llvmdsdl::SemanticFieldType arrayType;
    arrayType.scalarCategory = llvmdsdl::SemanticScalarCategory::SignedInt;
    arrayType.bitLength = 7;
    arrayType.arrayKind = llvmdsdl::ArrayKind::VariableInclusive;
    arrayType.arrayCapacity = 12;
    arrayType.arrayLengthPrefixBits = 8;
    arrayType.castMode = llvmdsdl::CastMode::Truncated;

    const auto elementType = llvmdsdl::arrayElementType(arrayType);
    if (elementType.arrayKind != llvmdsdl::ArrayKind::None ||
        elementType.arrayCapacity != 0 ||
        elementType.arrayLengthPrefixBits != 0 ||
        elementType.scalarCategory != arrayType.scalarCategory ||
        elementType.bitLength != arrayType.bitLength ||
        elementType.castMode != arrayType.castMode) {
      std::cerr << "array element type normalization mismatch\n";
      return false;
    }
  }

  return true;
}
