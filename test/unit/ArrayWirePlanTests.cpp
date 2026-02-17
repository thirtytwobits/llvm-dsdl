#include "llvmdsdl/CodeGen/ArrayWirePlan.h"

#include <iostream>

bool runArrayWirePlanTests() {
  {
    llvmdsdl::SemanticFieldType type;
    type.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
    type.bitLength = 8;
    type.arrayKind = llvmdsdl::ArrayKind::VariableInclusive;
    type.arrayCapacity = 32;
    type.arrayLengthPrefixBits = 8;

    llvmdsdl::LoweredFieldFacts facts;
    facts.serArrayLengthPrefixHelper = "prefix_ser";
    facts.deserArrayLengthPrefixHelper = "prefix_des";
    facts.arrayLengthValidateHelper = "validate_len";

    const auto ser = llvmdsdl::buildArrayWirePlan(
        type, &facts, 16U, llvmdsdl::HelperBindingDirection::Serialize);
    const auto des = llvmdsdl::buildArrayWirePlan(
        type, &facts, std::nullopt, llvmdsdl::HelperBindingDirection::Deserialize);

    if (!ser.variable || ser.prefixBits != 16 || !ser.descriptor ||
        ser.descriptor->prefixSymbol != "prefix_ser" ||
        ser.descriptor->validateSymbol != "validate_len" ||
        ser.descriptor->prefixBits != 16 || ser.descriptor->capacity != 32) {
      std::cerr << "serialize array wire plan mismatch\n";
      return false;
    }
    if (!des.variable || des.prefixBits != 8 || !des.descriptor ||
        des.descriptor->prefixSymbol != "prefix_des" ||
        des.descriptor->validateSymbol != "validate_len" ||
        des.descriptor->prefixBits != 8 || des.descriptor->capacity != 32) {
      std::cerr << "deserialize array wire plan mismatch\n";
      return false;
    }
  }

  {
    llvmdsdl::SemanticFieldType type;
    type.scalarCategory = llvmdsdl::SemanticScalarCategory::SignedInt;
    type.bitLength = 7;
    type.arrayKind = llvmdsdl::ArrayKind::Fixed;
    type.arrayCapacity = 4;
    type.arrayLengthPrefixBits = 0;

    const auto plan = llvmdsdl::buildArrayWirePlan(
        type, nullptr, std::nullopt, llvmdsdl::HelperBindingDirection::Serialize);
    if (plan.variable || plan.prefixBits != 0 || plan.descriptor) {
      std::cerr << "fixed array wire plan mismatch\n";
      return false;
    }
  }

  return true;
}
