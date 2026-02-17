#include "llvmdsdl/CodeGen/HelperSymbolResolver.h"

#include <iostream>

bool runHelperSymbolResolverTests() {
  {
    llvmdsdl::LoweredSectionFacts facts;
    facts.capacityCheckHelper = "cap";
    facts.unionTagValidateHelper = "tag_validate";
    facts.serUnionTagHelper = "tag_ser";
    facts.deserUnionTagHelper = "tag_des";

    if (llvmdsdl::resolveSectionCapacityCheckHelperSymbol(&facts) != "cap" ||
        llvmdsdl::resolveSectionUnionTagValidateHelperSymbol(&facts) !=
            "tag_validate" ||
        llvmdsdl::resolveSectionUnionTagMaskHelperSymbol(
            &facts, llvmdsdl::HelperBindingDirection::Serialize) != "tag_ser" ||
        llvmdsdl::resolveSectionUnionTagMaskHelperSymbol(
            &facts, llvmdsdl::HelperBindingDirection::Deserialize) != "tag_des") {
      std::cerr << "section helper symbol resolution mismatch\n";
      return false;
    }
  }

  {
    llvmdsdl::SemanticFieldType type;
    type.scalarCategory = llvmdsdl::SemanticScalarCategory::SignedInt;
    type.bitLength = 5;
    type.castMode = llvmdsdl::CastMode::Saturated;

    llvmdsdl::LoweredFieldFacts facts;
    facts.serSignedHelper = "signed_ser";
    facts.deserSignedHelper = "signed_des";

    if (llvmdsdl::resolveScalarHelperSymbol(
            type, &facts, llvmdsdl::HelperBindingDirection::Serialize) !=
            "signed_ser" ||
        llvmdsdl::resolveScalarHelperSymbol(
            type, &facts, llvmdsdl::HelperBindingDirection::Deserialize) !=
            "signed_des") {
      std::cerr << "scalar helper symbol resolution mismatch\n";
      return false;
    }
  }

  {
    llvmdsdl::SemanticFieldType type;
    type.arrayKind = llvmdsdl::ArrayKind::VariableInclusive;
    type.arrayCapacity = 32;
    type.arrayLengthPrefixBits = 8;

    llvmdsdl::LoweredFieldFacts facts;
    facts.serArrayLengthPrefixHelper = "prefix_ser";
    facts.deserArrayLengthPrefixHelper = "prefix_des";
    facts.arrayLengthValidateHelper = "len_validate";

    const auto ser = llvmdsdl::resolveArrayLengthHelperDescriptor(
        type, &facts, 8U, llvmdsdl::HelperBindingDirection::Serialize);
    const auto des = llvmdsdl::resolveArrayLengthHelperDescriptor(
        type, &facts, 8U, llvmdsdl::HelperBindingDirection::Deserialize);
    if (!ser || !des || ser->prefixSymbol != "prefix_ser" ||
        des->prefixSymbol != "prefix_des" ||
        ser->validateSymbol != "len_validate" || des->capacity != 32) {
      std::cerr << "array helper descriptor resolution mismatch\n";
      return false;
    }
  }

  {
    llvmdsdl::SemanticFieldType type;
    type.scalarCategory = llvmdsdl::SemanticScalarCategory::Composite;
    type.compositeSealed = false;
    llvmdsdl::LoweredFieldFacts facts;
    facts.delimiterValidateHelper = "delim_validate";
    if (llvmdsdl::resolveDelimiterValidateHelperSymbol(type, &facts) !=
        "delim_validate") {
      std::cerr << "delimiter helper symbol resolution mismatch\n";
      return false;
    }
  }

  return true;
}
