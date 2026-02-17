#ifndef LLVMDSDL_CODEGEN_SECTION_HELPER_BINDING_PLAN_H
#define LLVMDSDL_CODEGEN_SECTION_HELPER_BINDING_PLAN_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/SerDesHelperDescriptors.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvmdsdl {

enum class HelperBindingDirection {
  Serialize,
  Deserialize,
};

struct UnionTagMaskBindingDescriptor final {
  std::string symbol;
  std::uint32_t bits{0};
};

struct ScalarBindingDescriptor final {
  std::string symbol;
  ScalarHelperDescriptor descriptor;
};

struct ArrayPrefixBindingDescriptor final {
  std::string symbol;
  std::uint32_t bits{0};
};

struct ArrayValidateBindingDescriptor final {
  std::string symbol;
  std::int64_t capacity{0};
};

struct DelimiterValidateBindingDescriptor final {
  std::string symbol;
};

struct SectionHelperBindingPlan final {
  std::optional<CapacityCheckHelperDescriptor> capacityCheck;
  std::optional<UnionTagValidateHelperDescriptor> unionTagValidate;
  std::optional<UnionTagMaskBindingDescriptor> unionTagMask;
  std::vector<ScalarBindingDescriptor> scalarBindings;
  std::vector<ArrayPrefixBindingDescriptor> arrayPrefixBindings;
  std::vector<ArrayValidateBindingDescriptor> arrayValidateBindings;
  std::vector<DelimiterValidateBindingDescriptor> delimiterValidateBindings;
};

SectionHelperBindingPlan
buildSectionHelperBindingPlan(const SemanticSection &section,
                              const LoweredSectionFacts *sectionFacts,
                              HelperBindingDirection direction);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_SECTION_HELPER_BINDING_PLAN_H
