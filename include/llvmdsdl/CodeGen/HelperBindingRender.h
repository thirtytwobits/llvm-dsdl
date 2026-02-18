#ifndef LLVMDSDL_CODEGEN_HELPER_BINDING_RENDER_H
#define LLVMDSDL_CODEGEN_HELPER_BINDING_RENDER_H

#include "llvmdsdl/CodeGen/SerDesHelperDescriptors.h"
#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llvmdsdl {

enum class HelperBindingRenderLanguage {
  Cpp,
  Rust,
  Go,
};

enum class ScalarBindingRenderDirection {
  Serialize,
  Deserialize,
};

std::vector<std::string> renderCapacityCheckBinding(
    HelperBindingRenderLanguage language, const std::string &helperName,
    std::int64_t requiredBits);

std::vector<std::string> renderUnionTagMaskBinding(
    HelperBindingRenderLanguage language, const std::string &helperName,
    std::uint32_t bits);

std::vector<std::string> renderUnionTagValidateBinding(
    HelperBindingRenderLanguage language, const std::string &helperName,
    const std::vector<std::int64_t> &allowedTags);

std::vector<std::string> renderDelimiterValidateBinding(
    HelperBindingRenderLanguage language, const std::string &helperName);

std::vector<std::string> renderScalarBinding(
    HelperBindingRenderLanguage language, ScalarBindingRenderDirection direction,
    const std::string &helperName, const ScalarHelperDescriptor &descriptor);

std::vector<std::string> renderArrayPrefixBinding(
    HelperBindingRenderLanguage language, const std::string &helperName,
    std::uint32_t bits);

std::vector<std::string> renderArrayValidateBinding(
    HelperBindingRenderLanguage language, const std::string &helperName,
    std::int64_t capacity);

std::vector<std::string> renderSectionHelperBindings(
    const SectionHelperBindingPlan &plan, HelperBindingRenderLanguage language,
    ScalarBindingRenderDirection scalarDirection,
    const std::function<std::string(const std::string &)> &helperNameResolver,
    bool emitCapacityCheck);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_HELPER_BINDING_RENDER_H
