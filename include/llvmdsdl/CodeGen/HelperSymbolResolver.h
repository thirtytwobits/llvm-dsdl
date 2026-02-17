#ifndef LLVMDSDL_CODEGEN_HELPER_SYMBOL_RESOLVER_H
#define LLVMDSDL_CODEGEN_HELPER_SYMBOL_RESOLVER_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/SerDesHelperDescriptors.h"

#include <optional>
#include <string>

namespace llvmdsdl {

std::string resolveSectionCapacityCheckHelperSymbol(
    const LoweredSectionFacts *sectionFacts);

std::string resolveSectionUnionTagValidateHelperSymbol(
    const LoweredSectionFacts *sectionFacts);

std::string resolveSectionUnionTagMaskHelperSymbol(
    const LoweredSectionFacts *sectionFacts, HelperBindingDirection direction);

std::string resolveScalarHelperSymbol(const SemanticFieldType &type,
                                      const LoweredFieldFacts *fieldFacts,
                                      HelperBindingDirection direction);

std::optional<ArrayLengthHelperDescriptor>
resolveArrayLengthHelperDescriptor(const SemanticFieldType &type,
                                   const LoweredFieldFacts *fieldFacts,
                                   std::optional<std::uint32_t> prefixBitsOverride,
                                   HelperBindingDirection direction);

std::string resolveDelimiterValidateHelperSymbol(const SemanticFieldType &type,
                                                 const LoweredFieldFacts *fieldFacts);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_HELPER_SYMBOL_RESOLVER_H
