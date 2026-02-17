#ifndef LLVMDSDL_CODEGEN_ARRAY_WIRE_PLAN_H
#define LLVMDSDL_CODEGEN_ARRAY_WIRE_PLAN_H

#include "llvmdsdl/CodeGen/HelperSymbolResolver.h"

#include <cstdint>
#include <optional>

namespace llvmdsdl {

struct ArrayWirePlan final {
  bool variable{false};
  std::uint32_t prefixBits{0};
  std::optional<ArrayLengthHelperDescriptor> descriptor;
};

ArrayWirePlan buildArrayWirePlan(
    const SemanticFieldType &type, const LoweredFieldFacts *fieldFacts,
    std::optional<std::uint32_t> prefixBitsOverride,
    HelperBindingDirection direction);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_ARRAY_WIRE_PLAN_H
