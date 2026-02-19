//===----------------------------------------------------------------------===//
///
/// @file
/// Builds array wire-layout planning facts for lowered sections.
///
/// The implementation combines semantic array properties with lowered helper contracts so emitters can share consistent
/// serialization behavior.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/ArrayWirePlan.h"

#include "llvmdsdl/CodeGen/TypeStorage.h"

namespace llvmdsdl
{

ArrayWirePlan buildArrayWirePlan(const SemanticFieldType&           type,
                                 const LoweredFieldFacts*           fieldFacts,
                                 const std::optional<std::uint32_t> prefixBitsOverride,
                                 const HelperBindingDirection       direction)
{
    ArrayWirePlan out;
    out.variable   = isVariableArray(type.arrayKind);
    out.prefixBits = prefixBitsOverride.value_or(static_cast<std::uint32_t>(type.arrayLengthPrefixBits));
    out.descriptor = resolveArrayLengthHelperDescriptor(type, fieldFacts, out.prefixBits, direction);
    return out;
}

}  // namespace llvmdsdl
