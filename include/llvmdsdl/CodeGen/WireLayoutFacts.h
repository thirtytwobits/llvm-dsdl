#ifndef LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H
#define LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/Semantics/Model.h"

#include <cstdint>

namespace llvmdsdl {

std::uint32_t resolveUnionTagBits(const SemanticSection &section,
                                  const LoweredSectionFacts *sectionFacts);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H
