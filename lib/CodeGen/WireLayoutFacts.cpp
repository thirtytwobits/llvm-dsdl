#include "llvmdsdl/CodeGen/WireLayoutFacts.h"

#include <algorithm>

namespace llvmdsdl {

std::uint32_t resolveUnionTagBits(const SemanticSection &section,
                                  const LoweredSectionFacts *const sectionFacts) {
  if (sectionFacts && sectionFacts->unionTagBits) {
    return *sectionFacts->unionTagBits;
  }
  std::uint32_t tagBits = 8;
  for (const auto &f : section.fields) {
    if (!f.isPadding) {
      tagBits = std::max<std::uint32_t>(8U, f.unionTagBits);
      break;
    }
  }
  return tagBits;
}

} // namespace llvmdsdl
