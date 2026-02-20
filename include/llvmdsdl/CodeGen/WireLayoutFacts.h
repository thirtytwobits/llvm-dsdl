//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Helpers for resolving effective wire-layout facts from semantic and lowered metadata.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H
#define LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H

#include <cstdint>

namespace llvmdsdl
{
struct LoweredSectionFacts;
struct SemanticSection;

/// @file
/// @brief Wire-layout fact resolution helpers.

/// @brief Resolves effective union tag width for a semantic section.
/// @details Uses lowered facts when available, otherwise falls back to
/// @details semantic section metadata.
/// @param[in] section Semantic section.
/// @param[in] sectionFacts Optional lowered section facts.
/// @return Union tag width in bits.
std::uint32_t resolveUnionTagBits(const SemanticSection& section, const LoweredSectionFacts* sectionFacts);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_WIRE_LAYOUT_FACTS_H
