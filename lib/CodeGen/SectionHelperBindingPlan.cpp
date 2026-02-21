//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Plans section helper bindings required by lowered serdes bodies.
///
/// The planner determines which helper descriptors are needed and in what order they should be bound for rendering.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"

#include <set>
#include <string>
#include <utility>

#include "llvmdsdl/CodeGen/WireLayoutFacts.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/Semantics/Model.h"

namespace llvmdsdl
{

SectionHelperBindingPlan buildSectionHelperBindingPlan(const SemanticSection&       section,
                                                       const LoweredSectionFacts*   sectionFacts,
                                                       const HelperBindingDirection direction)
{
    SectionHelperBindingPlan out;
    std::set<std::string>    emittedSymbols;

    const auto shared =
        buildSharedSerDesHelperDescriptors(section,
                                           sectionFacts ? sectionFacts->capacityCheckHelper : std::string{},
                                           sectionFacts ? sectionFacts->unionTagValidateHelper : std::string{});
    if (shared.capacityCheck)
    {
        emittedSymbols.insert(shared.capacityCheck->symbol);
        out.capacityCheck = shared.capacityCheck;
    }

    if (section.isUnion)
    {
        if (shared.unionTagValidate && emittedSymbols.insert(shared.unionTagValidate->symbol).second)
        {
            out.unionTagValidate = shared.unionTagValidate;
        }
        const auto symbol = (direction == HelperBindingDirection::Serialize)
                                ? (sectionFacts ? sectionFacts->serUnionTagHelper : std::string{})
                                : (sectionFacts ? sectionFacts->deserUnionTagHelper : std::string{});
        if (!symbol.empty() && emittedSymbols.insert(symbol).second)
        {
            out.unionTagMask = UnionTagMaskBindingDescriptor{symbol, resolveUnionTagBits(section, sectionFacts)};
        }
    }

    for (const auto& field : section.fields)
    {
        if (field.isPadding)
        {
            continue;
        }
        const auto* const fieldFacts = findLoweredFieldFacts(sectionFacts, field.name);
        const auto        scalarDescriptor =
            buildScalarHelperDescriptor(field,
                                        ScalarHelperSymbols{fieldFacts ? fieldFacts->serUnsignedHelper : std::string{},
                                                            fieldFacts ? fieldFacts->deserUnsignedHelper
                                                                       : std::string{},
                                                            fieldFacts ? fieldFacts->serSignedHelper : std::string{},
                                                            fieldFacts ? fieldFacts->deserSignedHelper : std::string{},
                                                            fieldFacts ? fieldFacts->serFloatHelper : std::string{},
                                                            fieldFacts ? fieldFacts->deserFloatHelper : std::string{}});
        if (scalarDescriptor)
        {
            const auto& symbol = (direction == HelperBindingDirection::Serialize) ? scalarDescriptor->serSymbol
                                                                                  : scalarDescriptor->deserSymbol;
            if (!symbol.empty() && emittedSymbols.insert(symbol).second)
            {
                out.scalarBindings.push_back(ScalarBindingDescriptor{symbol, *scalarDescriptor});
            }
        }

        const auto delimiterDescriptor =
            buildDelimiterValidateHelperDescriptor(field,
                                                   fieldFacts ? fieldFacts->delimiterValidateHelper : std::string{});
        if (delimiterDescriptor && emittedSymbols.insert(delimiterDescriptor->symbol).second)
        {
            out.delimiterValidateBindings.push_back(DelimiterValidateBindingDescriptor{delimiterDescriptor->symbol});
        }

        const auto arrayDescriptor =
            buildArrayLengthHelperDescriptor(field,
                                             loweredFieldArrayPrefixBits(sectionFacts, field.name),
                                             (direction == HelperBindingDirection::Serialize)
                                                 ? (fieldFacts ? fieldFacts->serArrayLengthPrefixHelper : std::string{})
                                                 : (fieldFacts ? fieldFacts->deserArrayLengthPrefixHelper
                                                               : std::string{}),
                                             fieldFacts ? fieldFacts->arrayLengthValidateHelper : std::string{});
        if (!arrayDescriptor)
        {
            continue;
        }
        if (!arrayDescriptor->prefixSymbol.empty() && emittedSymbols.insert(arrayDescriptor->prefixSymbol).second)
        {
            out.arrayPrefixBindings.push_back(
                ArrayPrefixBindingDescriptor{arrayDescriptor->prefixSymbol, arrayDescriptor->prefixBits});
        }
        if (!arrayDescriptor->validateSymbol.empty() && emittedSymbols.insert(arrayDescriptor->validateSymbol).second)
        {
            out.arrayValidateBindings.push_back(
                ArrayValidateBindingDescriptor{arrayDescriptor->validateSymbol, arrayDescriptor->capacity});
        }
    }

    return out;
}

}  // namespace llvmdsdl
