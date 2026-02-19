#include "llvmdsdl/CodeGen/TsLoweredPlan.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <iostream>
#include <string>

bool runTsLoweredPlanTests()
{
    {
        llvmdsdl::SemanticSection section;

        llvmdsdl::SemanticField first;
        first.name                        = "first";
        first.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        first.resolvedType.bitLength      = 8;
        section.fields.push_back(first);

        llvmdsdl::SemanticField payload;
        payload.name                               = "payload";
        payload.resolvedType.scalarCategory        = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        payload.resolvedType.bitLength             = 8;
        payload.resolvedType.arrayKind             = llvmdsdl::ArrayKind::VariableInclusive;
        payload.resolvedType.arrayCapacity         = 8;
        payload.resolvedType.arrayLengthPrefixBits = 8;
        section.fields.push_back(payload);

        llvmdsdl::SemanticField pad;
        pad.name                        = "_pad0";
        pad.isPadding                   = true;
        pad.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::Void;
        pad.resolvedType.bitLength      = 3;
        section.fields.push_back(pad);

        llvmdsdl::LoweredSectionFacts facts;
        facts.fieldsByName["first"].stepIndex               = 20;
        facts.fieldsByName["payload"].stepIndex             = 10;
        facts.fieldsByName["payload"].arrayLengthPrefixBits = 16;
        facts.fieldsByName["_pad0"].stepIndex               = 30;

        auto orderedOrErr = llvmdsdl::buildTsOrderedFieldSteps(section, &facts);
        if (!orderedOrErr)
        {
            std::cerr << "ts lowered plan non-union unexpectedly failed: " << llvm::toString(orderedOrErr.takeError())
                      << "\n";
            return false;
        }
        const auto& ordered = *orderedOrErr;
        if (ordered.size() != 3)
        {
            std::cerr << "ts lowered plan non-union size mismatch\n";
            return false;
        }
        if (ordered[0].field == nullptr || ordered[0].field->name != "payload" || !ordered[0].arrayLengthPrefixBits ||
            *ordered[0].arrayLengthPrefixBits != 16U)
        {
            std::cerr << "ts lowered plan non-union reordered payload mismatch\n";
            return false;
        }
        if (ordered[1].field == nullptr || ordered[1].field->name != "first")
        {
            std::cerr << "ts lowered plan non-union first field mismatch\n";
            return false;
        }
        if (ordered[2].field == nullptr || !ordered[2].field->isPadding)
        {
            std::cerr << "ts lowered plan non-union padding mismatch\n";
            return false;
        }
    }

    {
        llvmdsdl::SemanticSection section;
        section.isUnion = true;

        llvmdsdl::SemanticField alpha;
        alpha.name                        = "alpha";
        alpha.unionOptionIndex            = 1;
        alpha.unionTagBits                = 2;
        alpha.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        alpha.resolvedType.bitLength      = 8;
        section.fields.push_back(alpha);

        llvmdsdl::SemanticField beta;
        beta.name                               = "beta";
        beta.unionOptionIndex                   = 0;
        beta.unionTagBits                       = 2;
        beta.resolvedType.scalarCategory        = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        beta.resolvedType.bitLength             = 8;
        beta.resolvedType.arrayKind             = llvmdsdl::ArrayKind::VariableInclusive;
        beta.resolvedType.arrayCapacity         = 4;
        beta.resolvedType.arrayLengthPrefixBits = 8;
        section.fields.push_back(beta);

        llvmdsdl::LoweredSectionFacts facts;
        facts.fieldsByName["alpha"].stepIndex            = 20;
        facts.fieldsByName["beta"].stepIndex             = 10;
        facts.fieldsByName["beta"].arrayLengthPrefixBits = 12;

        auto orderedOrErr = llvmdsdl::buildTsOrderedFieldSteps(section, &facts);
        if (!orderedOrErr)
        {
            std::cerr << "ts lowered plan union unexpectedly failed: " << llvm::toString(orderedOrErr.takeError())
                      << "\n";
            return false;
        }
        const auto& ordered = *orderedOrErr;
        if (ordered.size() != 2)
        {
            std::cerr << "ts lowered plan union size mismatch\n";
            return false;
        }
        if (ordered[0].field == nullptr || ordered[0].field->name != "beta" || !ordered[0].arrayLengthPrefixBits ||
            *ordered[0].arrayLengthPrefixBits != 12U)
        {
            std::cerr << "ts lowered plan union beta ordering mismatch\n";
            return false;
        }
        if (ordered[1].field == nullptr || ordered[1].field->name != "alpha")
        {
            std::cerr << "ts lowered plan union alpha ordering mismatch\n";
            return false;
        }
    }

    {
        llvmdsdl::SemanticSection section;
        llvmdsdl::SemanticField   first;
        first.name                        = "first";
        first.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        first.resolvedType.bitLength      = 8;
        section.fields.push_back(first);

        auto orderedOrErr = llvmdsdl::buildTsOrderedFieldSteps(section, nullptr);
        if (orderedOrErr)
        {
            std::cerr << "ts lowered plan should fail without section facts\n";
            return false;
        }
        const std::string errText = llvm::toString(orderedOrErr.takeError());
        if (!llvm::StringRef(errText).contains("missing lowered section facts"))
        {
            std::cerr << "unexpected missing-facts error text: " << errText << "\n";
            return false;
        }
    }

    {
        llvmdsdl::SemanticSection section;

        llvmdsdl::SemanticField first;
        first.name                        = "first";
        first.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        first.resolvedType.bitLength      = 8;
        section.fields.push_back(first);

        llvmdsdl::SemanticField second;
        second.name                        = "second";
        second.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        second.resolvedType.bitLength      = 8;
        section.fields.push_back(second);

        llvmdsdl::LoweredSectionFacts facts;
        facts.fieldsByName["first"].stepIndex              = 1;
        facts.fieldsByName["second"].arrayLengthPrefixBits = 8;

        auto orderedOrErr = llvmdsdl::buildTsOrderedFieldSteps(section, &facts);
        if (orderedOrErr)
        {
            std::cerr << "ts lowered plan should fail for missing step index\n";
            return false;
        }
        const std::string errText = llvm::toString(orderedOrErr.takeError());
        if (!llvm::StringRef(errText).contains("missing lowered step index"))
        {
            std::cerr << "unexpected missing-step-index error text: " << errText << "\n";
            return false;
        }
    }

    return true;
}
