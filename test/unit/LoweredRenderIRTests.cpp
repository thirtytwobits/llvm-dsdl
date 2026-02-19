#include "llvmdsdl/CodeGen/LoweredRenderIR.h"

#include <iostream>

bool runLoweredRenderIRTests()
{
    {
        llvmdsdl::SemanticSection section;
        section.serializationBufferSizeBits = 32;

        llvmdsdl::SemanticField scalar;
        scalar.name                        = "scalar";
        scalar.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        scalar.resolvedType.bitLength      = 8;
        section.fields.push_back(scalar);

        llvmdsdl::SemanticField pad;
        pad.name                        = "_pad0";
        pad.isPadding                   = true;
        pad.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::Void;
        pad.resolvedType.bitLength      = 1;
        section.fields.push_back(pad);

        llvmdsdl::LoweredSectionFacts facts;
        facts.capacityCheckHelper              = "cap";
        facts.fieldsByName["scalar"].stepIndex = 5;
        facts.fieldsByName["_pad0"].stepIndex  = 10;

        const auto renderIR =
            llvmdsdl::buildLoweredBodyRenderIR(section, &facts, llvmdsdl::HelperBindingDirection::Serialize);

        if (!renderIR.helperBindings.capacityCheck || renderIR.helperBindings.capacityCheck->symbol != "cap")
        {
            std::cerr << "lowered render ir missing capacity helper\n";
            return false;
        }
        if (renderIR.steps.size() != 2)
        {
            std::cerr << "lowered render ir non-union step count mismatch\n";
            return false;
        }
        if (renderIR.steps[0].kind != llvmdsdl::LoweredRenderStepKind::Field ||
            renderIR.steps[0].fieldStep.field == nullptr || renderIR.steps[0].fieldStep.field->name != "scalar")
        {
            std::cerr << "lowered render ir field step mismatch\n";
            return false;
        }
        if (renderIR.steps[1].kind != llvmdsdl::LoweredRenderStepKind::Padding ||
            renderIR.steps[1].fieldStep.field == nullptr || !renderIR.steps[1].fieldStep.field->isPadding)
        {
            std::cerr << "lowered render ir padding step mismatch\n";
            return false;
        }
    }

    {
        llvmdsdl::SemanticSection section;
        section.isUnion                     = true;
        section.serializationBufferSizeBits = 16;

        llvmdsdl::SemanticField alpha;
        alpha.name                        = "alpha";
        alpha.unionOptionIndex            = 2;
        alpha.unionTagBits                = 2;
        alpha.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        alpha.resolvedType.bitLength      = 8;
        section.fields.push_back(alpha);

        llvmdsdl::SemanticField beta;
        beta.name                        = "beta";
        beta.unionOptionIndex            = 0;
        beta.unionTagBits                = 2;
        beta.resolvedType.scalarCategory = llvmdsdl::SemanticScalarCategory::UnsignedInt;
        beta.resolvedType.bitLength      = 8;
        section.fields.push_back(beta);

        llvmdsdl::LoweredSectionFacts facts;
        facts.fieldsByName["alpha"].stepIndex = 20;
        facts.fieldsByName["beta"].stepIndex  = 10;
        facts.deserUnionTagHelper             = "tag_des";

        const auto renderIR =
            llvmdsdl::buildLoweredBodyRenderIR(section, &facts, llvmdsdl::HelperBindingDirection::Deserialize);

        if (renderIR.steps.size() != 1 || renderIR.steps.front().kind != llvmdsdl::LoweredRenderStepKind::UnionDispatch)
        {
            std::cerr << "lowered render ir union dispatch step mismatch\n";
            return false;
        }
        if (renderIR.steps.front().unionBranches.size() != 2 ||
            renderIR.steps.front().unionBranches[0].field == nullptr ||
            renderIR.steps.front().unionBranches[0].field->name != "beta" ||
            renderIR.steps.front().unionBranches[1].field == nullptr ||
            renderIR.steps.front().unionBranches[1].field->name != "alpha")
        {
            std::cerr << "lowered render ir union branch ordering mismatch\n";
            return false;
        }
        if (!renderIR.helperBindings.unionTagMask || renderIR.helperBindings.unionTagMask->symbol != "tag_des")
        {
            std::cerr << "lowered render ir union helper mismatch\n";
            return false;
        }
    }

    return true;
}
