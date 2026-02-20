//===----------------------------------------------------------------------===//
///
/// @file
/// Builds TypeScript-specific lowering plans from render IR.
///
/// The planning utilities convert generic render steps into TypeScript execution primitives used by the TS emitter.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/TsLoweredPlan.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <cstddef>
#include <utility>

#include "llvmdsdl/CodeGen/LoweredRenderIR.h"
#include "llvm/Support/Error.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/SerDesStatementPlan.h"
#include "llvmdsdl/Semantics/BitLengthSet.h"

namespace llvmdsdl
{
namespace
{

bool isTsKeyword(const std::string& name)
{
    static const std::set<std::string> kKeywords =
        {"break", "case",       "catch",     "class",      "const",   "continue", "debugger",  "default", "delete",
         "do",    "else",       "enum",      "export",     "extends", "false",    "finally",   "for",     "function",
         "if",    "import",     "in",        "instanceof", "new",     "null",     "return",    "super",   "switch",
         "this",  "throw",      "true",      "try",        "typeof",  "var",      "void",      "while",   "with",
         "as",    "implements", "interface", "let",        "package", "private",  "protected", "public",  "static",
         "yield", "any",        "boolean",   "number",     "string",  "symbol",   "type",      "from",    "of"};
    return kKeywords.contains(name);
}

std::string sanitizeTsIdent(std::string name)
{
    if (name.empty())
    {
        return "_";
    }
    for (char& c : name)
    {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
        {
            c = '_';
        }
    }
    if (std::isdigit(static_cast<unsigned char>(name.front())))
    {
        name.insert(name.begin(), '_');
    }
    if (isTsKeyword(name))
    {
        name += "_";
    }
    return name;
}

std::string toSnakeCase(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);

    bool prevUnderscore = false;
    for (std::size_t i = 0; i < in.size(); ++i)
    {
        const char c    = in[i];
        const char prev = (i > 0) ? in[i - 1] : '\0';
        const char next = (i + 1 < in.size()) ? in[i + 1] : '\0';
        if (!std::isalnum(static_cast<unsigned char>(c)))
        {
            if (!out.empty() && !prevUnderscore)
            {
                out.push_back('_');
                prevUnderscore = true;
            }
            continue;
        }

        if (std::isupper(static_cast<unsigned char>(c)))
        {
            const bool boundary =
                std::islower(static_cast<unsigned char>(prev)) ||
                (std::isupper(static_cast<unsigned char>(prev)) && std::islower(static_cast<unsigned char>(next)));
            if (!out.empty() && !prevUnderscore && boundary)
            {
                out.push_back('_');
            }
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            prevUnderscore = false;
        }
        else
        {
            out.push_back(c);
            prevUnderscore = (c == '_');
        }
    }

    if (out.empty())
    {
        out = "_";
    }
    if (std::isdigit(static_cast<unsigned char>(out.front())))
    {
        out.insert(out.begin(), '_');
    }
    return sanitizeTsIdent(out);
}

std::size_t expectedStepCount(const SemanticSection& section)
{
    if (!section.isUnion)
    {
        return section.fields.size();
    }
    std::size_t count = 0;
    for (const auto& field : section.fields)
    {
        if (!field.isPadding)
        {
            ++count;
        }
    }
    return count;
}

}  // namespace

llvm::Expected<std::vector<TsOrderedFieldStep>> buildTsOrderedFieldSteps(const SemanticSection&           section,
                                                                         const LoweredSectionFacts* const sectionFacts)
{
    if (sectionFacts == nullptr)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "missing lowered section facts required for TypeScript runtime planning");
    }

    const auto renderIR = buildLoweredBodyRenderIR(section, sectionFacts, HelperBindingDirection::Serialize);
    std::vector<TsOrderedFieldStep> out;

    if (section.isUnion)
    {
        const auto* unionStep = renderIR.steps.empty() ? nullptr : &renderIR.steps.front();
        if (unionStep == nullptr || unionStep->kind != LoweredRenderStepKind::UnionDispatch)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered render IR missing union-dispatch step");
        }
        if (renderIR.steps.size() != 1U)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered union render IR unexpectedly contains non-dispatch steps");
        }
        out.reserve(unionStep->unionBranches.size());
        for (const auto& branch : unionStep->unionBranches)
        {
            if (branch.field == nullptr)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered union render IR contains null field branch");
            }
            const auto* const facts = findLoweredFieldFacts(sectionFacts, branch.field->name);
            if (facts == nullptr || !facts->stepIndex)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "missing lowered step index for union field '%s'",
                                               branch.field->name.c_str());
            }
            out.push_back(TsOrderedFieldStep{branch.field, branch.arrayLengthPrefixBits});
        }
    }
    else
    {
        out.reserve(renderIR.steps.size());
        for (const auto& step : renderIR.steps)
        {
            if (step.kind != LoweredRenderStepKind::Field && step.kind != LoweredRenderStepKind::Padding)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered struct render IR contains unsupported step kind");
            }
            if (step.fieldStep.field == nullptr)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "lowered struct render IR contains null field step");
            }
            const auto* const facts = findLoweredFieldFacts(sectionFacts, step.fieldStep.field->name);
            if (facts == nullptr || !facts->stepIndex)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "missing lowered step index for field '%s'",
                                               step.fieldStep.field->name.c_str());
            }
            out.push_back(TsOrderedFieldStep{step.fieldStep.field, step.fieldStep.arrayLengthPrefixBits});
        }
    }

    if (out.size() != expectedStepCount(section))
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "lowered render IR step count does not match semantic field count");
    }
    std::set<const SemanticField*> uniqueness;
    for (const auto& step : out)
    {
        if (!uniqueness.insert(step.field).second)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered render IR contains duplicate field references");
        }
    }

    return out;
}

llvm::Expected<TsRuntimeSectionPlan> buildTsRuntimeSectionPlan(const SemanticSection&           section,
                                                               const LoweredSectionFacts* const sectionFacts)
{
    TsRuntimeSectionPlan plan;
    plan.isUnion                        = section.isUnion;
    std::int64_t                maxBits = 0;
    std::optional<std::int64_t> unionTagBits;
    std::set<std::uint32_t>     unionOptionIndexes;
    auto                        orderedStepsOrErr = buildTsOrderedFieldSteps(section, sectionFacts);
    if (!orderedStepsOrErr)
    {
        return orderedStepsOrErr.takeError();
    }
    const auto& orderedSteps = *orderedStepsOrErr;
    for (const auto& orderedStep : orderedSteps)
    {
        if (orderedStep.field == nullptr)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "lowered TypeScript section plan contains null field");
        }
        const auto&  field     = *orderedStep.field;
        std::int64_t fieldBits = static_cast<std::int64_t>(field.resolvedType.bitLength);
        if (fieldBits < 0)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "field '%s' has invalid negative bit length",
                                           field.name.c_str());
        }

        TsRuntimeArrayKind arrayKind             = TsRuntimeArrayKind::None;
        std::int64_t       arrayCapacity         = 0;
        std::int64_t       arrayLengthPrefixBits = 0;
        if (field.resolvedType.arrayKind == ArrayKind::None)
        {
            arrayKind = TsRuntimeArrayKind::None;
        }
        else if (field.resolvedType.arrayKind == ArrayKind::Fixed)
        {
            if (field.resolvedType.arrayCapacity <= 0)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' has invalid fixed-array capacity",
                                               field.name.c_str());
            }
            arrayKind     = TsRuntimeArrayKind::Fixed;
            arrayCapacity = field.resolvedType.arrayCapacity;
        }
        else if (field.resolvedType.arrayKind == ArrayKind::VariableInclusive ||
                 field.resolvedType.arrayKind == ArrayKind::VariableExclusive)
        {
            const auto prefixBits = orderedStep.arrayLengthPrefixBits
                                        ? static_cast<std::int64_t>(*orderedStep.arrayLengthPrefixBits)
                                        : field.resolvedType.arrayLengthPrefixBits;
            if (field.resolvedType.arrayCapacity < 0 || prefixBits <= 0)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' has invalid variable-array capacity or prefix width",
                                               field.name.c_str());
            }
            arrayKind             = TsRuntimeArrayKind::Variable;
            arrayCapacity         = field.resolvedType.arrayCapacity;
            arrayLengthPrefixBits = prefixBits;
        }
        else
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "field '%s' has unsupported array kind",
                                           field.name.c_str());
        }

        TsRuntimeFieldKind kind = TsRuntimeFieldKind::Unsigned;
        switch (field.resolvedType.scalarCategory)
        {
        case SemanticScalarCategory::Bool:
            if (fieldBits != 1)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' bool bit length must be 1",
                                               field.name.c_str());
            }
            kind = TsRuntimeFieldKind::Bool;
            break;
        case SemanticScalarCategory::Byte:
        case SemanticScalarCategory::Utf8:
        case SemanticScalarCategory::UnsignedInt:
            kind = TsRuntimeFieldKind::Unsigned;
            break;
        case SemanticScalarCategory::SignedInt:
            kind = TsRuntimeFieldKind::Signed;
            break;
        case SemanticScalarCategory::Float:
            if (fieldBits != 16 && fieldBits != 32 && fieldBits != 64)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' has unsupported float bit length %lld",
                                               field.name.c_str(),
                                               static_cast<long long>(fieldBits));
            }
            kind = TsRuntimeFieldKind::Float;
            break;
        case SemanticScalarCategory::Composite:
            if (!field.resolvedType.compositeType)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' is composite but missing target type metadata",
                                               field.name.c_str());
            }
            if (fieldBits <= 0)
            {
                fieldBits = field.resolvedType.compositeExtentBits;
            }
            if (fieldBits <= 0)
            {
                fieldBits = field.resolvedType.bitLengthSet.max();
            }
            kind = TsRuntimeFieldKind::Composite;
            break;
        case SemanticScalarCategory::Void:
            if (arrayKind != TsRuntimeArrayKind::None)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "padding field '%s' cannot be an array",
                                               field.name.c_str());
            }
            kind = TsRuntimeFieldKind::Padding;
            break;
        }

        if (kind == TsRuntimeFieldKind::Composite || kind == TsRuntimeFieldKind::Padding)
        {
            if (fieldBits < 0)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' has invalid bit-length metadata",
                                               field.name.c_str());
            }
        }
        else
        {
            if (fieldBits <= 0 || fieldBits > 64)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "field '%s' has unsupported scalar bit length %lld",
                                               field.name.c_str(),
                                               static_cast<long long>(fieldBits));
            }
        }

        if (plan.isUnion && kind == TsRuntimeFieldKind::Padding)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "union section cannot contain padding runtime fields");
        }

        TsRuntimeFieldPlan fieldPlan;
        fieldPlan.fieldName = sanitizeTsIdent(toSnakeCase(field.name));
        fieldPlan.kind      = kind;
        fieldPlan.castMode  = field.resolvedType.castMode;
        fieldPlan.bitLength = fieldBits;
        fieldPlan.alignmentBits =
            std::max<std::int64_t>(1, static_cast<std::int64_t>(field.resolvedType.alignmentBits));
        fieldPlan.useBigInt =
            (kind == TsRuntimeFieldKind::Unsigned || kind == TsRuntimeFieldKind::Signed) && (fieldBits > 53);
        fieldPlan.compositeType           = field.resolvedType.compositeType;
        fieldPlan.compositeSealed         = field.resolvedType.compositeSealed;
        fieldPlan.compositePayloadMaxBits = field.resolvedType.bitLengthSet.max();
        fieldPlan.unionOptionIndex        = field.unionOptionIndex;
        fieldPlan.arrayKind               = arrayKind;
        fieldPlan.arrayCapacity           = arrayCapacity;
        fieldPlan.arrayLengthPrefixBits   = arrayLengthPrefixBits;
        if (fieldPlan.compositePayloadMaxBits < 0)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "composite field '%s' has invalid payload max bits",
                                           field.name.c_str());
        }
        plan.fields.push_back(fieldPlan);

        if (plan.isUnion)
        {
            const auto tagBits = static_cast<std::int64_t>(field.unionTagBits);
            if (tagBits <= 0 || tagBits > 53)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "union field '%s' has invalid union-tag width %lld",
                                               field.name.c_str(),
                                               static_cast<long long>(tagBits));
            }
            if (unionTagBits && *unionTagBits != tagBits)
            {
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                               "union section has inconsistent union-tag widths");
            }
            unionTagBits = tagBits;
            unionOptionIndexes.insert(field.unionOptionIndex);
        }

        if (arrayKind == TsRuntimeArrayKind::None)
        {
            maxBits += fieldBits;
        }
        else if (arrayKind == TsRuntimeArrayKind::Fixed)
        {
            maxBits += fieldBits * arrayCapacity;
        }
        else
        {
            maxBits += arrayLengthPrefixBits + (fieldBits * arrayCapacity);
        }
    }

    if (plan.isUnion)
    {
        if (plan.fields.empty() || !unionTagBits || unionOptionIndexes.size() != plan.fields.size())
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "union runtime plan metadata is incomplete or inconsistent");
        }
        std::sort(plan.fields.begin(),
                  plan.fields.end(),
                  [](const TsRuntimeFieldPlan& lhs, const TsRuntimeFieldPlan& rhs) {
                      return lhs.unionOptionIndex < rhs.unionOptionIndex;
                  });
        plan.unionTagBits          = *unionTagBits;
        std::int64_t maxOptionBits = 0;
        for (const auto& fieldPlan : plan.fields)
        {
            std::int64_t optionBits = fieldPlan.bitLength;
            if (fieldPlan.arrayKind == TsRuntimeArrayKind::Fixed)
            {
                optionBits = fieldPlan.bitLength * fieldPlan.arrayCapacity;
            }
            else if (fieldPlan.arrayKind == TsRuntimeArrayKind::Variable)
            {
                optionBits = fieldPlan.arrayLengthPrefixBits + (fieldPlan.bitLength * fieldPlan.arrayCapacity);
            }
            maxOptionBits = std::max(maxOptionBits, optionBits);
        }
        const auto fallbackMaxBits = plan.unionTagBits + maxOptionBits;
        plan.maxBits               = std::max(section.maxBitLength, fallbackMaxBits);
    }
    else
    {
        plan.maxBits = std::max(section.maxBitLength, maxBits);
    }
    return plan;
}

}  // namespace llvmdsdl
