//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Extracts lowered helper facts from MLIR contracts.
///
/// This implementation runs transform pipelines and collects helper metadata required by language-specific code
/// generators.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/ilist_iterator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "llvmdsdl/Transforms/Passes.h"
#include "llvmdsdl/Transforms/LoweredSerDesContract.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"

namespace llvmdsdl
{

namespace
{

std::int64_t nonNegative(const std::int64_t value)
{
    return std::max<std::int64_t>(value, 0);
}

bool isSupportedArrayKind(llvm::StringRef arrayKind)
{
    return arrayKind == "none" || arrayKind == "fixed" || arrayKind == "variable_inclusive" ||
           arrayKind == "variable_exclusive";
}

}  // namespace

std::string loweredTypeKey(const std::string& name, std::uint32_t major, std::uint32_t minor)
{
    return name + ":" + std::to_string(major) + ":" + std::to_string(minor);
}

const LoweredFieldFacts* findLoweredFieldFacts(const LoweredSectionFacts* const sectionFacts,
                                               const std::string&               fieldName)
{
    if (sectionFacts == nullptr)
    {
        return nullptr;
    }
    const auto it = sectionFacts->fieldsByName.find(fieldName);
    if (it == sectionFacts->fieldsByName.end())
    {
        return nullptr;
    }
    return &it->second;
}

std::optional<std::uint32_t> loweredFieldArrayPrefixBits(const LoweredSectionFacts* const sectionFacts,
                                                         const std::string&               fieldName)
{
    const auto* const fieldFacts = findLoweredFieldFacts(sectionFacts, fieldName);
    if (fieldFacts == nullptr)
    {
        return std::nullopt;
    }
    return fieldFacts->arrayLengthPrefixBits;
}

bool collectLoweredFactsFromMlir(const SemanticModule&  semantic,
                                 mlir::ModuleOp         module,
                                 DiagnosticEngine&      diagnostics,
                                 const std::string&     backendLabel,
                                 LoweredFactsMap* const outFacts,
                                 const bool             optimizeLoweredSerDes)
{
    std::unordered_map<std::string, std::set<std::string>> keyToSections;
    LoweredFactsMap                                        loweredFacts;
    auto              loweredModule = mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));
    mlir::PassManager pm(module.getContext());
    pm.addPass(createLowerDSDLSerializationPass());
    if (optimizeLoweredSerDes)
    {
        addOptimizeLoweredSerDesPipeline(pm);
    }
    if (mlir::failed(pm.run(*loweredModule)))
    {
        diagnostics.error({"<mlir>", 1, 1},
                          "failed to run lower-dsdl-serialization for " + backendLabel + " backend validation");
        return false;
    }
    const auto contractVersion =
        loweredModule->getOperation()->getAttrOfType<mlir::IntegerAttr>(kLoweredSerDesContractVersionAttr);
    if (!contractVersion)
    {
        diagnostics.error({"<mlir>", 1, 1},
                          "lowered SerDes contract missing module attribute '" +
                              std::string(kLoweredSerDesContractVersionAttr) + "' for " + backendLabel +
                              " backend validation");
        return false;
    }
    if (contractVersion.getInt() != kLoweredSerDesContractVersion)
    {
        diagnostics.error({"<mlir>", 1, 1},
                          "lowered SerDes contract version mismatch for " + backendLabel +
                              " backend validation: expected " + std::to_string(kLoweredSerDesContractVersion) +
                              ", got " + std::to_string(contractVersion.getInt()));
        return false;
    }
    const auto contractProducer =
        loweredModule->getOperation()->getAttrOfType<mlir::StringAttr>(kLoweredSerDesContractProducerAttr);
    if (!contractProducer || contractProducer.getValue() != kLoweredSerDesContractProducer)
    {
        diagnostics.error({"<mlir>", 1, 1},
                          "lowered SerDes contract producer mismatch for " + backendLabel +
                              " backend validation: expected '" + std::string(kLoweredSerDesContractProducer) + "'");
        return false;
    }

    for (mlir::Operation& op : loweredModule->getBodyRegion().front())
    {
        if (op.getName().getStringRef() != "dsdl.schema")
        {
            continue;
        }

        const auto fullName = op.getAttrOfType<mlir::StringAttr>("full_name");
        const auto major    = op.getAttrOfType<mlir::IntegerAttr>("major");
        const auto minor    = op.getAttrOfType<mlir::IntegerAttr>("minor");
        if (!fullName || !major || !minor)
        {
            diagnostics.error({"<mlir>", 1, 1}, "dsdl.schema missing identity attributes");
            return false;
        }

        const auto key      = loweredTypeKey(fullName.getValue().str(),
                                        static_cast<std::uint32_t>(major.getInt()),
                                        static_cast<std::uint32_t>(minor.getInt()));
        auto&      sections = keyToSections[key];

        if (op.getNumRegions() == 0 || op.getRegion(0).empty())
        {
            diagnostics.error({"<mlir>", 1, 1}, "dsdl.schema has no body region for " + fullName.getValue().str());
            return false;
        }

        for (mlir::Operation& child : op.getRegion(0).front())
        {
            if (child.getName().getStringRef() != "dsdl.serialization_plan")
            {
                continue;
            }
            const auto planContractVersion = child.getAttrOfType<mlir::IntegerAttr>(kLoweredSerDesContractVersionAttr);
            if (!planContractVersion)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan missing lowered contract "
                                  "attribute '" +
                                      std::string(kLoweredSerDesContractVersionAttr) + "' for " +
                                      fullName.getValue().str());
                return false;
            }
            if (planContractVersion.getInt() != kLoweredSerDesContractVersion)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan lowered contract version mismatch for " +
                                      fullName.getValue().str() + ": expected " +
                                      std::to_string(kLoweredSerDesContractVersion) + ", got " +
                                      std::to_string(planContractVersion.getInt()));
                return false;
            }
            const auto planContractProducer = child.getAttrOfType<mlir::StringAttr>(kLoweredSerDesContractProducerAttr);
            if (!planContractProducer || planContractProducer.getValue() != kLoweredSerDesContractProducer)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan lowered contract producer mismatch for " +
                                      fullName.getValue().str() + ": expected '" +
                                      std::string(kLoweredSerDesContractProducer) + "'");
                return false;
            }
            std::string section;
            if (const auto sectionAttr = child.getAttrOfType<mlir::StringAttr>("section"))
            {
                section = sectionAttr.getValue().str();
            }
            auto& sectionFacts = loweredFacts[key][section];
            if (!sections.insert(section).second)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "duplicate dsdl.serialization_plan section '" + section + "' for " +
                                      fullName.getValue().str());
                return false;
            }

            if (!child.hasAttr(kLoweredPlanMarkerAttr))
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan missing lowered marker '" + std::string(kLoweredPlanMarkerAttr) +
                                      "' for " + fullName.getValue().str());
                return false;
            }

            const auto minBits      = child.getAttrOfType<mlir::IntegerAttr>(kLoweredMinBitsAttr);
            const auto maxBits      = child.getAttrOfType<mlir::IntegerAttr>(kLoweredMaxBitsAttr);
            const auto stepCount    = child.getAttrOfType<mlir::IntegerAttr>(kLoweredStepCountAttr);
            const auto fieldCount   = child.getAttrOfType<mlir::IntegerAttr>(kLoweredFieldCountAttr);
            const auto paddingCount = child.getAttrOfType<mlir::IntegerAttr>(kLoweredPaddingCountAttr);
            const auto alignCount   = child.getAttrOfType<mlir::IntegerAttr>(kLoweredAlignCountAttr);
            if (!minBits || !maxBits || !stepCount || !fieldCount || !paddingCount || !alignCount)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan missing lowered metadata for " + fullName.getValue().str());
                return false;
            }
            if (maxBits.getInt() < minBits.getInt())
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan has invalid lowered bit-range for " + fullName.getValue().str());
                return false;
            }
            if (stepCount.getInt() < 0 || fieldCount.getInt() < 0 || paddingCount.getInt() < 0 ||
                alignCount.getInt() < 0)
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan has invalid lowered counts for " + fullName.getValue().str());
                return false;
            }
            const auto capacityCheckHelper = child.getAttrOfType<mlir::StringAttr>(kLoweredCapacityCheckHelperAttr);
            if (!capacityCheckHelper || capacityCheckHelper.getValue().empty())
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan missing lowered capacity helper "
                                  "for " +
                                      fullName.getValue().str());
                return false;
            }
            sectionFacts.capacityCheckHelper = capacityCheckHelper.getValue().str();
            if (!loweredModule->lookupSymbol<mlir::func::FuncOp>(sectionFacts.capacityCheckHelper))
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan references missing lowered "
                                  "capacity helper '" +
                                      sectionFacts.capacityCheckHelper + "' for " + fullName.getValue().str());
                return false;
            }

            if (child.hasAttr("is_union"))
            {
                const auto unionTagBits     = child.getAttrOfType<mlir::IntegerAttr>("union_tag_bits");
                const auto unionOptionCount = child.getAttrOfType<mlir::IntegerAttr>("union_option_count");
                if (!unionTagBits || !unionOptionCount)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "union plan missing union_tag_bits/union_option_count for " +
                                          fullName.getValue().str());
                    return false;
                }
                if (unionTagBits.getInt() <= 0 || unionTagBits.getInt() > 64 || unionOptionCount.getInt() <= 0)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "union plan has invalid union metadata for " + fullName.getValue().str());
                    return false;
                }
                const auto serUnionTagHelper   = child.getAttrOfType<mlir::StringAttr>(kLoweredSerUnionTagHelperAttr);
                const auto deserUnionTagHelper = child.getAttrOfType<mlir::StringAttr>(kLoweredDeserUnionTagHelperAttr);
                const auto unionTagValidateHelper =
                    child.getAttrOfType<mlir::StringAttr>(kLoweredUnionTagValidateHelperAttr);
                if (!serUnionTagHelper || !deserUnionTagHelper || !unionTagValidateHelper)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "union plan missing lowered union-tag helpers for " + fullName.getValue().str());
                    return false;
                }
                if (serUnionTagHelper.getValue().empty() || deserUnionTagHelper.getValue().empty() ||
                    unionTagValidateHelper.getValue().empty())
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "union plan has empty lowered union-tag helper names for " +
                                          fullName.getValue().str());
                    return false;
                }
                sectionFacts.unionTagBits           = static_cast<std::uint32_t>(unionTagBits.getInt());
                sectionFacts.unionTagValidateHelper = unionTagValidateHelper.getValue().str();
                sectionFacts.serUnionTagHelper      = serUnionTagHelper.getValue().str();
                sectionFacts.deserUnionTagHelper    = deserUnionTagHelper.getValue().str();
                if (!loweredModule->lookupSymbol<mlir::func::FuncOp>(sectionFacts.unionTagValidateHelper) ||
                    !loweredModule->lookupSymbol<mlir::func::FuncOp>(sectionFacts.serUnionTagHelper) ||
                    !loweredModule->lookupSymbol<mlir::func::FuncOp>(sectionFacts.deserUnionTagHelper))
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "union plan references missing lowered union-tag "
                                      "helpers for " +
                                          fullName.getValue().str());
                    return false;
                }
            }

            if (child.getNumRegions() == 0 || child.getRegion(0).empty())
            {
                diagnostics.error({"<mlir>", 1, 1}, "serialization plan has no body for " + fullName.getValue().str());
                return false;
            }

            std::int64_t           observedStepCount    = 0;
            std::int64_t           observedFieldCount   = 0;
            std::int64_t           observedPaddingCount = 0;
            std::int64_t           observedAlignCount   = 0;
            std::set<std::int64_t> seenStepIndexes;

            for (mlir::Operation& step : child.getRegion(0).front())
            {
                const auto stepName = step.getName().getStringRef();
                if (stepName == "dsdl.align")
                {
                    const auto bits      = step.getAttrOfType<mlir::IntegerAttr>("bits");
                    const auto stepIndex = step.getAttrOfType<mlir::IntegerAttr>("step_index");
                    if (!bits || !stepIndex)
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "dsdl.align missing lowered metadata for " + fullName.getValue().str());
                        return false;
                    }
                    if (bits.getInt() <= 1)
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "dsdl.align must not contain no-op alignment for " +
                                              fullName.getValue().str());
                        return false;
                    }
                    if (!seenStepIndexes.insert(stepIndex.getInt()).second)
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "serialization plan has duplicate step_index values for " +
                                              fullName.getValue().str());
                        return false;
                    }
                    ++observedStepCount;
                    ++observedAlignCount;
                    continue;
                }
                if (stepName != "dsdl.io")
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "unsupported operation in serialization plan body for " +
                                          fullName.getValue().str());
                    return false;
                }

                const auto stepMinBits = step.getAttrOfType<mlir::IntegerAttr>("min_bits");
                const auto stepMaxBits = step.getAttrOfType<mlir::IntegerAttr>("max_bits");
                const auto loweredBits = step.getAttrOfType<mlir::IntegerAttr>("lowered_bits");
                const auto stepIndex   = step.getAttrOfType<mlir::IntegerAttr>("step_index");
                if (!stepMinBits || !stepMaxBits || !loweredBits || !stepIndex)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "dsdl.io missing lowered sizing/order metadata for " + fullName.getValue().str());
                    return false;
                }
                if (stepMinBits.getInt() < 0 || stepMaxBits.getInt() < stepMinBits.getInt() ||
                    loweredBits.getInt() < 0 || loweredBits.getInt() != stepMaxBits.getInt())
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "dsdl.io has invalid lowered sizing metadata for " + fullName.getValue().str());
                    return false;
                }
                if (!seenStepIndexes.insert(stepIndex.getInt()).second)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "serialization plan has duplicate step_index values for " +
                                          fullName.getValue().str());
                    return false;
                }
                ++observedStepCount;

                const auto scalarCategory = step.getAttrOfType<mlir::StringAttr>("scalar_category");
                const auto arrayKind      = step.getAttrOfType<mlir::StringAttr>("array_kind");
                const auto kind           = step.getAttrOfType<mlir::StringAttr>("kind");
                const auto bitLength      = step.getAttrOfType<mlir::IntegerAttr>("bit_length");
                const auto alignmentBits  = step.getAttrOfType<mlir::IntegerAttr>("alignment_bits");
                if (!scalarCategory || !arrayKind || !kind || !bitLength || !alignmentBits)
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "dsdl.io missing core type metadata for " + fullName.getValue().str());
                    return false;
                }
                if (!isSupportedArrayKind(arrayKind.getValue()))
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "dsdl.io has unsupported array_kind for " + fullName.getValue().str());
                    return false;
                }
                if (kind.getValue() != "field" && kind.getValue() != "padding")
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "dsdl.io has unsupported kind for " + fullName.getValue().str());
                    return false;
                }
                const bool isPadding = kind.getValue() == "padding";
                if (isPadding)
                {
                    ++observedPaddingCount;
                }
                else
                {
                    ++observedFieldCount;
                }
                auto requireStepHelperSymbol = [&](llvm::StringRef attrName,
                                                   llvm::StringRef helperLabel) -> mlir::StringAttr {
                    const auto helper = step.getAttrOfType<mlir::StringAttr>(attrName);
                    if (!helper || helper.getValue().empty())
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "dsdl.io missing lowered " + helperLabel.str() + " helper '" +
                                              attrName.str() + "' for " + fullName.getValue().str());
                        return {};
                    }
                    if (!loweredModule->lookupSymbol<mlir::func::FuncOp>(helper.getValue().str()))
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "dsdl.io references missing lowered " + helperLabel.str() + " helper '" +
                                              helper.getValue().str() + "' for " + fullName.getValue().str());
                        return {};
                    }
                    return helper;
                };

                const auto arrayPrefixBits = step.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits");
                if (arrayKind.getValue().starts_with("variable") && (!arrayPrefixBits || arrayPrefixBits.getInt() <= 0))
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "variable array step missing valid prefix width for " +
                                          fullName.getValue().str());
                    return false;
                }
                if (!isPadding && arrayKind.getValue().starts_with("variable"))
                {
                    if (!requireStepHelperSymbol("lowered_ser_array_length_prefix_helper", "array-length-prefix") ||
                        !requireStepHelperSymbol("lowered_deser_array_length_prefix_helper", "array-length-prefix") ||
                        !requireStepHelperSymbol("lowered_array_length_validate_helper", "array-length-validate"))
                    {
                        return false;
                    }
                }

                if (!isPadding)
                {
                    const auto category = scalarCategory.getValue();
                    if (category == "unsigned" || category == "byte" || category == "utf8")
                    {
                        if (!requireStepHelperSymbol("lowered_ser_unsigned_helper", "scalar-unsigned") ||
                            !requireStepHelperSymbol("lowered_deser_unsigned_helper", "scalar-unsigned"))
                        {
                            return false;
                        }
                    }
                    else if (category == "signed")
                    {
                        if (!requireStepHelperSymbol("lowered_ser_signed_helper", "scalar-signed") ||
                            !requireStepHelperSymbol("lowered_deser_signed_helper", "scalar-signed"))
                        {
                            return false;
                        }
                    }
                    else if (category == "float")
                    {
                        if (!requireStepHelperSymbol("lowered_ser_float_helper", "scalar-float") ||
                            !requireStepHelperSymbol("lowered_deser_float_helper", "scalar-float"))
                        {
                            return false;
                        }
                    }
                }

                if (scalarCategory.getValue() == "composite")
                {
                    const auto compositeFullName  = step.getAttrOfType<mlir::StringAttr>("composite_full_name");
                    const auto compositeCTypeName = step.getAttrOfType<mlir::StringAttr>("composite_c_type_name");
                    if (!compositeFullName || !compositeCTypeName)
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "composite dsdl.io missing target metadata for " + fullName.getValue().str());
                        return false;
                    }
                    const auto compositeSealed = step.getAttrOfType<mlir::BoolAttr>("composite_sealed");
                    if (compositeSealed && !compositeSealed.getValue() &&
                        !step.getAttrOfType<mlir::IntegerAttr>("composite_extent_bits"))
                    {
                        diagnostics.error({"<mlir>", 1, 1},
                                          "delimited composite step missing composite_extent_bits for " +
                                              fullName.getValue().str());
                        return false;
                    }
                    if (!isPadding && compositeSealed && !compositeSealed.getValue() &&
                        !requireStepHelperSymbol("lowered_delimiter_validate_helper", "delimiter-validate"))
                    {
                        return false;
                    }
                }

                if (const auto fieldNameAttr = step.getAttrOfType<mlir::StringAttr>("name"); fieldNameAttr)
                {
                    auto& fieldFacts = sectionFacts.fieldsByName[fieldNameAttr.getValue().str()];
                    if (const auto stepIndex = step.getAttrOfType<mlir::IntegerAttr>("step_index"))
                    {
                        fieldFacts.stepIndex = nonNegative(stepIndex.getInt());
                    }
                    if (arrayKind.getValue().starts_with("variable") && arrayPrefixBits && arrayPrefixBits.getInt() > 0)
                    {
                        fieldFacts.arrayLengthPrefixBits = static_cast<std::uint32_t>(arrayPrefixBits.getInt());
                        if (const auto serArrayPrefixHelper =
                                step.getAttrOfType<mlir::StringAttr>("lowered_ser_array_length_prefix_helper"))
                        {
                            fieldFacts.serArrayLengthPrefixHelper = serArrayPrefixHelper.getValue().str();
                        }
                        if (const auto deserArrayPrefixHelper =
                                step.getAttrOfType<mlir::StringAttr>("lowered_deser_array_length_prefix_helper"))
                        {
                            fieldFacts.deserArrayLengthPrefixHelper = deserArrayPrefixHelper.getValue().str();
                        }
                        if (const auto arrayValidateHelper =
                                step.getAttrOfType<mlir::StringAttr>("lowered_array_length_validate_helper"))
                        {
                            fieldFacts.arrayLengthValidateHelper = arrayValidateHelper.getValue().str();
                        }
                    }
                    if (const auto serUnsigned = step.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper"))
                    {
                        fieldFacts.serUnsignedHelper = serUnsigned.getValue().str();
                    }
                    if (const auto deserUnsigned =
                            step.getAttrOfType<mlir::StringAttr>("lowered_deser_unsigned_helper"))
                    {
                        fieldFacts.deserUnsignedHelper = deserUnsigned.getValue().str();
                    }
                    if (const auto serSigned = step.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper"))
                    {
                        fieldFacts.serSignedHelper = serSigned.getValue().str();
                    }
                    if (const auto deserSigned = step.getAttrOfType<mlir::StringAttr>("lowered_deser_signed_helper"))
                    {
                        fieldFacts.deserSignedHelper = deserSigned.getValue().str();
                    }
                    if (const auto serFloat = step.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper"))
                    {
                        fieldFacts.serFloatHelper = serFloat.getValue().str();
                    }
                    if (const auto deserFloat = step.getAttrOfType<mlir::StringAttr>("lowered_deser_float_helper"))
                    {
                        fieldFacts.deserFloatHelper = deserFloat.getValue().str();
                    }
                    if (const auto delimiterValidateHelper =
                            step.getAttrOfType<mlir::StringAttr>("lowered_delimiter_validate_helper"))
                    {
                        fieldFacts.delimiterValidateHelper = delimiterValidateHelper.getValue().str();
                    }
                }
            }

            if (observedStepCount != stepCount.getInt() || observedFieldCount != fieldCount.getInt() ||
                observedPaddingCount != paddingCount.getInt() || observedAlignCount != alignCount.getInt())
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "serialization plan lowered counts do not match plan body for " +
                                      fullName.getValue().str());
                return false;
            }
            for (const auto stepIndex : seenStepIndexes)
            {
                if (stepIndex < 0 || stepIndex >= stepCount.getInt())
                {
                    diagnostics.error({"<mlir>", 1, 1},
                                      "serialization plan step_index out of declared bounds for " +
                                          fullName.getValue().str());
                    return false;
                }
            }
        }
    }

    for (const auto& def : semantic.definitions)
    {
        const auto key = loweredTypeKey(def.info.fullName, def.info.majorVersion, def.info.minorVersion);
        const auto it  = keyToSections.find(key);
        if (it == keyToSections.end())
        {
            diagnostics.error({"<mlir>", 1, 1}, "missing dsdl.schema for " + def.info.fullName);
            return false;
        }

        std::set<std::string> expectedSections;
        if (def.isService)
        {
            expectedSections.insert("request");
            expectedSections.insert("response");
        }
        else
        {
            expectedSections.insert("");
        }

        for (const auto& sectionName : expectedSections)
        {
            if (!it->second.contains(sectionName))
            {
                diagnostics.error({"<mlir>", 1, 1},
                                  "missing dsdl.serialization_plan section '" + sectionName + "' for " +
                                      def.info.fullName);
                return false;
            }
        }
    }

    if (outFacts != nullptr)
    {
        *outFacts = std::move(loweredFacts);
    }
    return true;
}

}  // namespace llvmdsdl
