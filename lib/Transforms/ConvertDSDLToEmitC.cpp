//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements conversion from DSDL dialect ops to EmitC constructs.
///
/// The pass lowers dialect-specific control flow and bit operations into EmitC-compatible forms for C code emission.
///
//===----------------------------------------------------------------------===//

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/ilist_iterator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/TypeName.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cstddef>
#include <memory>
#include <utility>

#include "llvmdsdl/Transforms/LoweredSerDesContract.h"
#include "llvmdsdl/Transforms/Passes.h"
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace llvmdsdl
{
namespace
{

std::int64_t nonNegative(std::int64_t value)
{
    return std::max<std::int64_t>(value, 0);
}

bool isVariableArrayKind(llvm::StringRef arrayKind)
{
    return arrayKind == "variable_inclusive" || arrayKind == "variable_exclusive";
}

bool isSupportedArrayKind(llvm::StringRef arrayKind)
{
    return arrayKind == "none" || arrayKind == "fixed" || isVariableArrayKind(arrayKind);
}

std::int64_t ioStepBits(mlir::Operation* ioOp)
{
    if (auto bits = ioOp->getAttrOfType<mlir::IntegerAttr>("lowered_bits"))
    {
        return nonNegative(bits.getInt());
    }
    if (auto bits = ioOp->getAttrOfType<mlir::IntegerAttr>("max_bits"))
    {
        return nonNegative(bits.getInt());
    }
    return 0;
}

std::string sectionSuffix(llvm::StringRef sectionName)
{
    if (sectionName == "request")
    {
        return "__request";
    }
    if (sectionName == "response")
    {
        return "__response";
    }
    return "";
}

enum class PlanStepKind
{
    Align,
    Padding,
    Field
};

struct PlanStep final
{
    PlanStepKind kind{PlanStepKind::Field};
    std::int64_t bits{0};
    std::string  name;
    std::string  cName;
    std::string  scalarCategory;
    std::string  castMode;
    std::string  arrayKind;
    std::int64_t bitLength{0};
    std::int64_t arrayCapacity{0};
    std::int64_t arrayLengthPrefixBits{0};
    std::int64_t alignmentBits{1};
    std::int64_t unionOptionIndex{0};
    std::int64_t unionTagBits{0};
    std::string  compositeCTypeName;
    std::string  serUnsignedHelper;
    std::string  deserUnsignedHelper;
    std::string  serSignedHelper;
    std::string  deserSignedHelper;
    std::string  serFloatHelper;
    std::string  deserFloatHelper;
    std::string  serArrayLengthPrefixHelper;
    std::string  deserArrayLengthPrefixHelper;
    std::string  arrayLengthValidateHelper;
    std::string  delimiterValidateHelper;
    bool         compositeSealed{true};
    std::int64_t compositeExtentBits{0};
};

mlir::LogicalResult verifyLoweredPlanContract(mlir::ModuleOp module, mlir::Operation* plan)
{
    if (!plan->hasAttr(kLoweredPlanMarkerAttr))
    {
        return plan->emitOpError("missing lowered marker attribute '" + std::string(kLoweredPlanMarkerAttr) + "'");
    }
    const auto minBits      = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredMinBitsAttr);
    const auto maxBits      = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredMaxBitsAttr);
    const auto stepCount    = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredStepCountAttr);
    const auto fieldCount   = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredFieldCountAttr);
    const auto paddingCount = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredPaddingCountAttr);
    const auto alignCount   = plan->getAttrOfType<mlir::IntegerAttr>(kLoweredAlignCountAttr);
    if (!minBits || !maxBits || !stepCount || !fieldCount || !paddingCount || !alignCount)
    {
        return plan->emitOpError("missing required lowered plan metadata");
    }
    if (minBits.getInt() < 0 || maxBits.getInt() < minBits.getInt() || stepCount.getInt() < 0 ||
        fieldCount.getInt() < 0 || paddingCount.getInt() < 0 || alignCount.getInt() < 0)
    {
        return plan->emitOpError("invalid lowered plan metadata values");
    }

    const auto capacityCheckHelper = plan->getAttrOfType<mlir::StringAttr>(kLoweredCapacityCheckHelperAttr);
    if (!capacityCheckHelper || capacityCheckHelper.getValue().empty())
    {
        return plan->emitOpError("missing lowered capacity-check helper attribute '" +
                                 std::string(kLoweredCapacityCheckHelperAttr) + "'");
    }
    if (!module.lookupSymbol<mlir::func::FuncOp>(capacityCheckHelper.getValue()))
    {
        return plan->emitOpError("missing lowered capacity-check helper symbol: " +
                                 capacityCheckHelper.getValue().str());
    }

    if (plan->hasAttr("is_union"))
    {
        const auto unionTagBits            = plan->getAttrOfType<mlir::IntegerAttr>("union_tag_bits");
        const auto unionOptionCount        = plan->getAttrOfType<mlir::IntegerAttr>("union_option_count");
        const auto unionTagValidateHelper  = plan->getAttrOfType<mlir::StringAttr>(kLoweredUnionTagValidateHelperAttr);
        const auto unionTagSerializeHelper = plan->getAttrOfType<mlir::StringAttr>(kLoweredSerUnionTagHelperAttr);
        const auto unionTagDeserializeHelper = plan->getAttrOfType<mlir::StringAttr>(kLoweredDeserUnionTagHelperAttr);
        if (!unionTagBits || !unionOptionCount || !unionTagValidateHelper || !unionTagSerializeHelper ||
            !unionTagDeserializeHelper)
        {
            return plan->emitOpError("missing required lowered union metadata");
        }
        if (unionTagBits.getInt() <= 0 || unionTagBits.getInt() > 64 || unionOptionCount.getInt() <= 0)
        {
            return plan->emitOpError("invalid lowered union metadata values");
        }
        if (!module.lookupSymbol<mlir::func::FuncOp>(unionTagValidateHelper.getValue()) ||
            !module.lookupSymbol<mlir::func::FuncOp>(unionTagSerializeHelper.getValue()) ||
            !module.lookupSymbol<mlir::func::FuncOp>(unionTagDeserializeHelper.getValue()))
        {
            return plan->emitOpError("missing lowered union-tag helper symbol body");
        }
    }

    if (plan->getNumRegions() == 0 || plan->getRegion(0).empty())
    {
        return plan->emitOpError("must contain a non-empty lowered plan body");
    }

    std::int64_t           observedStepCount    = 0;
    std::int64_t           observedFieldCount   = 0;
    std::int64_t           observedPaddingCount = 0;
    std::int64_t           observedAlignCount   = 0;
    std::set<std::int64_t> seenStepIndexes;
    for (mlir::Operation& step : plan->getRegion(0).front())
    {
        const auto stepName = step.getName().getStringRef();
        if (stepName == "dsdl.align")
        {
            const auto bits      = step.getAttrOfType<mlir::IntegerAttr>("bits");
            const auto stepIndex = step.getAttrOfType<mlir::IntegerAttr>("step_index");
            if (!bits || !stepIndex)
            {
                return step.emitOpError("missing lowered align metadata");
            }
            if (bits.getInt() <= 1)
            {
                return step.emitOpError("unexpected no-op alignment in lowered plan");
            }
            if (!seenStepIndexes.insert(stepIndex.getInt()).second)
            {
                return step.emitOpError("duplicate lowered step_index");
            }
            ++observedStepCount;
            ++observedAlignCount;
            continue;
        }
        if (stepName != "dsdl.io")
        {
            return step.emitOpError("unsupported lowered plan operation");
        }

        const auto stepMinBits    = step.getAttrOfType<mlir::IntegerAttr>("min_bits");
        const auto stepMaxBits    = step.getAttrOfType<mlir::IntegerAttr>("max_bits");
        const auto loweredBits    = step.getAttrOfType<mlir::IntegerAttr>("lowered_bits");
        const auto stepIndex      = step.getAttrOfType<mlir::IntegerAttr>("step_index");
        const auto scalarCategory = step.getAttrOfType<mlir::StringAttr>("scalar_category");
        const auto arrayKind      = step.getAttrOfType<mlir::StringAttr>("array_kind");
        const auto kind           = step.getAttrOfType<mlir::StringAttr>("kind");
        const auto bitLength      = step.getAttrOfType<mlir::IntegerAttr>("bit_length");
        const auto alignmentBits  = step.getAttrOfType<mlir::IntegerAttr>("alignment_bits");
        if (!stepMinBits || !stepMaxBits || !loweredBits || !stepIndex || !scalarCategory || !arrayKind || !kind ||
            !bitLength || !alignmentBits)
        {
            return step.emitOpError("missing required lowered step metadata");
        }
        if (stepMinBits.getInt() < 0 || stepMaxBits.getInt() < stepMinBits.getInt() || loweredBits.getInt() < 0 ||
            loweredBits.getInt() != stepMaxBits.getInt() || bitLength.getInt() < 0 || alignmentBits.getInt() <= 0)
        {
            return step.emitOpError("invalid lowered step metadata values");
        }
        if (!isSupportedArrayKind(arrayKind.getValue()))
        {
            return step.emitOpError("unsupported array kind in lowered step metadata");
        }
        if (kind.getValue() != "field" && kind.getValue() != "padding")
        {
            return step.emitOpError("unsupported step kind in lowered step metadata");
        }
        if (!seenStepIndexes.insert(stepIndex.getInt()).second)
        {
            return step.emitOpError("duplicate lowered step_index");
        }
        ++observedStepCount;

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
                                           llvm::StringRef helperLabel) -> mlir::LogicalResult {
            const auto helper = step.getAttrOfType<mlir::StringAttr>(attrName);
            if (!helper || helper.getValue().empty())
            {
                return step.emitOpError("missing lowered " + helperLabel.str() + " helper attribute '" +
                                        attrName.str() + "'");
            }
            if (!module.lookupSymbol<mlir::func::FuncOp>(helper.getValue()))
            {
                return step.emitOpError("missing lowered " + helperLabel.str() +
                                        " helper symbol: " + helper.getValue().str());
            }
            return mlir::success();
        };

        const bool variableArray   = isVariableArrayKind(arrayKind.getValue());
        const auto arrayPrefixBits = step.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits");
        if (variableArray && (!arrayPrefixBits || arrayPrefixBits.getInt() <= 0 || arrayPrefixBits.getInt() > 64))
        {
            return step.emitOpError("missing valid array-length prefix width");
        }
        if (!isPadding && variableArray)
        {
            if (mlir::failed(
                    requireStepHelperSymbol("lowered_ser_array_length_prefix_helper", "array-length-prefix")) ||
                mlir::failed(
                    requireStepHelperSymbol("lowered_deser_array_length_prefix_helper", "array-length-prefix")) ||
                mlir::failed(requireStepHelperSymbol("lowered_array_length_validate_helper", "array-length-validate")))
            {
                return mlir::failure();
            }
        }

        if (!isPadding)
        {
            const auto category = scalarCategory.getValue();
            if (category == "unsigned" || category == "byte" || category == "utf8")
            {
                if (mlir::failed(requireStepHelperSymbol("lowered_ser_unsigned_helper", "scalar-unsigned")) ||
                    mlir::failed(requireStepHelperSymbol("lowered_deser_unsigned_helper", "scalar-unsigned")))
                {
                    return mlir::failure();
                }
            }
            else if (category == "signed")
            {
                if (mlir::failed(requireStepHelperSymbol("lowered_ser_signed_helper", "scalar-signed")) ||
                    mlir::failed(requireStepHelperSymbol("lowered_deser_signed_helper", "scalar-signed")))
                {
                    return mlir::failure();
                }
            }
            else if (category == "float")
            {
                if (mlir::failed(requireStepHelperSymbol("lowered_ser_float_helper", "scalar-float")) ||
                    mlir::failed(requireStepHelperSymbol("lowered_deser_float_helper", "scalar-float")))
                {
                    return mlir::failure();
                }
            }
        }

        if (!isPadding && scalarCategory.getValue() == "composite")
        {
            const auto compositeSealed = step.getAttrOfType<mlir::BoolAttr>("composite_sealed");
            if (compositeSealed && !compositeSealed.getValue())
            {
                if (!step.getAttrOfType<mlir::IntegerAttr>("composite_extent_bits"))
                {
                    return step.emitOpError("delimited composite missing composite_extent_bits metadata");
                }
                if (mlir::failed(requireStepHelperSymbol("lowered_delimiter_validate_helper", "delimiter-validate")))
                {
                    return mlir::failure();
                }
            }
        }
    }

    if (observedStepCount != stepCount.getInt() || observedFieldCount != fieldCount.getInt() ||
        observedPaddingCount != paddingCount.getInt() || observedAlignCount != alignCount.getInt())
    {
        return plan->emitOpError("lowered plan counts do not match plan body");
    }
    for (const auto stepIndex : seenStepIndexes)
    {
        if (stepIndex < 0 || stepIndex >= stepCount.getInt())
        {
            return plan->emitOpError("step_index out of lowered plan bounds");
        }
    }

    return mlir::success();
}

std::vector<PlanStep> collectPlanSteps(mlir::Operation* plan)
{
    std::vector<PlanStep> steps;
    if (plan->getNumRegions() == 0 || plan->getRegion(0).empty())
    {
        return steps;
    }
    for (mlir::Operation& op : plan->getRegion(0).front())
    {
        if (op.getName().getStringRef() == "dsdl.align")
        {
            const auto bits = op.getAttrOfType<mlir::IntegerAttr>("bits");
            PlanStep   alignStep;
            alignStep.kind = PlanStepKind::Align;
            alignStep.bits = bits ? nonNegative(bits.getInt()) : 1;
            steps.push_back(std::move(alignStep));
            continue;
        }
        if (op.getName().getStringRef() != "dsdl.io")
        {
            continue;
        }

        const std::int64_t bits     = ioStepBits(&op);
        const auto         kindAttr = op.getAttrOfType<mlir::StringAttr>("kind");
        const auto         nameAttr = op.getAttrOfType<mlir::StringAttr>("name");
        const auto         kind     = kindAttr ? kindAttr.getValue() : llvm::StringRef("field");
        if (kind == "padding")
        {
            steps.push_back(
                PlanStep{PlanStepKind::Padding,
                         bits,
                         nameAttr ? nameAttr.getValue().str() : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("c_name")
                             ? op.getAttrOfType<mlir::StringAttr>("c_name").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("scalar_category")
                             ? op.getAttrOfType<mlir::StringAttr>("scalar_category").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("cast_mode")
                             ? op.getAttrOfType<mlir::StringAttr>("cast_mode").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("array_kind")
                             ? op.getAttrOfType<mlir::StringAttr>("array_kind").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::IntegerAttr>("bit_length")
                             ? op.getAttrOfType<mlir::IntegerAttr>("bit_length").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("array_capacity")
                             ? op.getAttrOfType<mlir::IntegerAttr>("array_capacity").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("alignment_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("alignment_bits").getInt()
                             : 1,
                         op.getAttrOfType<mlir::IntegerAttr>("union_option_index")
                             ? op.getAttrOfType<mlir::IntegerAttr>("union_option_index").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("union_tag_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("union_tag_bits").getInt()
                             : 0,
                         op.getAttrOfType<mlir::StringAttr>("composite_c_type_name")
                             ? op.getAttrOfType<mlir::StringAttr>("composite_c_type_name").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_unsigned_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_unsigned_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_signed_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_signed_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_float_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_float_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_array_length_prefix_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_array_length_prefix_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_array_length_prefix_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_array_length_prefix_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_array_length_validate_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_array_length_validate_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_delimiter_validate_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_delimiter_validate_helper").getValue().str()
                             : std::string{}});
            if (auto sealed = op.getAttrOfType<mlir::BoolAttr>("composite_sealed"))
            {
                steps.back().compositeSealed = sealed.getValue();
            }
            if (auto extent = op.getAttrOfType<mlir::IntegerAttr>("composite_extent_bits"))
            {
                steps.back().compositeExtentBits = nonNegative(extent.getInt());
            }
        }
        else
        {
            steps.push_back(
                PlanStep{PlanStepKind::Field,
                         bits,
                         nameAttr ? nameAttr.getValue().str() : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("c_name")
                             ? op.getAttrOfType<mlir::StringAttr>("c_name").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("scalar_category")
                             ? op.getAttrOfType<mlir::StringAttr>("scalar_category").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("cast_mode")
                             ? op.getAttrOfType<mlir::StringAttr>("cast_mode").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("array_kind")
                             ? op.getAttrOfType<mlir::StringAttr>("array_kind").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::IntegerAttr>("bit_length")
                             ? op.getAttrOfType<mlir::IntegerAttr>("bit_length").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("array_capacity")
                             ? op.getAttrOfType<mlir::IntegerAttr>("array_capacity").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("alignment_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("alignment_bits").getInt()
                             : 1,
                         op.getAttrOfType<mlir::IntegerAttr>("union_option_index")
                             ? op.getAttrOfType<mlir::IntegerAttr>("union_option_index").getInt()
                             : 0,
                         op.getAttrOfType<mlir::IntegerAttr>("union_tag_bits")
                             ? op.getAttrOfType<mlir::IntegerAttr>("union_tag_bits").getInt()
                             : 0,
                         op.getAttrOfType<mlir::StringAttr>("composite_c_type_name")
                             ? op.getAttrOfType<mlir::StringAttr>("composite_c_type_name").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_unsigned_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_unsigned_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_signed_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_signed_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_float_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_float_helper").getValue().str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_ser_array_length_prefix_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_ser_array_length_prefix_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_deser_array_length_prefix_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_deser_array_length_prefix_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_array_length_validate_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_array_length_validate_helper")
                                   .getValue()
                                   .str()
                             : std::string{},
                         op.getAttrOfType<mlir::StringAttr>("lowered_delimiter_validate_helper")
                             ? op.getAttrOfType<mlir::StringAttr>("lowered_delimiter_validate_helper").getValue().str()
                             : std::string{}});
            if (auto sealed = op.getAttrOfType<mlir::BoolAttr>("composite_sealed"))
            {
                steps.back().compositeSealed = sealed.getValue();
            }
            if (auto extent = op.getAttrOfType<mlir::IntegerAttr>("composite_extent_bits"))
            {
                steps.back().compositeExtentBits = nonNegative(extent.getInt());
            }
        }
    }
    return steps;
}

std::string renderGenericSerializeFunction(llvm::StringRef              functionName,
                                           llvm::StringRef              cTypeName,
                                           llvm::StringRef              cSerializeSymbol,
                                           llvm::StringRef              fullName,
                                           llvm::StringRef              sectionName,
                                           std::int64_t                 minBits,
                                           std::int64_t                 maxBits,
                                           const std::vector<PlanStep>& steps,
                                           llvm::StringRef              capacityCheckSymbol)
{
    const std::string  functionNameText     = functionName.str();
    const std::string  cTypeNameText        = cTypeName.str();
    const std::string  cSerializeSymbolText = cSerializeSymbol.str();
    const std::string  fullNameText         = fullName.str();
    const std::string  sectionNameText      = sectionName.str();
    std::ostringstream out;
    if (cTypeNameText.empty())
    {
        out << "int8_t " << functionNameText
            << "(const void* obj, uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    else
    {
        out << "int8_t " << functionNameText << "(const " << cTypeNameText
            << "* const obj, uint8_t* buffer, size_t* const inout_buffer_size_bytes)\n";
    }
    out << "{\n";
    out << "  // IR section: " << fullNameText;
    if (!sectionNameText.empty())
    {
        out << " (" << sectionNameText << ")";
    }
    out << ", min_bits=" << minBits << ", max_bits=" << maxBits << ".\n";
    if (!cSerializeSymbolText.empty())
    {
        out << "  // Public C API symbol: " << cSerializeSymbolText << "\n";
    }
    out << "  // Generic bitstream mapping: non-padding fields are packed in\n";
    out << "  // declaration order from/to object memory as contiguous bits.\n";
    out << "  if ((obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == "
           "NULL)) {\n";
    out << "    return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;\n";
    out << "  }\n";
    out << "  const uint8_t* const obj_bytes = (const uint8_t*)obj;\n";
    out << "  const size_t capacity_bits = (*inout_buffer_size_bytes) * 8U;\n";
    out << "  const int8_t _err_capacity = " << capacityCheckSymbol.str() << "((int64_t)capacity_bits);\n";
    out << "  if (_err_capacity < 0) {\n";
    out << "    return _err_capacity;\n";
    out << "  }\n";
    out << "  size_t offset_bits = 0U;\n";
    out << "  size_t obj_offset_bits = 0U;\n";
    out << "  (void)obj_bytes;\n";
    out << "  (void)obj_offset_bits;\n";
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        const auto& step = steps[index];
        if (step.kind == PlanStepKind::Align)
        {
            if (step.bits > 1)
            {
                out << "  offset_bits = ((offset_bits + " << (step.bits - 1) << "U) / " << step.bits << "U) * "
                    << step.bits << "U;\n";
            }
            continue;
        }

        out << "  {\n";
        out << "    const size_t bits_" << index << " = " << nonNegative(step.bits) << "U;\n";
        if (!step.name.empty())
        {
            out << "    /* " << step.name << " */\n";
        }
        out << "    if (bits_" << index << " > 0U) {\n";
        out << "      if (offset_bits + bits_" << index << " > capacity_bits) {\n";
        out << "        return "
               "-(int8_t)DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL;\n";
        out << "      }\n";
        if (step.kind == PlanStepKind::Padding)
        {
            out << "      for (size_t bit_" << index << " = 0U; bit_" << index << " < bits_" << index << "; ++bit_"
                << index << ") {\n";
            out << "        dsdl_runtime_set_bit(buffer, *inout_buffer_size_bytes, "
                   "offset_bits + bit_"
                << index << ", false);\n";
            out << "      }\n";
        }
        else
        {
            out << "      dsdl_runtime_copy_bits(buffer, offset_bits, bits_" << index
                << ", obj_bytes, obj_offset_bits);\n";
            out << "      obj_offset_bits += bits_" << index << ";\n";
        }
        out << "      offset_bits += bits_" << index << ";\n";
        out << "    }\n";
        out << "  }\n";
    }
    out << "  *inout_buffer_size_bytes = (offset_bits + 7U) / 8U;\n";
    out << "  return (int8_t)DSDL_RUNTIME_SUCCESS;\n";
    out << "}\n";
    return out.str();
}

std::string renderGenericDeserializeFunction(llvm::StringRef              functionName,
                                             llvm::StringRef              cTypeName,
                                             llvm::StringRef              cDeserializeSymbol,
                                             llvm::StringRef              fullName,
                                             llvm::StringRef              sectionName,
                                             std::int64_t                 minBits,
                                             std::int64_t                 maxBits,
                                             const std::vector<PlanStep>& steps)
{
    const std::string  functionNameText       = functionName.str();
    const std::string  cTypeNameText          = cTypeName.str();
    const std::string  cDeserializeSymbolText = cDeserializeSymbol.str();
    const std::string  fullNameText           = fullName.str();
    const std::string  sectionNameText        = sectionName.str();
    std::ostringstream out;
    if (cTypeNameText.empty())
    {
        out << "int8_t " << functionNameText
            << "(void* out_obj, const uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    else
    {
        out << "int8_t " << functionNameText << "(" << cTypeNameText
            << "* const out_obj, const uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    out << "{\n";
    out << "  // IR section: " << fullNameText;
    if (!sectionNameText.empty())
    {
        out << " (" << sectionNameText << ")";
    }
    out << ", min_bits=" << minBits << ", max_bits=" << maxBits << ".\n";
    if (!cDeserializeSymbolText.empty())
    {
        out << "  // Public C API symbol: " << cDeserializeSymbolText << "\n";
    }
    out << "  // Generic bitstream mapping: non-padding fields are unpacked in\n";
    out << "  // declaration order into object memory as contiguous bits.\n";
    out << "  if ((out_obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes "
           "== NULL)) {\n";
    out << "    return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;\n";
    out << "  }\n";
    out << "  uint8_t* const obj_bytes = (uint8_t*)out_obj;\n";
    out << "  const size_t capacity_bits = (*inout_buffer_size_bytes) * 8U;\n";
    out << "  const size_t required_bits = " << nonNegative(maxBits) << "U;\n";
    out << "  const size_t obj_capacity_bytes = (required_bits + 7U) / 8U;\n";
    out << "  size_t offset_bits = 0U;\n";
    out << "  size_t obj_offset_bits = 0U;\n";
    out << "  (void)obj_bytes;\n";
    out << "  (void)obj_capacity_bytes;\n";
    out << "  (void)obj_offset_bits;\n";
    for (std::size_t index = 0; index < steps.size(); ++index)
    {
        const auto& step = steps[index];
        if (step.kind == PlanStepKind::Align)
        {
            if (step.bits > 1)
            {
                out << "  offset_bits = ((offset_bits + " << (step.bits - 1) << "U) / " << step.bits << "U) * "
                    << step.bits << "U;\n";
            }
            continue;
        }

        out << "  {\n";
        out << "    const size_t bits_" << index << " = " << nonNegative(step.bits) << "U;\n";
        if (!step.name.empty())
        {
            out << "    /* " << step.name << " */\n";
        }
        out << "    if (bits_" << index << " > 0U) {\n";
        if (step.kind == PlanStepKind::Field)
        {
            out << "      const size_t available_bits_" << index
                << " = (offset_bits < capacity_bits) ? (capacity_bits - offset_bits) : "
                   "0U;\n";
            out << "      const size_t copy_bits_" << index << " = (available_bits_" << index << " < bits_" << index
                << ") ? available_bits_" << index << " : bits_" << index << ";\n";
            out << "      if (copy_bits_" << index << " > 0U) {\n";
            out << "        dsdl_runtime_copy_bits(obj_bytes, obj_offset_bits, "
                   "copy_bits_"
                << index << ", buffer, offset_bits);\n";
            out << "      }\n";
            out << "      if (copy_bits_" << index << " < bits_" << index << ") {\n";
            out << "        const size_t zero_bits_" << index << " = bits_" << index << " - copy_bits_" << index
                << ";\n";
            out << "        for (size_t bit_" << index << " = 0U; bit_" << index << " < zero_bits_" << index
                << "; ++bit_" << index << ") {\n";
            out << "          dsdl_runtime_set_bit(obj_bytes, obj_capacity_bytes, "
                   "obj_offset_bits + copy_bits_"
                << index << " + bit_" << index << ", false);\n";
            out << "        }\n";
            out << "      }\n";
            out << "      obj_offset_bits += bits_" << index << ";\n";
        }
        out << "      offset_bits += bits_" << index << ";\n";
        out << "    }\n";
        out << "  }\n";
    }
    out << "  const size_t consumed_bits = (offset_bits < capacity_bits) "
           "? offset_bits : capacity_bits;\n";
    out << "  *inout_buffer_size_bytes = consumed_bits / 8U;\n";
    out << "  return (int8_t)DSDL_RUNTIME_SUCCESS;\n";
    out << "}\n";
    return out.str();
}

void emitLine(std::ostringstream& out, const int indent, const std::string& line)
{
    for (int i = 0; i < indent; ++i)
    {
        out << "  ";
    }
    out << line << "\n";
}

bool supportsTypedFieldStep(const PlanStep& step)
{
    if (step.kind != PlanStepKind::Field)
    {
        return true;
    }
    if (step.cName.empty())
    {
        return false;
    }

    if (!isSupportedArrayKind(step.arrayKind))
    {
        return false;
    }
    if (step.arrayKind != "none")
    {
        if (step.arrayCapacity < 0)
        {
            return false;
        }
        if (isVariableArrayKind(step.arrayKind) && (step.arrayLengthPrefixBits <= 0 || step.arrayLengthPrefixBits > 64))
        {
            return false;
        }
    }

    if (step.scalarCategory == "void")
    {
        return false;
    }

    if (step.scalarCategory == "composite")
    {
        return !step.compositeCTypeName.empty();
    }
    if (step.scalarCategory == "bool")
    {
        return step.bitLength == 1;
    }
    if (step.scalarCategory == "byte" || step.scalarCategory == "utf8")
    {
        return step.bitLength == 8;
    }
    if (step.scalarCategory == "unsigned" || step.scalarCategory == "signed")
    {
        return step.bitLength >= 1 && step.bitLength <= 64;
    }
    if (step.scalarCategory == "float")
    {
        return step.bitLength == 16 || step.bitLength == 32 || step.bitLength == 64;
    }
    return false;
}

bool supportsTypedLowering(const std::vector<PlanStep>& steps, const bool isUnion, const std::int64_t unionTagBits)
{
    if (isUnion && (unionTagBits <= 0 || unionTagBits > 64))
    {
        return false;
    }

    std::set<std::int64_t> unionOptions;
    for (const auto& step : steps)
    {
        if (!supportsTypedFieldStep(step))
        {
            return false;
        }
        if (step.kind == PlanStepKind::Align && step.bits <= 0)
        {
            return false;
        }
        if (isUnion && step.kind == PlanStepKind::Field)
        {
            unionOptions.insert(step.unionOptionIndex);
        }
    }
    if (isUnion && unionOptions.empty())
    {
        return false;
    }
    return true;
}

void emitDeserializeAlign(std::ostringstream& out, const int indent, const std::int64_t alignmentBits)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    emitLine(out,
             indent,
             "offset_bits = ((offset_bits + " + std::to_string(alignmentBits - 1) + "U) / " +
                 std::to_string(alignmentBits) + "U) * " + std::to_string(alignmentBits) + "U;");
}

void emitSerializeAlign(std::ostringstream& out,
                        const int           indent,
                        const std::int64_t  alignmentBits,
                        const std::string&  tag)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    const std::string alignedName = "_aligned_offset_bits_" + tag;
    const std::string padBitName  = "_pad_bit_" + tag;
    const std::string errName     = "_err_align_" + tag;
    emitLine(out,
             indent,
             "const size_t " + alignedName + " = ((offset_bits + " + std::to_string(alignmentBits - 1) + "U) / " +
                 std::to_string(alignmentBits) + "U) * " + std::to_string(alignmentBits) + "U;");
    emitLine(out,
             indent,
             "for (size_t " + padBitName + " = offset_bits; " + padBitName + " < " + alignedName + "; ++" + padBitName +
                 ") {");
    emitLine(out,
             indent + 1,
             "const int8_t " + errName + " = dsdl_runtime_set_bit(buffer, capacity_bytes, " + padBitName + ", false);");
    emitLine(out, indent + 1, "if (" + errName + " < 0) {");
    emitLine(out, indent + 2, "return " + errName + ";");
    emitLine(out, indent + 1, "}");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offset_bits = " + alignedName + ";");
}

std::string unsignedGetterForBits(const std::int64_t bits)
{
    if (bits <= 8)
    {
        return "dsdl_runtime_get_u8";
    }
    if (bits <= 16)
    {
        return "dsdl_runtime_get_u16";
    }
    if (bits <= 32)
    {
        return "dsdl_runtime_get_u32";
    }
    return "dsdl_runtime_get_u64";
}

std::string signedGetterForBits(const std::int64_t bits)
{
    if (bits <= 8)
    {
        return "dsdl_runtime_get_i8";
    }
    if (bits <= 16)
    {
        return "dsdl_runtime_get_i16";
    }
    if (bits <= 32)
    {
        return "dsdl_runtime_get_i32";
    }
    return "dsdl_runtime_get_i64";
}

void emitSerializePadding(std::ostringstream& out, const std::size_t index, const std::int64_t bits, const int indent)
{
    emitLine(out,
             indent,
             "for (size_t bit_" + std::to_string(index) + " = 0U; bit_" + std::to_string(index) + " < " +
                 std::to_string(nonNegative(bits)) + "U; ++bit_" + std::to_string(index) + ") {");
    emitLine(out,
             indent + 1,
             "const int8_t _err_pad_" + std::to_string(index) +
                 " = dsdl_runtime_set_bit(buffer, capacity_bytes, offset_bits + "
                 "bit_" +
                 std::to_string(index) + ", false);");
    emitLine(out, indent + 1, "if (_err_pad_" + std::to_string(index) + " < 0) {");
    emitLine(out, indent + 2, "return _err_pad_" + std::to_string(index) + ";");
    emitLine(out, indent + 1, "}");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(bits)) + "U;");
}

bool emitSerializeField(std::ostringstream& out,
                        const PlanStep&     step,
                        const std::string&  expr,
                        const std::size_t   index,
                        const int           indent);
bool emitDeserializeField(std::ostringstream& out,
                          const PlanStep&     step,
                          const std::string&  expr,
                          const std::size_t   index,
                          const int           indent);

bool emitSerializeArrayField(std::ostringstream& out,
                             const PlanStep&     step,
                             const std::string&  expr,
                             const std::size_t   index,
                             const int           indent)
{
    const bool variable     = isVariableArrayKind(step.arrayKind);
    const auto capacityExpr = std::to_string(nonNegative(step.arrayCapacity)) + "U";

    if (variable)
    {
        if (step.serArrayLengthPrefixHelper.empty())
        {
            return false;
        }
        if (!step.arrayLengthValidateHelper.empty())
        {
            emitLine(out,
                     indent,
                     "const int8_t _err_lenchk_" + std::to_string(index) + " = " + step.arrayLengthValidateHelper +
                         "((int64_t)(" + expr + ".count));");
            emitLine(out, indent, "if (_err_lenchk_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_lenchk_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
        }
        else
        {
            emitLine(out, indent, "if (" + expr + ".count > " + capacityExpr + ") {");
            emitLine(out, indent + 1, "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH;");
            emitLine(out, indent, "}");
        }
        emitLine(out,
                 indent,
                 "const uint64_t _wire_len_" + std::to_string(index) + " = (uint64_t)" +
                     step.serArrayLengthPrefixHelper + "((int64_t)(" + expr + ".count));");
        emitLine(out,
                 indent,
                 "const int8_t _err_len_" + std::to_string(index) +
                     " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, "
                     "_wire_len_" +
                     std::to_string(index) + ", (uint8_t)" + std::to_string(nonNegative(step.arrayLengthPrefixBits)) +
                     "U);");
        emitLine(out, indent, "if (_err_len_" + std::to_string(index) + " < 0) {");
        emitLine(out, indent + 1, "return _err_len_" + std::to_string(index) + ";");
        emitLine(out, indent, "}");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.arrayLengthPrefixBits)) + "U;");
    }

    const auto countExpr = variable ? (expr + ".count") : capacityExpr;
    if (step.scalarCategory == "bool")
    {
        const auto sourceExpr = variable ? ("&" + expr + ".bitpacked[0]") : ("&" + expr + "[0]");
        emitLine(out,
                 indent,
                 "dsdl_runtime_copy_bits(&buffer[0], offset_bits, " + countExpr + ", " + sourceExpr + ", 0U);");
        emitLine(out, indent, "offset_bits += " + countExpr + ";");
        return true;
    }

    const auto loopIndex    = "_i_" + std::to_string(index);
    const auto accessPrefix = variable ? (expr + ".elements") : expr;
    emitLine(out,
             indent,
             "for (size_t " + loopIndex + " = 0U; " + loopIndex + " < " + countExpr + "; ++" + loopIndex + ") {");
    auto elementStep                  = step;
    elementStep.arrayKind             = "none";
    elementStep.arrayCapacity         = 0;
    elementStep.arrayLengthPrefixBits = 0;
    if (!emitSerializeField(out, elementStep, accessPrefix + "[" + loopIndex + "]", index, indent + 1))
    {
        return false;
    }
    emitLine(out, indent, "}");
    return true;
}

bool emitDeserializeArrayField(std::ostringstream& out,
                               const PlanStep&     step,
                               const std::string&  expr,
                               const std::size_t   index,
                               const int           indent)
{
    const bool variable     = isVariableArrayKind(step.arrayKind);
    const auto capacityExpr = std::to_string(nonNegative(step.arrayCapacity)) + "U";

    if (variable)
    {
        if (step.deserArrayLengthPrefixHelper.empty())
        {
            return false;
        }
        emitLine(out,
                 indent,
                 "const uint64_t _wire_len_" + std::to_string(index) + " = (uint64_t)" +
                     unsignedGetterForBits(step.arrayLengthPrefixBits) +
                     "(buffer, capacity_bytes, offset_bits, (uint8_t)" +
                     std::to_string(nonNegative(step.arrayLengthPrefixBits)) + "U);");
        emitLine(out,
                 indent,
                 expr + ".count = (size_t)" + step.deserArrayLengthPrefixHelper + "((int64_t)_wire_len_" +
                     std::to_string(index) + ");");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.arrayLengthPrefixBits)) + "U;");
        if (!step.arrayLengthValidateHelper.empty())
        {
            emitLine(out,
                     indent,
                     "const int8_t _err_lenchk_" + std::to_string(index) + " = " + step.arrayLengthValidateHelper +
                         "((int64_t)(" + expr + ".count));");
            emitLine(out, indent, "if (_err_lenchk_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_lenchk_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
        }
        else
        {
            emitLine(out, indent, "if (" + expr + ".count > " + capacityExpr + ") {");
            emitLine(out, indent + 1, "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH;");
            emitLine(out, indent, "}");
        }
    }

    const auto countExpr = variable ? (expr + ".count") : capacityExpr;
    if (step.scalarCategory == "bool")
    {
        const auto targetExpr = variable ? ("&" + expr + ".bitpacked[0]") : ("&" + expr + "[0]");
        emitLine(out,
                 indent,
                 "dsdl_runtime_get_bits(" + targetExpr + ", &buffer[0], capacity_bytes, offset_bits, " + countExpr +
                     ");");
        emitLine(out, indent, "offset_bits += " + countExpr + ";");
        return true;
    }

    const auto loopIndex    = "_i_" + std::to_string(index);
    const auto accessPrefix = variable ? (expr + ".elements") : expr;
    emitLine(out,
             indent,
             "for (size_t " + loopIndex + " = 0U; " + loopIndex + " < " + countExpr + "; ++" + loopIndex + ") {");
    auto elementStep                  = step;
    elementStep.arrayKind             = "none";
    elementStep.arrayCapacity         = 0;
    elementStep.arrayLengthPrefixBits = 0;
    if (!emitDeserializeField(out, elementStep, accessPrefix + "[" + loopIndex + "]", index, indent + 1))
    {
        return false;
    }
    emitLine(out, indent, "}");
    return true;
}

bool emitSerializeField(std::ostringstream& out,
                        const PlanStep&     step,
                        const std::string&  expr,
                        const std::size_t   index,
                        const int           indent)
{
    if (step.arrayKind != "none")
    {
        return emitSerializeArrayField(out, step, expr, index, indent);
    }

    if (step.scalarCategory == "composite")
    {
        if (!step.compositeSealed)
        {
            emitLine(out, indent, "const size_t _delim_start_bytes_" + std::to_string(index) + " = offset_bits / 8U;");
            emitLine(out, indent, "offset_bits += 32U;");
            emitLine(out,
                     indent,
                     "const size_t _remaining_bytes_" + std::to_string(index) +
                         " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, "
                         "capacity_bytes);");
            emitLine(out,
                     indent,
                     "size_t _size_bytes_" + std::to_string(index) +
                         " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, "
                         "capacity_bytes);");
            emitLine(out,
                     indent,
                     "const int8_t _err_" + std::to_string(index) + " = " + step.compositeCTypeName + "__serialize_(&" +
                         expr + ", &buffer[offset_bits / 8U], &_size_bytes_" + std::to_string(index) + ");");
            emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            if (step.delimiterValidateHelper.empty())
            {
                return false;
            }
            emitLine(out,
                     indent,
                     "const int8_t _delim_chk_" + std::to_string(index) + " = " + step.delimiterValidateHelper +
                         "((int64_t)_size_bytes_" + std::to_string(index) + ", (int64_t)_remaining_bytes_" +
                         std::to_string(index) + ");");
            emitLine(out, indent, "if (_delim_chk_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _delim_chk_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            emitLine(out, indent, "offset_bits += _size_bytes_" + std::to_string(index) + " * 8U;");
            emitLine(out,
                     indent,
                     "const int8_t _hdr_err_" + std::to_string(index) +
                         " = dsdl_runtime_set_uxx(buffer, capacity_bytes, "
                         "_delim_start_bytes_" +
                         std::to_string(index) + " * 8U, (uint64_t)_size_bytes_" + std::to_string(index) + ", 32U);");
            emitLine(out, indent, "if (_hdr_err_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _hdr_err_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
        }
        else
        {
            emitLine(out,
                     indent,
                     "size_t _size_bytes_" + std::to_string(index) +
                         " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, "
                         "capacity_bytes);");
            emitLine(out,
                     indent,
                     "const int8_t _err_" + std::to_string(index) + " = " + step.compositeCTypeName + "__serialize_(&" +
                         expr + ", &buffer[offset_bits / 8U], &_size_bytes_" + std::to_string(index) + ");");
            emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            emitLine(out, indent, "offset_bits += _size_bytes_" + std::to_string(index) + " * 8U;");
        }
        return true;
    }

    if (step.scalarCategory == "bool")
    {
        emitLine(out,
                 indent,
                 "const int8_t _err_" + std::to_string(index) +
                     " = dsdl_runtime_set_bit(buffer, capacity_bytes, offset_bits, " + expr + ");");
        emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
        emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
        emitLine(out, indent, "}");
        emitLine(out, indent, "offset_bits += 1U;");
        return true;
    }

    if (step.scalarCategory == "byte" || step.scalarCategory == "utf8" || step.scalarCategory == "unsigned")
    {
        std::string valueExpr = "(uint64_t)(" + expr + ")";
        if (step.serUnsignedHelper.empty())
        {
            return false;
        }
        const auto normName = "_norm_" + std::to_string(index);
        emitLine(out,
                 indent,
                 "const uint64_t " + normName + " = (uint64_t)" + step.serUnsignedHelper + "((int64_t)(" + valueExpr +
                     "));");
        valueExpr = normName;
        emitLine(out,
                 indent,
                 "const int8_t _err_" + std::to_string(index) +
                     " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, " + valueExpr + ", (uint8_t)" +
                     std::to_string(nonNegative(step.bitLength)) + "U);");
        emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
        emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
        emitLine(out, indent, "}");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    if (step.scalarCategory == "signed")
    {
        std::string valueExpr = "(int64_t)(" + expr + ")";
        if (step.serSignedHelper.empty())
        {
            return false;
        }
        const auto normName = "_norms_" + std::to_string(index);
        emitLine(out,
                 indent,
                 "const int64_t " + normName + " = (int64_t)" + step.serSignedHelper + "((int64_t)(" + valueExpr +
                     "));");
        valueExpr = normName;
        emitLine(out,
                 indent,
                 "const int8_t _err_" + std::to_string(index) +
                     " = dsdl_runtime_set_ixx(buffer, capacity_bytes, offset_bits, " + valueExpr + ", (uint8_t)" +
                     std::to_string(nonNegative(step.bitLength)) + "U);");
        emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
        emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
        emitLine(out, indent, "}");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    if (step.scalarCategory == "float")
    {
        std::string setter;
        std::string castType;
        if (step.bitLength == 16 || step.bitLength == 32)
        {
            castType = "float";
            setter   = (step.bitLength == 16) ? "dsdl_runtime_set_f16" : "dsdl_runtime_set_f32";
        }
        else if (step.bitLength == 64)
        {
            castType = "double";
            setter   = "dsdl_runtime_set_f64";
        }
        else
        {
            return false;
        }
        if (step.serFloatHelper.empty())
        {
            return false;
        }
        const auto normName = "_normf_" + std::to_string(index);
        emitLine(out, indent, "const double " + normName + " = " + step.serFloatHelper + "((double)(" + expr + "));");
        std::string valueExpr = "(" + castType + ")(" + normName + ")";
        emitLine(out,
                 indent,
                 "const int8_t _err_" + std::to_string(index) + " = " + setter +
                     "(buffer, capacity_bytes, offset_bits, " + valueExpr + ");");
        emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
        emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
        emitLine(out, indent, "}");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    return false;
}

bool emitDeserializeField(std::ostringstream& out,
                          const PlanStep&     step,
                          const std::string&  expr,
                          const std::size_t   index,
                          const int           indent)
{
    if (step.arrayKind != "none")
    {
        return emitDeserializeArrayField(out, step, expr, index, indent);
    }

    if (step.scalarCategory == "composite")
    {
        if (!step.compositeSealed)
        {
            emitLine(out,
                     indent,
                     "size_t _size_bytes_" + std::to_string(index) +
                         " = (size_t)dsdl_runtime_get_u32(buffer, capacity_bytes, "
                         "offset_bits, 32U);");
            emitLine(out, indent, "offset_bits += 32U;");
            emitLine(out,
                     indent,
                     "const size_t _remaining_bytes_" + std::to_string(index) +
                         " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, "
                         "capacity_bytes);");
            if (step.delimiterValidateHelper.empty())
            {
                return false;
            }
            emitLine(out,
                     indent,
                     "const int8_t _delim_chk_" + std::to_string(index) + " = " + step.delimiterValidateHelper +
                         "((int64_t)_size_bytes_" + std::to_string(index) + ", (int64_t)_remaining_bytes_" +
                         std::to_string(index) + ");");
            emitLine(out, indent, "if (_delim_chk_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _delim_chk_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            emitLine(out,
                     indent,
                     "const int8_t _err_" + std::to_string(index) + " = " + step.compositeCTypeName +
                         "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &_size_bytes_" +
                         std::to_string(index) + ");");
            emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            emitLine(out, indent, "offset_bits += _size_bytes_" + std::to_string(index) + " * 8U;");
        }
        else
        {
            emitLine(out,
                     indent,
                     "size_t _size_bytes_" + std::to_string(index) +
                         " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, "
                         "capacity_bytes);");
            emitLine(out,
                     indent,
                     "const int8_t _err_" + std::to_string(index) + " = " + step.compositeCTypeName +
                         "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &_size_bytes_" +
                         std::to_string(index) + ");");
            emitLine(out, indent, "if (_err_" + std::to_string(index) + " < 0) {");
            emitLine(out, indent + 1, "return _err_" + std::to_string(index) + ";");
            emitLine(out, indent, "}");
            emitLine(out, indent, "offset_bits += _size_bytes_" + std::to_string(index) + " * 8U;");
        }
        return true;
    }

    if (step.scalarCategory == "bool")
    {
        emitLine(out, indent, expr + " = dsdl_runtime_get_bit(buffer, capacity_bytes, offset_bits);");
        emitLine(out, indent, "offset_bits += 1U;");
        return true;
    }

    if (step.scalarCategory == "byte" || step.scalarCategory == "utf8" || step.scalarCategory == "unsigned")
    {
        const auto rawName = "_raw_" + std::to_string(index);
        emitLine(out,
                 indent,
                 "const uint64_t " + rawName + " = (uint64_t)" + unsignedGetterForBits(step.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, (uint8_t)" + std::to_string(nonNegative(step.bitLength)) +
                     "U);");
        if (step.deserUnsignedHelper.empty())
        {
            return false;
        }
        emitLine(out, indent, expr + " = (uint64_t)" + step.deserUnsignedHelper + "((int64_t)" + rawName + ");");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    if (step.scalarCategory == "signed")
    {
        const auto rawName = "_raws_" + std::to_string(index);
        emitLine(out,
                 indent,
                 "const int64_t " + rawName + " = (int64_t)" + signedGetterForBits(step.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, (uint8_t)" + std::to_string(nonNegative(step.bitLength)) +
                     "U);");
        if (step.deserSignedHelper.empty())
        {
            return false;
        }
        emitLine(out, indent, expr + " = (int64_t)" + step.deserSignedHelper + "((int64_t)" + rawName + ");");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    if (step.scalarCategory == "float")
    {
        std::string getter;
        if (step.bitLength == 16)
        {
            getter = "dsdl_runtime_get_f16";
        }
        else if (step.bitLength == 32)
        {
            getter = "dsdl_runtime_get_f32";
        }
        else if (step.bitLength == 64)
        {
            getter = "dsdl_runtime_get_f64";
        }
        else
        {
            return false;
        }
        std::string castType = (step.bitLength == 64) ? "double" : "float";
        const auto  rawName  = "_rawf_" + std::to_string(index);
        emitLine(out,
                 indent,
                 "const double " + rawName + " = (double)" + getter + "(buffer, capacity_bytes, offset_bits);");
        if (step.deserFloatHelper.empty())
        {
            return false;
        }
        emitLine(out, indent, expr + " = (" + castType + ")(" + step.deserFloatHelper + "(" + rawName + "));");
        emitLine(out, indent, "offset_bits += " + std::to_string(nonNegative(step.bitLength)) + "U;");
        return true;
    }

    return false;
}

std::string renderTypedSerializeFunction(llvm::StringRef              functionName,
                                         llvm::StringRef              cTypeName,
                                         llvm::StringRef              cSerializeSymbol,
                                         llvm::StringRef              fullName,
                                         llvm::StringRef              sectionName,
                                         std::int64_t                 minBits,
                                         std::int64_t                 maxBits,
                                         const std::vector<PlanStep>& steps,
                                         const bool                   isUnion,
                                         const std::int64_t           unionTagBits,
                                         llvm::StringRef              capacityCheckSymbol,
                                         llvm::StringRef              unionTagValidateSymbol,
                                         llvm::StringRef              unionTagSerializeSymbol)
{
    const std::string  functionNameText     = functionName.str();
    const std::string  cTypeNameText        = cTypeName.str();
    const std::string  cSerializeSymbolText = cSerializeSymbol.str();
    const std::string  fullNameText         = fullName.str();
    const std::string  sectionNameText      = sectionName.str();
    std::ostringstream out;
    if (cTypeNameText.empty())
    {
        out << "int8_t " << functionNameText
            << "(const void* obj, uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    else
    {
        out << "int8_t " << functionNameText << "(const " << cTypeNameText
            << "* const obj, uint8_t* buffer, size_t* const inout_buffer_size_bytes)\n";
    }
    out << "{\n";
    emitLine(out,
             1,
             "// IR section: " + fullNameText +
                 (sectionNameText.empty() ? std::string{} : (" (" + sectionNameText + ")")) +
                 ", min_bits=" + std::to_string(minBits) + ", max_bits=" + std::to_string(maxBits) + ".");
    if (!cSerializeSymbolText.empty())
    {
        emitLine(out, 1, "// Public C API symbol: " + cSerializeSymbolText);
    }
    emitLine(out, 1, "// Typed IR lowering path.");
    emitLine(out,
             1,
             "if ((obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == "
             "NULL)) {");
    emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "const int8_t _err_capacity = " + capacityCheckSymbol.str() + "((int64_t)(capacity_bytes * 8U));");
    emitLine(out, 1, "if (_err_capacity < 0) {");
    emitLine(out, 2, "return _err_capacity;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "size_t offset_bits = 0U;");

    if (isUnion)
    {
        const auto tagBits = nonNegative(unionTagBits);
        emitLine(out,
                 1,
                 "const uint64_t _tag_value = (uint64_t)" + unionTagSerializeSymbol.str() + "((int64_t)(obj->_tag_));");
        emitLine(out, 1, "const int8_t _err_union_tag = " + unionTagValidateSymbol.str() + "((int64_t)_tag_value);");
        emitLine(out, 1, "if (_err_union_tag < 0) {");
        emitLine(out, 2, "return _err_union_tag;");
        emitLine(out, 1, "}");
        emitLine(out,
                 1,
                 "const int8_t _err_tag_ = dsdl_runtime_set_uxx(buffer, "
                 "capacity_bytes, offset_bits, _tag_value, (uint8_t)" +
                     std::to_string(tagBits) + "U);");
        emitLine(out, 1, "if (_err_tag_ < 0) {");
        emitLine(out, 2, "return _err_tag_;");
        emitLine(out, 1, "}");
        emitLine(out, 1, "offset_bits += " + std::to_string(tagBits) + "U;");

        std::vector<const PlanStep*> unionFields;
        for (const auto& step : steps)
        {
            if (step.kind == PlanStepKind::Field)
            {
                unionFields.push_back(&step);
            }
        }
        std::sort(unionFields.begin(), unionFields.end(), [](const PlanStep* lhs, const PlanStep* rhs) {
            return lhs->unionOptionIndex < rhs->unionOptionIndex;
        });

        bool first = true;
        for (std::size_t i = 0; i < unionFields.size(); ++i)
        {
            const auto& step = *unionFields[i];
            emitLine(out,
                     1,
                     std::string(first ? "if" : "else if") +
                         " (obj->_tag_ == " + std::to_string(nonNegative(step.unionOptionIndex)) + "U) {");
            first = false;
            emitSerializeAlign(out, 2, step.alignmentBits, "u" + std::to_string(i));
            if (!emitSerializeField(out, step, "obj->" + step.cName, i, 2))
            {
                emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
            }
            emitLine(out, 1, "}");
        }
        emitLine(out, 1, "else {");
        emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG;");
        emitLine(out, 1, "}");
    }
    else
    {
        for (std::size_t i = 0; i < steps.size(); ++i)
        {
            const auto& step = steps[i];
            if (step.kind == PlanStepKind::Align)
            {
                emitSerializeAlign(out, 1, step.bits, "a" + std::to_string(i));
                continue;
            }
            if (step.kind == PlanStepKind::Padding)
            {
                emitSerializePadding(out, i, step.bits, 1);
                continue;
            }
            if (!emitSerializeField(out, step, "obj->" + step.cName, i, 1))
            {
                emitLine(out, 1, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
            }
        }
    }

    emitSerializeAlign(out, 1, 8, "final");
    emitLine(out, 1, "*inout_buffer_size_bytes = (size_t)(offset_bits / 8U);");
    emitLine(out, 1, "return (int8_t)DSDL_RUNTIME_SUCCESS;");
    emitLine(out, 0, "}");
    return out.str();
}

std::string renderTypedDeserializeFunction(llvm::StringRef              functionName,
                                           llvm::StringRef              cTypeName,
                                           llvm::StringRef              cDeserializeSymbol,
                                           llvm::StringRef              fullName,
                                           llvm::StringRef              sectionName,
                                           std::int64_t                 minBits,
                                           std::int64_t                 maxBits,
                                           const std::vector<PlanStep>& steps,
                                           const bool                   isUnion,
                                           const std::int64_t           unionTagBits,
                                           llvm::StringRef              unionTagValidateSymbol,
                                           llvm::StringRef              unionTagDeserializeSymbol)
{
    const std::string  functionNameText       = functionName.str();
    const std::string  cTypeNameText          = cTypeName.str();
    const std::string  cDeserializeSymbolText = cDeserializeSymbol.str();
    const std::string  fullNameText           = fullName.str();
    const std::string  sectionNameText        = sectionName.str();
    std::ostringstream out;
    if (cTypeNameText.empty())
    {
        out << "int8_t " << functionNameText
            << "(void* out_obj, const uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    else
    {
        out << "int8_t " << functionNameText << "(" << cTypeNameText
            << "* const out_obj, const uint8_t* buffer, size_t* const "
               "inout_buffer_size_bytes)\n";
    }
    out << "{\n";
    emitLine(out,
             1,
             "// IR section: " + fullNameText +
                 (sectionNameText.empty() ? std::string{} : (" (" + sectionNameText + ")")) +
                 ", min_bits=" + std::to_string(minBits) + ", max_bits=" + std::to_string(maxBits) + ".");
    if (!cDeserializeSymbolText.empty())
    {
        emitLine(out, 1, "// Public C API symbol: " + cDeserializeSymbolText);
    }
    emitLine(out, 1, "// Typed IR lowering path.");
    emitLine(out,
             1,
             "if ((out_obj == NULL) || (inout_buffer_size_bytes == NULL) || "
             "((buffer == NULL) && (0U != *inout_buffer_size_bytes))) {");
    emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (buffer == NULL) {");
    emitLine(out, 2, "buffer = (const uint8_t*)\"\";");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "const size_t capacity_bits = capacity_bytes * 8U;");
    emitLine(out, 1, "size_t offset_bits = 0U;");

    if (isUnion)
    {
        const auto tagBits = nonNegative(unionTagBits);
        emitLine(out,
                 1,
                 "const uint64_t _tag_wire = " + unsignedGetterForBits(tagBits) +
                     "(buffer, capacity_bytes, offset_bits, (uint8_t)" + std::to_string(tagBits) + "U);");
        emitLine(out,
                 1,
                 "const uint64_t _tag_value = (uint64_t)" + unionTagDeserializeSymbol.str() + "((int64_t)_tag_wire);");
        emitLine(out, 1, "const int8_t _err_union_tag = " + unionTagValidateSymbol.str() + "((int64_t)_tag_value);");
        emitLine(out, 1, "if (_err_union_tag < 0) {");
        emitLine(out, 2, "return _err_union_tag;");
        emitLine(out, 1, "}");
        emitLine(out, 1, "out_obj->_tag_ = (uint8_t)_tag_value;");
        emitLine(out, 1, "offset_bits += " + std::to_string(tagBits) + "U;");

        std::vector<const PlanStep*> unionFields;
        for (const auto& step : steps)
        {
            if (step.kind == PlanStepKind::Field)
            {
                unionFields.push_back(&step);
            }
        }
        std::sort(unionFields.begin(), unionFields.end(), [](const PlanStep* lhs, const PlanStep* rhs) {
            return lhs->unionOptionIndex < rhs->unionOptionIndex;
        });

        bool first = true;
        for (std::size_t i = 0; i < unionFields.size(); ++i)
        {
            const auto& step = *unionFields[i];
            emitLine(out,
                     1,
                     std::string(first ? "if" : "else if") +
                         " (_tag_value == " + std::to_string(nonNegative(step.unionOptionIndex)) + "U) {");
            first = false;
            emitDeserializeAlign(out, 2, step.alignmentBits);
            if (!emitDeserializeField(out, step, "out_obj->" + step.cName, i, 2))
            {
                emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
            }
            emitLine(out, 1, "}");
        }
        emitLine(out, 1, "else {");
        emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG;");
        emitLine(out, 1, "}");
    }
    else
    {
        for (std::size_t i = 0; i < steps.size(); ++i)
        {
            const auto& step = steps[i];
            if (step.kind == PlanStepKind::Align)
            {
                emitDeserializeAlign(out, 1, step.bits);
                continue;
            }
            if (step.kind == PlanStepKind::Padding)
            {
                emitLine(out, 1, "offset_bits += " + std::to_string(nonNegative(step.bits)) + "U;");
                continue;
            }
            if (!emitDeserializeField(out, step, "out_obj->" + step.cName, i, 1))
            {
                emitLine(out, 1, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
            }
        }
    }

    emitDeserializeAlign(out, 1, 8);
    emitLine(out,
             1,
             "*inout_buffer_size_bytes = (size_t)(dsdl_runtime_choose_min(offset_bits, "
             "capacity_bits) / 8U);");
    emitLine(out, 1, "return (int8_t)DSDL_RUNTIME_SUCCESS;");
    emitLine(out, 0, "}");
    return out.str();
}

struct ConvertDSDLToEmitCPass : public mlir::PassWrapper<ConvertDSDLToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    llvm::StringRef getArgument() const final
    {
        return "convert-dsdl-to-emitc";
    }
    llvm::StringRef getDescription() const final
    {
        return "Lower DSDL dialect schema ops into Func/Arith ops for EmitC lowering";
    }
    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::emitc::EmitCDialect>();
    }

    void runOnOperation() override
    {
        auto       module               = getOperation();
        const bool headersAvailable     = module->hasAttr("llvmdsdl.headers_available");
        const bool requireTypedLowering = module->hasAttr("llvmdsdl.require_typed_lowering");
        if (requireTypedLowering && !headersAvailable)
        {
            module.emitError("typed lowering requires header availability");
            signalPassFailure();
            return;
        }
        auto& body = module.getBodyRegion().front();

        std::vector<mlir::Operation*> schemaOps;
        for (mlir::Operation& op : body)
        {
            if (op.getName().getStringRef() == "dsdl.schema")
            {
                schemaOps.push_back(&op);
            }
        }
        if (schemaOps.empty())
        {
            return;
        }
        const auto contractVersion = module->getAttrOfType<mlir::IntegerAttr>(kLoweredSerDesContractVersionAttr);
        if (!contractVersion)
        {
            module.emitError("lowered SerDes contract missing module attribute '" +
                             std::string(kLoweredSerDesContractVersionAttr) +
                             "'; run lower-dsdl-serialization before "
                             "convert-dsdl-to-emitc");
            signalPassFailure();
            return;
        }
        if (contractVersion.getInt() != kLoweredSerDesContractVersion)
        {
            module.emitError("lowered SerDes contract version mismatch: expected " +
                             std::to_string(kLoweredSerDesContractVersion) + ", got " +
                             std::to_string(contractVersion.getInt()) +
                             "; run matching lower-dsdl-serialization before convert-dsdl-to-emitc");
            signalPassFailure();
            return;
        }
        const auto contractProducer = module->getAttrOfType<mlir::StringAttr>(kLoweredSerDesContractProducerAttr);
        if (!contractProducer || contractProducer.getValue() != kLoweredSerDesContractProducer)
        {
            module.emitError("lowered SerDes contract producer mismatch: expected '" +
                             std::string(kLoweredSerDesContractProducer) +
                             "'; run lower-dsdl-serialization before "
                             "convert-dsdl-to-emitc");
            signalPassFailure();
            return;
        }

        std::vector<std::string> emittedFunctions;
        emittedFunctions.reserve(schemaOps.size() * 4U);
        std::set<std::string> forwardDeclaredTypes;
        std::set<std::string> capacityCheckSymbols;
        std::set<std::string> unionTagValidateSymbols;
        std::set<std::string> unionTagIoHelperSymbols;
        std::set<std::string> scalarUnsignedHelperSymbols;
        std::set<std::string> scalarSignedHelperSymbols;
        std::set<std::string> scalarFloatHelperSymbols;
        std::set<std::string> arrayLengthPrefixHelperSymbols;
        std::set<std::string> arrayLengthValidateSymbols;
        std::set<std::string> delimiterValidateSymbols;
        std::set<std::string> typedHeaders;

        for (mlir::Operation* schema : schemaOps)
        {
            const auto symNameAttr = schema->getAttrOfType<mlir::StringAttr>("sym_name");
            if (!symNameAttr)
            {
                continue;
            }
            const auto        fullNameAttr = schema->getAttrOfType<mlir::StringAttr>("full_name");
            const std::string fullName = fullNameAttr ? fullNameAttr.getValue().str() : symNameAttr.getValue().str();
            const auto        headerPathAttr = schema->getAttrOfType<mlir::StringAttr>("header_path");
            const std::string headerPath     = headerPathAttr ? headerPathAttr.getValue().str() : std::string{};

            if (schema->getNumRegions() == 0 || schema->getRegion(0).empty())
            {
                continue;
            }

            for (mlir::Operation& child : schema->getRegion(0).front())
            {
                if (child.getName().getStringRef() != "dsdl.serialization_plan")
                {
                    continue;
                }
                const auto planContractVersion =
                    child.getAttrOfType<mlir::IntegerAttr>(kLoweredSerDesContractVersionAttr);
                if (!planContractVersion || planContractVersion.getInt() != kLoweredSerDesContractVersion)
                {
                    child.emitOpError("missing or incompatible lowered contract version; run "
                                      "lower-dsdl-serialization before convert-dsdl-to-emitc");
                    signalPassFailure();
                    return;
                }
                const auto planContractProducer =
                    child.getAttrOfType<mlir::StringAttr>(kLoweredSerDesContractProducerAttr);
                if (!planContractProducer || planContractProducer.getValue() != kLoweredSerDesContractProducer)
                {
                    child.emitOpError("missing lowered contract producer marker; run "
                                      "lower-dsdl-serialization before convert-dsdl-to-emitc");
                    signalPassFailure();
                    return;
                }
                if (mlir::failed(verifyLoweredPlanContract(module, &child)))
                {
                    signalPassFailure();
                    return;
                }
                const auto        sectionAttr = child.getAttrOfType<mlir::StringAttr>("section");
                const std::string section     = sectionAttr ? sectionAttr.getValue().str() : std::string{};
                const std::string fnStem      = symNameAttr.getValue().str() + sectionSuffix(section);
                const auto capacityCheckAttr  = child.getAttrOfType<mlir::StringAttr>(kLoweredCapacityCheckHelperAttr);
                const std::string capacityCheckSymbol =
                    capacityCheckAttr ? capacityCheckAttr.getValue().str() : std::string{};
                const auto loweredMinBitsAttr    = child.getAttrOfType<mlir::IntegerAttr>(kLoweredMinBitsAttr);
                const auto loweredMaxBitsAttr    = child.getAttrOfType<mlir::IntegerAttr>(kLoweredMaxBitsAttr);
                const auto loweredStepCountAttr  = child.getAttrOfType<mlir::IntegerAttr>(kLoweredStepCountAttr);
                const auto loweredFieldCountAttr = child.getAttrOfType<mlir::IntegerAttr>(kLoweredFieldCountAttr);
                if (!loweredMinBitsAttr || !loweredMaxBitsAttr || !loweredStepCountAttr || !loweredFieldCountAttr)
                {
                    child.emitOpError("missing required lowered plan metadata; run "
                                      "lower-dsdl-serialization before convert-dsdl-to-emitc");
                    signalPassFailure();
                    return;
                }
                const std::int64_t minBits = nonNegative(loweredMinBitsAttr.getInt());
                const std::int64_t maxBits = nonNegative(loweredMaxBitsAttr.getInt());
                if (maxBits < minBits)
                {
                    child.emitOpError("invalid lowered bit-range metadata");
                    signalPassFailure();
                    return;
                }
                const auto cTypeNameAttr          = child.getAttrOfType<mlir::StringAttr>("c_type_name");
                const auto cSerializeSymbolAttr   = child.getAttrOfType<mlir::StringAttr>("c_serialize_symbol");
                const auto cDeserializeSymbolAttr = child.getAttrOfType<mlir::StringAttr>("c_deserialize_symbol");
                const std::string cTypeName       = cTypeNameAttr ? cTypeNameAttr.getValue().str() : std::string{};
                if (!cTypeName.empty())
                {
                    forwardDeclaredTypes.insert(cTypeName);
                }
                const auto steps = collectPlanSteps(&child);
                for (const auto& step : steps)
                {
                    if (!step.serUnsignedHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.serUnsignedHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.serUnsignedHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarUnsignedHelperSymbols.insert(step.serUnsignedHelper);
                    }
                    if (!step.deserUnsignedHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.deserUnsignedHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.deserUnsignedHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarUnsignedHelperSymbols.insert(step.deserUnsignedHelper);
                    }
                    if (!step.serSignedHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.serSignedHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.serSignedHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarSignedHelperSymbols.insert(step.serSignedHelper);
                    }
                    if (!step.deserSignedHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.deserSignedHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.deserSignedHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarSignedHelperSymbols.insert(step.deserSignedHelper);
                    }
                    if (!step.serFloatHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.serFloatHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.serFloatHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarFloatHelperSymbols.insert(step.serFloatHelper);
                    }
                    if (!step.deserFloatHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.deserFloatHelper))
                        {
                            child.emitOpError("missing lowered scalar helper symbol: " + step.deserFloatHelper);
                            signalPassFailure();
                            return;
                        }
                        scalarFloatHelperSymbols.insert(step.deserFloatHelper);
                    }
                    if (!step.serArrayLengthPrefixHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.serArrayLengthPrefixHelper))
                        {
                            child.emitOpError("missing lowered array-length-prefix helper symbol: " +
                                              step.serArrayLengthPrefixHelper);
                            signalPassFailure();
                            return;
                        }
                        arrayLengthPrefixHelperSymbols.insert(step.serArrayLengthPrefixHelper);
                    }
                    if (!step.deserArrayLengthPrefixHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.deserArrayLengthPrefixHelper))
                        {
                            child.emitOpError("missing lowered array-length-prefix helper symbol: " +
                                              step.deserArrayLengthPrefixHelper);
                            signalPassFailure();
                            return;
                        }
                        arrayLengthPrefixHelperSymbols.insert(step.deserArrayLengthPrefixHelper);
                    }
                    if (!step.arrayLengthValidateHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.arrayLengthValidateHelper))
                        {
                            child.emitOpError("missing lowered array-length helper symbol: " +
                                              step.arrayLengthValidateHelper);
                            signalPassFailure();
                            return;
                        }
                        arrayLengthValidateSymbols.insert(step.arrayLengthValidateHelper);
                    }
                    if (!step.delimiterValidateHelper.empty())
                    {
                        if (!module.lookupSymbol<mlir::func::FuncOp>(step.delimiterValidateHelper))
                        {
                            child.emitOpError("missing lowered delimiter helper symbol: " +
                                              step.delimiterValidateHelper);
                            signalPassFailure();
                            return;
                        }
                        delimiterValidateSymbols.insert(step.delimiterValidateHelper);
                    }
                }
                const bool         isUnion          = child.hasAttr("is_union");
                const auto         unionTagBitsAttr = child.getAttrOfType<mlir::IntegerAttr>("union_tag_bits");
                const std::int64_t unionTagBits     = unionTagBitsAttr ? nonNegative(unionTagBitsAttr.getInt()) : 0;
                const auto unionTagSerHelperAttr = child.getAttrOfType<mlir::StringAttr>(kLoweredSerUnionTagHelperAttr);
                const auto unionTagDeserHelperAttr =
                    child.getAttrOfType<mlir::StringAttr>(kLoweredDeserUnionTagHelperAttr);
                const std::string unionTagSerializeHelper =
                    unionTagSerHelperAttr ? unionTagSerHelperAttr.getValue().str() : std::string{};
                const std::string unionTagDeserializeHelper =
                    unionTagDeserHelperAttr ? unionTagDeserHelperAttr.getValue().str() : std::string{};
                std::string unionTagValidateSymbol;
                if (isUnion)
                {
                    const auto unionTagValidateAttr =
                        child.getAttrOfType<mlir::StringAttr>(kLoweredUnionTagValidateHelperAttr);
                    unionTagValidateSymbol =
                        unionTagValidateAttr ? unionTagValidateAttr.getValue().str() : std::string{};
                    unionTagValidateSymbols.insert(unionTagValidateSymbol);
                    if (unionTagSerializeHelper.empty() || unionTagDeserializeHelper.empty())
                    {
                        child.emitOpError("missing lowered union-tag IO helper symbol; run "
                                          "lower-dsdl-serialization before "
                                          "convert-dsdl-to-emitc");
                        signalPassFailure();
                        return;
                    }
                    if (!module.lookupSymbol<mlir::func::FuncOp>(unionTagSerializeHelper) ||
                        !module.lookupSymbol<mlir::func::FuncOp>(unionTagDeserializeHelper))
                    {
                        child.emitOpError("missing lowered union-tag IO helper body; run "
                                          "lower-dsdl-serialization before "
                                          "convert-dsdl-to-emitc");
                        signalPassFailure();
                        return;
                    }
                    unionTagIoHelperSymbols.insert(unionTagSerializeHelper);
                    unionTagIoHelperSymbols.insert(unionTagDeserializeHelper);
                }
                const bool useTyped = headersAvailable && supportsTypedLowering(steps, isUnion, unionTagBits);
                if (requireTypedLowering && !useTyped)
                {
                    std::string reason = "typed lowering is required for this module";
                    if (!headersAvailable)
                    {
                        reason += " but generated headers are not available";
                    }
                    else
                    {
                        reason += " but the serialization plan contains unsupported constructs";
                    }
                    if (!section.empty())
                    {
                        reason += " (section: " + section + ")";
                    }
                    child.emitOpError(reason);
                    signalPassFailure();
                    return;
                }
                if (useTyped)
                {
                    if (isUnion && (unionTagSerializeHelper.empty() || unionTagDeserializeHelper.empty()))
                    {
                        child.emitOpError("typed lowering requires lowered union-tag IO helpers; run "
                                          "lower-dsdl-serialization before convert-dsdl-to-emitc");
                        signalPassFailure();
                        return;
                    }
                    for (const auto& step : steps)
                    {
                        if (step.kind != PlanStepKind::Field)
                        {
                            continue;
                        }
                        if (isVariableArrayKind(step.arrayKind) && step.arrayLengthValidateHelper.empty())
                        {
                            child.emitOpError("typed lowering requires lowered array-length validation helper "
                                              "for variable array field '" +
                                              step.cName +
                                              "'; run lower-dsdl-serialization before "
                                              "convert-dsdl-to-emitc");
                            signalPassFailure();
                            return;
                        }
                        if (isVariableArrayKind(step.arrayKind) &&
                            (step.serArrayLengthPrefixHelper.empty() || step.deserArrayLengthPrefixHelper.empty()))
                        {
                            child.emitOpError("typed lowering requires lowered array-length-prefix IO "
                                              "helpers for variable array field '" +
                                              step.cName +
                                              "'; run lower-dsdl-serialization before "
                                              "convert-dsdl-to-emitc");
                            signalPassFailure();
                            return;
                        }
                        if (step.scalarCategory == "unsigned" || step.scalarCategory == "byte" ||
                            step.scalarCategory == "utf8")
                        {
                            if (step.serUnsignedHelper.empty() || step.deserUnsignedHelper.empty())
                            {
                                child.emitOpError("typed lowering requires lowered unsigned scalar helpers for "
                                                  "field '" +
                                                  step.cName +
                                                  "'; run lower-dsdl-serialization before "
                                                  "convert-dsdl-to-emitc");
                                signalPassFailure();
                                return;
                            }
                        }
                        else if (step.scalarCategory == "signed")
                        {
                            if (step.serSignedHelper.empty() || step.deserSignedHelper.empty())
                            {
                                child.emitOpError("typed lowering requires lowered signed scalar helpers for "
                                                  "field '" +
                                                  step.cName +
                                                  "'; run lower-dsdl-serialization before "
                                                  "convert-dsdl-to-emitc");
                                signalPassFailure();
                                return;
                            }
                        }
                        else if (step.scalarCategory == "float")
                        {
                            if (step.serFloatHelper.empty() || step.deserFloatHelper.empty())
                            {
                                child.emitOpError("typed lowering requires lowered float scalar helpers for "
                                                  "field '" +
                                                  step.cName +
                                                  "'; run lower-dsdl-serialization before "
                                                  "convert-dsdl-to-emitc");
                                signalPassFailure();
                                return;
                            }
                        }
                        else if (step.scalarCategory == "composite" && !step.compositeSealed)
                        {
                            if (step.delimiterValidateHelper.empty())
                            {
                                child.emitOpError("typed lowering requires lowered delimiter-header validation "
                                                  "helper for delimited composite field '" +
                                                  step.cName +
                                                  "'; run lower-dsdl-serialization before "
                                                  "convert-dsdl-to-emitc");
                                signalPassFailure();
                                return;
                            }
                        }
                    }
                }
                if (useTyped)
                {
                    if (!headerPath.empty())
                    {
                        typedHeaders.insert(headerPath);
                    }
                }
                capacityCheckSymbols.insert(capacityCheckSymbol);

                if (useTyped)
                {
                    emittedFunctions.push_back(renderTypedSerializeFunction(fnStem + "__serialize_ir_",
                                                                            cTypeName,
                                                                            cSerializeSymbolAttr
                                                                                ? cSerializeSymbolAttr.getValue()
                                                                                : llvm::StringRef{},
                                                                            fullName,
                                                                            section,
                                                                            minBits,
                                                                            maxBits,
                                                                            steps,
                                                                            isUnion,
                                                                            unionTagBits,
                                                                            capacityCheckSymbol,
                                                                            unionTagValidateSymbol,
                                                                            unionTagSerializeHelper));
                    emittedFunctions.push_back(renderTypedDeserializeFunction(fnStem + "__deserialize_ir_",
                                                                              cTypeName,
                                                                              cDeserializeSymbolAttr
                                                                                  ? cDeserializeSymbolAttr.getValue()
                                                                                  : llvm::StringRef{},
                                                                              fullName,
                                                                              section,
                                                                              minBits,
                                                                              maxBits,
                                                                              steps,
                                                                              isUnion,
                                                                              unionTagBits,
                                                                              unionTagValidateSymbol,
                                                                              unionTagDeserializeHelper));
                }
                else
                {
                    emittedFunctions.push_back(renderGenericSerializeFunction(fnStem + "__serialize_ir_",
                                                                              cTypeName,
                                                                              cSerializeSymbolAttr
                                                                                  ? cSerializeSymbolAttr.getValue()
                                                                                  : llvm::StringRef{},
                                                                              fullName,
                                                                              section,
                                                                              minBits,
                                                                              maxBits,
                                                                              steps,
                                                                              capacityCheckSymbol));
                    emittedFunctions.push_back(renderGenericDeserializeFunction(fnStem + "__deserialize_ir_",
                                                                                cTypeName,
                                                                                cDeserializeSymbolAttr
                                                                                    ? cDeserializeSymbolAttr.getValue()
                                                                                    : llvm::StringRef{},
                                                                                fullName,
                                                                                section,
                                                                                minBits,
                                                                                maxBits,
                                                                                steps));
                }
            }
        }

        mlir::OpBuilder builder(module.getContext());
        builder.setInsertionPointToStart(&body);
        const mlir::Location loc = builder.getUnknownLoc();
        builder.create<mlir::emitc::VerbatimOp>(loc, "#include <stddef.h>");
        builder.create<mlir::emitc::VerbatimOp>(loc, "#include <stdint.h>");
        builder.create<mlir::emitc::VerbatimOp>(loc, "#include \"dsdl_runtime.h\"");
        if (headersAvailable)
        {
            for (const auto& headerPath : typedHeaders)
            {
                builder.create<mlir::emitc::VerbatimOp>(loc, "#include \"" + headerPath + "\"");
            }
        }
        builder.create<mlir::emitc::VerbatimOp>(loc, "/* Generated from DSDL IR by convert-dsdl-to-emitc. */");
        for (const auto& typeName : forwardDeclaredTypes)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "typedef struct " + typeName + " " + typeName + ";");
        }
        for (const auto& symbol : capacityCheckSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int8_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : unionTagValidateSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int8_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : unionTagIoHelperSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int64_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : scalarUnsignedHelperSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int64_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : scalarSignedHelperSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int64_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : scalarFloatHelperSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "double " + symbol + "(double);");
        }
        for (const auto& symbol : arrayLengthPrefixHelperSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int64_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : arrayLengthValidateSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int8_t " + symbol + "(int64_t);");
        }
        for (const auto& symbol : delimiterValidateSymbols)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, "int8_t " + symbol + "(int64_t, int64_t);");
        }
        for (const auto& fn : emittedFunctions)
        {
            builder.create<mlir::emitc::VerbatimOp>(loc, fn);
        }

        for (mlir::Operation* op : schemaOps)
        {
            op->erase();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> createConvertDSDLToEmitCPass()
{
    return std::make_unique<ConvertDSDLToEmitCPass>();
}

void registerDSDLConvertPasses()
{
    static bool once = false;
    if (once)
    {
        return;
    }
    once = true;
    static mlir::PassRegistration<ConvertDSDLToEmitCPass> reg;
}

}  // namespace llvmdsdl
