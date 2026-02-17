#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace llvmdsdl {

namespace {

std::int64_t nonNegative(const std::int64_t value) {
  return std::max<std::int64_t>(value, 0);
}

} // namespace

std::string loweredTypeKey(const std::string &name, std::uint32_t major,
                           std::uint32_t minor) {
  return name + ":" + std::to_string(major) + ":" + std::to_string(minor);
}

const LoweredFieldFacts *
findLoweredFieldFacts(const LoweredSectionFacts *const sectionFacts,
                      const std::string &fieldName) {
  if (sectionFacts == nullptr) {
    return nullptr;
  }
  const auto it = sectionFacts->fieldsByName.find(fieldName);
  if (it == sectionFacts->fieldsByName.end()) {
    return nullptr;
  }
  return &it->second;
}

std::optional<std::uint32_t>
loweredFieldArrayPrefixBits(const LoweredSectionFacts *const sectionFacts,
                            const std::string &fieldName) {
  const auto *const fieldFacts = findLoweredFieldFacts(sectionFacts, fieldName);
  if (fieldFacts == nullptr) {
    return std::nullopt;
  }
  return fieldFacts->arrayLengthPrefixBits;
}

bool collectLoweredFactsFromMlir(const SemanticModule &semantic,
                                 mlir::ModuleOp module,
                                 DiagnosticEngine &diagnostics,
                                 const std::string &backendLabel,
                                 LoweredFactsMap *const outFacts) {
  std::unordered_map<std::string, std::set<std::string>> keyToSections;
  LoweredFactsMap loweredFacts;
  auto loweredModule = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  mlir::PassManager pm(module.getContext());
  pm.addPass(createLowerDSDLSerializationPass());
  if (mlir::failed(pm.run(*loweredModule))) {
    diagnostics.error({"<mlir>", 1, 1},
                      "failed to run lower-dsdl-serialization for " +
                          backendLabel + " backend validation");
    return false;
  }

  for (mlir::Operation &op : loweredModule->getBodyRegion().front()) {
    if (op.getName().getStringRef() != "dsdl.schema") {
      continue;
    }

    const auto fullName = op.getAttrOfType<mlir::StringAttr>("full_name");
    const auto major = op.getAttrOfType<mlir::IntegerAttr>("major");
    const auto minor = op.getAttrOfType<mlir::IntegerAttr>("minor");
    if (!fullName || !major || !minor) {
      diagnostics.error({"<mlir>", 1, 1},
                        "dsdl.schema missing identity attributes");
      return false;
    }

    const auto key = loweredTypeKey(fullName.getValue().str(),
                                    static_cast<std::uint32_t>(major.getInt()),
                                    static_cast<std::uint32_t>(minor.getInt()));
    auto &sections = keyToSections[key];

    if (op.getNumRegions() == 0 || op.getRegion(0).empty()) {
      diagnostics.error({"<mlir>", 1, 1},
                        "dsdl.schema has no body region for " +
                            fullName.getValue().str());
      return false;
    }

    for (mlir::Operation &child : op.getRegion(0).front()) {
      if (child.getName().getStringRef() != "dsdl.serialization_plan") {
        continue;
      }
      std::string section;
      if (const auto sectionAttr =
              child.getAttrOfType<mlir::StringAttr>("section")) {
        section = sectionAttr.getValue().str();
      }
      auto &sectionFacts = loweredFacts[key][section];
      if (!sections.insert(section).second) {
        diagnostics.error({"<mlir>", 1, 1},
                          "duplicate dsdl.serialization_plan section '" +
                              section + "' for " + fullName.getValue().str());
        return false;
      }

      const auto minBits = child.getAttrOfType<mlir::IntegerAttr>("min_bits");
      const auto maxBits = child.getAttrOfType<mlir::IntegerAttr>("max_bits");
      if (!minBits || !maxBits) {
        diagnostics.error({"<mlir>", 1, 1},
                          "serialization plan missing min_bits/max_bits for " +
                              fullName.getValue().str());
        return false;
      }
      const auto capacityCheckHelper =
          child.getAttrOfType<mlir::StringAttr>("lowered_capacity_check_helper");
      if (!capacityCheckHelper) {
        diagnostics.error({"<mlir>", 1, 1},
                          "serialization plan missing lowered capacity helper "
                          "for " +
                              fullName.getValue().str());
        return false;
      }
      sectionFacts.capacityCheckHelper = capacityCheckHelper.getValue().str();

      if (child.hasAttr("is_union")) {
        const auto unionTagBits =
            child.getAttrOfType<mlir::IntegerAttr>("union_tag_bits");
        const auto unionOptionCount =
            child.getAttrOfType<mlir::IntegerAttr>("union_option_count");
        if (!unionTagBits || !unionOptionCount) {
          diagnostics.error(
              {"<mlir>", 1, 1},
              "union plan missing union_tag_bits/union_option_count for " +
                  fullName.getValue().str());
          return false;
        }
        const auto serUnionTagHelper =
            child.getAttrOfType<mlir::StringAttr>("lowered_ser_union_tag_helper");
        const auto deserUnionTagHelper = child.getAttrOfType<mlir::StringAttr>(
            "lowered_deser_union_tag_helper");
        const auto unionTagValidateHelper = child.getAttrOfType<mlir::StringAttr>(
            "lowered_union_tag_validate_helper");
        if (!serUnionTagHelper || !deserUnionTagHelper ||
            !unionTagValidateHelper) {
          diagnostics.error({"<mlir>", 1, 1},
                            "union plan missing lowered union-tag helpers for " +
                                fullName.getValue().str());
          return false;
        }
        sectionFacts.unionTagBits =
            static_cast<std::uint32_t>(unionTagBits.getInt());
        sectionFacts.unionTagValidateHelper =
            unionTagValidateHelper.getValue().str();
        sectionFacts.serUnionTagHelper = serUnionTagHelper.getValue().str();
        sectionFacts.deserUnionTagHelper = deserUnionTagHelper.getValue().str();
      }

      if (child.getNumRegions() == 0 || child.getRegion(0).empty()) {
        diagnostics.error({"<mlir>", 1, 1},
                          "serialization plan has no body for " +
                              fullName.getValue().str());
        return false;
      }

      for (mlir::Operation &step : child.getRegion(0).front()) {
        const auto stepName = step.getName().getStringRef();
        if (stepName == "dsdl.align") {
          if (!step.getAttrOfType<mlir::IntegerAttr>("bits")) {
            diagnostics.error({"<mlir>", 1, 1},
                              "dsdl.align missing bits attribute for " +
                                  fullName.getValue().str());
            return false;
          }
          continue;
        }
        if (stepName != "dsdl.io") {
          continue;
        }

        const auto scalarCategory =
            step.getAttrOfType<mlir::StringAttr>("scalar_category");
        const auto arrayKind = step.getAttrOfType<mlir::StringAttr>("array_kind");
        const auto kind = step.getAttrOfType<mlir::StringAttr>("kind");
        const auto bitLength = step.getAttrOfType<mlir::IntegerAttr>("bit_length");
        const auto alignmentBits =
            step.getAttrOfType<mlir::IntegerAttr>("alignment_bits");
        if (!scalarCategory || !arrayKind || !kind || !bitLength ||
            !alignmentBits) {
          diagnostics.error({"<mlir>", 1, 1},
                            "dsdl.io missing core type metadata for " +
                                fullName.getValue().str());
          return false;
        }
        const bool isPadding = kind.getValue() == "padding";

        const auto arrayPrefixBits =
            step.getAttrOfType<mlir::IntegerAttr>("array_length_prefix_bits");
        if (arrayKind.getValue().starts_with("variable") &&
            (!arrayPrefixBits || arrayPrefixBits.getInt() <= 0)) {
          diagnostics.error({"<mlir>", 1, 1},
                            "variable array step missing valid prefix width for " +
                                fullName.getValue().str());
          return false;
        }
        if (!isPadding && arrayKind.getValue().starts_with("variable")) {
          const auto serArrayPrefixHelper = step.getAttrOfType<mlir::StringAttr>(
              "lowered_ser_array_length_prefix_helper");
          const auto deserArrayPrefixHelper =
              step.getAttrOfType<mlir::StringAttr>(
                  "lowered_deser_array_length_prefix_helper");
          const auto arrayValidateHelper = step.getAttrOfType<mlir::StringAttr>(
              "lowered_array_length_validate_helper");
          if (!serArrayPrefixHelper || !deserArrayPrefixHelper ||
              !arrayValidateHelper) {
            diagnostics.error(
                {"<mlir>", 1, 1},
                "variable array step missing lowered array-length helpers for " +
                    fullName.getValue().str());
            return false;
          }
        }

        if (!isPadding && arrayKind.getValue() == "none") {
          const auto category = scalarCategory.getValue();
          if (category == "unsigned" || category == "byte" ||
              category == "utf8") {
            if (!step.getAttrOfType<mlir::StringAttr>("lowered_ser_unsigned_helper") ||
                !step.getAttrOfType<mlir::StringAttr>(
                    "lowered_deser_unsigned_helper")) {
              diagnostics.error(
                  {"<mlir>", 1, 1},
                  "unsigned scalar step missing lowered scalar helpers for " +
                      fullName.getValue().str());
              return false;
            }
          } else if (category == "signed") {
            if (!step.getAttrOfType<mlir::StringAttr>("lowered_ser_signed_helper") ||
                !step.getAttrOfType<mlir::StringAttr>(
                    "lowered_deser_signed_helper")) {
              diagnostics.error(
                  {"<mlir>", 1, 1},
                  "signed scalar step missing lowered scalar helpers for " +
                      fullName.getValue().str());
              return false;
            }
          } else if (category == "float") {
            if (!step.getAttrOfType<mlir::StringAttr>("lowered_ser_float_helper") ||
                !step.getAttrOfType<mlir::StringAttr>(
                    "lowered_deser_float_helper")) {
              diagnostics.error(
                  {"<mlir>", 1, 1},
                  "float scalar step missing lowered scalar helpers for " +
                      fullName.getValue().str());
              return false;
            }
          }
        }

        if (scalarCategory.getValue() == "composite") {
          const auto compositeFullName =
              step.getAttrOfType<mlir::StringAttr>("composite_full_name");
          const auto compositeCTypeName =
              step.getAttrOfType<mlir::StringAttr>("composite_c_type_name");
          if (!compositeFullName || !compositeCTypeName) {
            diagnostics.error({"<mlir>", 1, 1},
                              "composite dsdl.io missing target metadata for " +
                                  fullName.getValue().str());
            return false;
          }
          const auto compositeSealed =
              step.getAttrOfType<mlir::BoolAttr>("composite_sealed");
          if (compositeSealed && !compositeSealed.getValue() &&
              !step.getAttrOfType<mlir::IntegerAttr>("composite_extent_bits")) {
            diagnostics.error(
                {"<mlir>", 1, 1},
                "delimited composite step missing composite_extent_bits for " +
                    fullName.getValue().str());
            return false;
          }
          if (!isPadding && compositeSealed && !compositeSealed.getValue() &&
              !step.getAttrOfType<mlir::StringAttr>(
                  "lowered_delimiter_validate_helper")) {
            diagnostics.error(
                {"<mlir>", 1, 1},
                "delimited composite step missing lowered delimiter helper for " +
                    fullName.getValue().str());
            return false;
          }
        }

        if (const auto fieldNameAttr = step.getAttrOfType<mlir::StringAttr>("name");
            fieldNameAttr) {
          auto &fieldFacts =
              sectionFacts.fieldsByName[fieldNameAttr.getValue().str()];
          if (const auto stepIndex =
                  step.getAttrOfType<mlir::IntegerAttr>("step_index")) {
            fieldFacts.stepIndex = nonNegative(stepIndex.getInt());
          }
          if (arrayKind.getValue().starts_with("variable") && arrayPrefixBits &&
              arrayPrefixBits.getInt() > 0) {
            fieldFacts.arrayLengthPrefixBits =
                static_cast<std::uint32_t>(arrayPrefixBits.getInt());
            if (const auto serArrayPrefixHelper =
                    step.getAttrOfType<mlir::StringAttr>(
                        "lowered_ser_array_length_prefix_helper")) {
              fieldFacts.serArrayLengthPrefixHelper =
                  serArrayPrefixHelper.getValue().str();
            }
            if (const auto deserArrayPrefixHelper =
                    step.getAttrOfType<mlir::StringAttr>(
                        "lowered_deser_array_length_prefix_helper")) {
              fieldFacts.deserArrayLengthPrefixHelper =
                  deserArrayPrefixHelper.getValue().str();
            }
            if (const auto arrayValidateHelper =
                    step.getAttrOfType<mlir::StringAttr>(
                        "lowered_array_length_validate_helper")) {
              fieldFacts.arrayLengthValidateHelper =
                  arrayValidateHelper.getValue().str();
            }
          }
          if (const auto serUnsigned = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_ser_unsigned_helper")) {
            fieldFacts.serUnsignedHelper = serUnsigned.getValue().str();
          }
          if (const auto deserUnsigned = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_deser_unsigned_helper")) {
            fieldFacts.deserUnsignedHelper = deserUnsigned.getValue().str();
          }
          if (const auto serSigned = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_ser_signed_helper")) {
            fieldFacts.serSignedHelper = serSigned.getValue().str();
          }
          if (const auto deserSigned = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_deser_signed_helper")) {
            fieldFacts.deserSignedHelper = deserSigned.getValue().str();
          }
          if (const auto serFloat = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_ser_float_helper")) {
            fieldFacts.serFloatHelper = serFloat.getValue().str();
          }
          if (const auto deserFloat = step.getAttrOfType<mlir::StringAttr>(
                  "lowered_deser_float_helper")) {
            fieldFacts.deserFloatHelper = deserFloat.getValue().str();
          }
          if (const auto delimiterValidateHelper =
                  step.getAttrOfType<mlir::StringAttr>(
                      "lowered_delimiter_validate_helper")) {
            fieldFacts.delimiterValidateHelper =
                delimiterValidateHelper.getValue().str();
          }
        }
      }
    }
  }

  for (const auto &def : semantic.definitions) {
    const auto key = loweredTypeKey(def.info.fullName, def.info.majorVersion,
                                    def.info.minorVersion);
    const auto it = keyToSections.find(key);
    if (it == keyToSections.end()) {
      diagnostics.error({"<mlir>", 1, 1},
                        "missing dsdl.schema for " + def.info.fullName);
      return false;
    }

    std::set<std::string> expectedSections;
    if (def.isService) {
      expectedSections.insert("request");
      expectedSections.insert("response");
    } else {
      expectedSections.insert("");
    }

    for (const auto &sectionName : expectedSections) {
      if (!it->second.contains(sectionName)) {
        diagnostics.error({"<mlir>", 1, 1},
                          "missing dsdl.serialization_plan section '" +
                              sectionName + "' for " + def.info.fullName);
        return false;
      }
    }
  }

  if (outFacts != nullptr) {
    *outFacts = std::move(loweredFacts);
  }
  return true;
}

} // namespace llvmdsdl
