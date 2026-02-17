#include "llvmdsdl/CodeGen/HelperBindingRender.h"

#include <functional>
#include <string>
#include <vector>

namespace llvmdsdl {

namespace {

std::string renderU64MaskLiteral(const HelperBindingRenderLanguage language,
                                 const std::uint32_t bits) {
  if (language == HelperBindingRenderLanguage::Cpp) {
    if (bits == 0U) {
      return "0ULL";
    }
    if (bits >= 64U) {
      return "18446744073709551615ULL";
    }
    return std::to_string((1ULL << bits) - 1ULL) + "ULL";
  }

  if (bits == 0U) {
    return "0u64";
  }
  if (bits >= 64U) {
    return "u64::MAX";
  }
  return std::to_string((1ULL << bits) - 1ULL) + "u64";
}

void appendLines(std::vector<std::string> &out,
                 const std::vector<std::string> &in) {
  out.insert(out.end(), in.begin(), in.end());
}

} // namespace

std::vector<std::string> renderCapacityCheckBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName,
    const std::int64_t requiredBits) {
  if (language == HelperBindingRenderLanguage::Cpp) {
    return {
        "const auto " + helperName +
            " = [](const std::int64_t capacity_bits) -> std::int8_t {",
        "return (" + std::to_string(requiredBits) +
            "LL > capacity_bits) ? static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL) : "
            "static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS);",
        "};",
    };
  }

  return {
      "let " + helperName + " = |capacity_bits: i64| -> i8 {",
      "if " + std::to_string(requiredBits) + "i64 > capacity_bits {",
      "-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL",
      "} else {",
      "crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS",
      "}",
      "};",
  };
}

std::vector<std::string> renderUnionTagMaskBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName,
    const std::uint32_t bits) {
  const auto bitMaskLiteral = renderU64MaskLiteral(language, bits);
  if (language == HelperBindingRenderLanguage::Cpp) {
    return {
        "const auto " + helperName +
            " = [](const std::uint64_t value) -> std::uint64_t { return value & " +
            bitMaskLiteral + "; };",
    };
  }

  return {
      "let " + helperName + " = |value: u64| -> u64 { value & " + bitMaskLiteral +
          " };",
  };
}

std::vector<std::string> renderUnionTagValidateBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName,
    const std::vector<std::int64_t> &allowedTags) {
  if (language == HelperBindingRenderLanguage::Cpp) {
    std::string condition;
    for (const auto tag : allowedTags) {
      if (!condition.empty()) {
        condition += " || ";
      }
      condition += "(tag_value == " + std::to_string(tag) + "LL)";
    }
    if (condition.empty()) {
      return {
          "const auto " + helperName +
              " = [](const std::int64_t tag_value) -> std::int8_t {",
          "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG);",
          "};",
      };
    }
    return {
        "const auto " + helperName +
            " = [](const std::int64_t tag_value) -> std::int8_t {",
        "return (" + condition +
            ") ? static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS) : static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG);",
        "};",
    };
  }

  std::string condition;
  for (const auto tag : allowedTags) {
    if (!condition.empty()) {
      condition += " || ";
    }
    condition += "(tag_value == " + std::to_string(tag) + "i64)";
  }
  if (condition.empty()) {
    return {
        "let " + helperName + " = |tag_value: i64| -> i8 {",
        "-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG",
        "};",
    };
  }
  return {
      "let " + helperName + " = |tag_value: i64| -> i8 {",
      "if " + condition + " {",
      "crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS",
      "} else {",
      "-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG",
      "}",
      "};",
  };
}

std::vector<std::string> renderDelimiterValidateBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName) {
  if (language == HelperBindingRenderLanguage::Cpp) {
    return {
        "const auto " + helperName +
            " = [](const std::int64_t payload_bytes, const std::int64_t remaining_bytes) -> std::int8_t {",
        "return ((payload_bytes < 0LL) || (payload_bytes > remaining_bytes)) ? static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER) : static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS);",
        "};",
    };
  }

  return {
      "let " + helperName +
          " = |payload_bytes: i64, remaining_bytes: i64| -> i8 {",
      "if (payload_bytes < 0i64) || (payload_bytes > remaining_bytes) {",
      "-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER",
      "} else {",
      "crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS",
      "}",
      "};",
  };
}

std::vector<std::string> renderScalarBinding(
    const HelperBindingRenderLanguage language,
    const ScalarBindingRenderDirection direction, const std::string &helperName,
    const ScalarHelperDescriptor &descriptor) {
  const auto maskLiteral = renderU64MaskLiteral(language, descriptor.bitLength);

  if (language == HelperBindingRenderLanguage::Cpp) {
    switch (descriptor.kind) {
    case ScalarHelperKind::Unsigned: {
      if (direction == ScalarBindingRenderDirection::Serialize &&
          descriptor.castMode == CastMode::Saturated &&
          descriptor.bitLength < 64U) {
        return {
            "const auto " + helperName +
                " = [](const std::uint64_t value) -> std::uint64_t {",
            "return (value > " + maskLiteral + ") ? " + maskLiteral +
                " : value;",
            "};",
        };
      }
      if (descriptor.bitLength < 64U) {
        return {
            "const auto " + helperName +
                " = [](const std::uint64_t value) -> std::uint64_t {",
            "return value & " + maskLiteral + ";",
            "};",
        };
      }
      return {
          "const auto " + helperName +
              " = [](const std::uint64_t value) -> std::uint64_t {",
          "return value;",
          "};",
      };
    }
    case ScalarHelperKind::Signed: {
      if (direction == ScalarBindingRenderDirection::Serialize &&
          descriptor.castMode == CastMode::Saturated &&
          descriptor.bitLength > 0U && descriptor.bitLength < 64U) {
        const auto minVal = -(1LL << (descriptor.bitLength - 1U));
        const auto maxVal = (1LL << (descriptor.bitLength - 1U)) - 1LL;
        return {
            "const auto " + helperName +
                " = [](const std::int64_t value) -> std::int64_t {",
            "if (value < " + std::to_string(minVal) + "LL) { return " +
                std::to_string(minVal) + "LL; }",
            "if (value > " + std::to_string(maxVal) + "LL) { return " +
                std::to_string(maxVal) + "LL; }",
            "return value;",
            "};",
        };
      }
      if (direction == ScalarBindingRenderDirection::Deserialize &&
          descriptor.bitLength > 0U && descriptor.bitLength < 64U) {
        const auto signMask =
            std::to_string((1ULL << (descriptor.bitLength - 1U))) + "ULL";
        return {
            "const auto " + helperName +
                " = [](const std::int64_t value) -> std::int64_t {",
            "const std::uint64_t raw = static_cast<std::uint64_t>(value) & " +
                maskLiteral + ";",
            "if ((raw & " + signMask + ") != 0ULL) {",
            "return static_cast<std::int64_t>(raw | (~" + maskLiteral + "));",
            "}",
            "return static_cast<std::int64_t>(raw);",
            "};",
        };
      }
      return {
          "const auto " + helperName +
              " = [](const std::int64_t value) -> std::int64_t {",
          "return value;",
          "};",
      };
    }
    case ScalarHelperKind::Float:
      return {
          "const auto " + helperName +
              " = [](const double value) -> double { return value; };",
      };
    }
  }

  switch (descriptor.kind) {
  case ScalarHelperKind::Unsigned: {
    if (direction == ScalarBindingRenderDirection::Serialize &&
        descriptor.castMode == CastMode::Saturated && descriptor.bitLength < 64U) {
      return {
          "let " + helperName + " = |value: u64| -> u64 {",
          "if value > " + maskLiteral + " { " + maskLiteral + " } else { value }",
          "};",
      };
    }
    if (descriptor.bitLength < 64U) {
      return {
          "let " + helperName + " = |value: u64| -> u64 {",
          "value & " + maskLiteral,
          "};",
      };
    }
    return {
        "let " + helperName + " = |value: u64| -> u64 {",
        "value",
        "};",
    };
  }
  case ScalarHelperKind::Signed: {
    if (direction == ScalarBindingRenderDirection::Serialize &&
        descriptor.castMode == CastMode::Saturated && descriptor.bitLength > 0U &&
        descriptor.bitLength < 64U) {
      const auto minVal = -(1LL << (descriptor.bitLength - 1U));
      const auto maxVal = (1LL << (descriptor.bitLength - 1U)) - 1LL;
      return {
          "let " + helperName + " = |value: i64| -> i64 {",
          "if value < " + std::to_string(minVal) + "i64 {",
          std::to_string(minVal) + "i64",
          "} else if value > " + std::to_string(maxVal) + "i64 {",
          std::to_string(maxVal) + "i64",
          "} else {",
          "value",
          "}",
          "};",
      };
    }
    if (direction == ScalarBindingRenderDirection::Deserialize &&
        descriptor.bitLength > 0U && descriptor.bitLength < 64U) {
      const auto signMask =
          std::to_string((1ULL << (descriptor.bitLength - 1U))) + "u64";
      return {
          "let " + helperName + " = |value: i64| -> i64 {",
          "let raw = (value as u64) & " + maskLiteral + ";",
          "if (raw & " + signMask + ") != 0u64 {",
          "(raw | (!" + maskLiteral + ")) as i64",
          "} else {",
          "raw as i64",
          "}",
          "};",
      };
    }
    return {
        "let " + helperName + " = |value: i64| -> i64 {",
        "value",
        "};",
    };
  }
  case ScalarHelperKind::Float:
    return {
        "let " + helperName + " = |value: f64| -> f64 { value };",
    };
  }
}

std::vector<std::string> renderArrayPrefixBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName,
    const std::uint32_t bits) {
  return renderUnionTagMaskBinding(language, helperName, bits);
}

std::vector<std::string> renderArrayValidateBinding(
    const HelperBindingRenderLanguage language, const std::string &helperName,
    const std::int64_t capacity) {
  if (language == HelperBindingRenderLanguage::Cpp) {
    return {
        "const auto " + helperName +
            " = [](const std::int64_t value) -> std::int8_t {",
        "return ((value < 0LL) || (value > " + std::to_string(capacity) +
            "LL)) ? static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH) : static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS);",
        "};",
    };
  }

  return {
      "let " + helperName + " = |value: i64| -> i8 {",
      "if (value < 0i64) || (value > " + std::to_string(capacity) + "i64) {",
      "-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH",
      "} else {",
      "crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS",
      "}",
      "};",
  };
}

std::vector<std::string> renderSectionHelperBindings(
    const SectionHelperBindingPlan &plan,
    const HelperBindingRenderLanguage language,
    const ScalarBindingRenderDirection scalarDirection,
    const std::function<std::string(const std::string &)> &helperNameResolver,
    const bool emitCapacityCheck) {
  std::vector<std::string> out;

  if (emitCapacityCheck && plan.capacityCheck) {
    appendLines(out, renderCapacityCheckBinding(
                         language, helperNameResolver(plan.capacityCheck->symbol),
                         plan.capacityCheck->requiredBits));
  }
  if (plan.unionTagValidate) {
    appendLines(out, renderUnionTagValidateBinding(
                         language,
                         helperNameResolver(plan.unionTagValidate->symbol),
                         plan.unionTagValidate->allowedTags));
  }
  if (plan.unionTagMask) {
    appendLines(out,
                renderUnionTagMaskBinding(
                    language, helperNameResolver(plan.unionTagMask->symbol),
                    plan.unionTagMask->bits));
  }
  for (const auto &binding : plan.scalarBindings) {
    appendLines(out, renderScalarBinding(
                         language, scalarDirection,
                         helperNameResolver(binding.symbol), binding.descriptor));
  }
  for (const auto &binding : plan.delimiterValidateBindings) {
    appendLines(out, renderDelimiterValidateBinding(
                         language, helperNameResolver(binding.symbol)));
  }
  for (const auto &binding : plan.arrayPrefixBindings) {
    appendLines(out,
                renderArrayPrefixBinding(
                    language, helperNameResolver(binding.symbol), binding.bits));
  }
  for (const auto &binding : plan.arrayValidateBindings) {
    appendLines(out, renderArrayValidateBinding(
                         language, helperNameResolver(binding.symbol),
                         binding.capacity));
  }

  return out;
}

} // namespace llvmdsdl
