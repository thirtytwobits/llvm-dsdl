#ifndef LLVMDSDL_CODEGEN_SERDES_HELPER_DESCRIPTORS_H
#define LLVMDSDL_CODEGEN_SERDES_HELPER_DESCRIPTORS_H

#include "llvmdsdl/Semantics/Model.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvmdsdl {

struct CapacityCheckHelperDescriptor final {
  std::string symbol;
  std::int64_t requiredBits{0};
};

struct UnionTagValidateHelperDescriptor final {
  std::string symbol;
  std::vector<std::int64_t> allowedTags;
};

struct SharedSerDesHelperDescriptors final {
  std::optional<CapacityCheckHelperDescriptor> capacityCheck;
  std::optional<UnionTagValidateHelperDescriptor> unionTagValidate;
};

struct ArrayLengthHelperDescriptor final {
  std::string prefixSymbol;
  std::string validateSymbol;
  std::uint32_t prefixBits{0};
  std::int64_t capacity{0};
};

struct ScalarHelperSymbols final {
  std::string serUnsignedSymbol;
  std::string deserUnsignedSymbol;
  std::string serSignedSymbol;
  std::string deserSignedSymbol;
  std::string serFloatSymbol;
  std::string deserFloatSymbol;
};

enum class ScalarHelperKind {
  Unsigned,
  Signed,
  Float,
};

struct ScalarHelperDescriptor final {
  ScalarHelperKind kind{ScalarHelperKind::Unsigned};
  std::string serSymbol;
  std::string deserSymbol;
  std::uint32_t bitLength{0};
  CastMode castMode{CastMode::Truncated};
};

struct DelimiterValidateHelperDescriptor final {
  std::string symbol;
};

SharedSerDesHelperDescriptors buildSharedSerDesHelperDescriptors(
    const SemanticSection &section, const std::string &capacityCheckSymbol,
    const std::string &unionTagValidateSymbol);

std::optional<ArrayLengthHelperDescriptor> buildArrayLengthHelperDescriptor(
    const SemanticField &field, std::optional<std::uint32_t> prefixBitsOverride,
    const std::string &prefixSymbol, const std::string &validateSymbol);
std::optional<ArrayLengthHelperDescriptor> buildArrayLengthHelperDescriptor(
    const SemanticFieldType &type, std::optional<std::uint32_t> prefixBitsOverride,
    const std::string &prefixSymbol, const std::string &validateSymbol);

std::optional<ScalarHelperDescriptor> buildScalarHelperDescriptor(
    const SemanticField &field, const ScalarHelperSymbols &symbols);
std::optional<ScalarHelperDescriptor> buildScalarHelperDescriptor(
    const SemanticFieldType &type, const ScalarHelperSymbols &symbols);

std::optional<DelimiterValidateHelperDescriptor>
buildDelimiterValidateHelperDescriptor(const SemanticField &field,
                                       const std::string &symbol);
std::optional<DelimiterValidateHelperDescriptor>
buildDelimiterValidateHelperDescriptor(const SemanticFieldType &type,
                                       const std::string &symbol);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_SERDES_HELPER_DESCRIPTORS_H
