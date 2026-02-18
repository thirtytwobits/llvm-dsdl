#ifndef LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H
#define LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace llvmdsdl {

struct LoweredFieldFacts final {
  std::optional<std::int64_t> stepIndex;
  std::optional<std::uint32_t> arrayLengthPrefixBits;
  std::string serArrayLengthPrefixHelper;
  std::string deserArrayLengthPrefixHelper;
  std::string arrayLengthValidateHelper;
  std::string delimiterValidateHelper;
  std::string serUnsignedHelper;
  std::string deserUnsignedHelper;
  std::string serSignedHelper;
  std::string deserSignedHelper;
  std::string serFloatHelper;
  std::string deserFloatHelper;
};

struct LoweredSectionFacts final {
  std::string capacityCheckHelper;
  std::optional<std::uint32_t> unionTagBits;
  std::string unionTagValidateHelper;
  std::string serUnionTagHelper;
  std::string deserUnionTagHelper;
  std::unordered_map<std::string, LoweredFieldFacts> fieldsByName;
};

using LoweredDefinitionFacts = std::unordered_map<std::string, LoweredSectionFacts>;
using LoweredFactsMap = std::unordered_map<std::string, LoweredDefinitionFacts>;

std::string loweredTypeKey(const std::string &name, std::uint32_t major,
                           std::uint32_t minor);

bool collectLoweredFactsFromMlir(const SemanticModule &semantic,
                                 mlir::ModuleOp module,
                                 DiagnosticEngine &diagnostics,
                                 const std::string &backendLabel,
                                 LoweredFactsMap *outFacts,
                                 bool optimizeLoweredSerDes = false);

const LoweredFieldFacts *
findLoweredFieldFacts(const LoweredSectionFacts *sectionFacts,
                      const std::string &fieldName);

std::optional<std::uint32_t>
loweredFieldArrayPrefixBits(const LoweredSectionFacts *sectionFacts,
                            const std::string &fieldName);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_MLIR_LOWERED_FACTS_H
