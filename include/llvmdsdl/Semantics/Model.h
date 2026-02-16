#ifndef LLVMDSDL_SEMANTICS_MODEL_H
#define LLVMDSDL_SEMANTICS_MODEL_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/BitLengthSet.h"
#include "llvmdsdl/Semantics/Evaluator.h"

#include <optional>
#include <string>
#include <vector>

namespace llvmdsdl {

struct SemanticOptions final {
  bool strict{true};
  bool compatMode{false};
};

enum class SemanticScalarCategory {
  Bool,
  Byte,
  Utf8,
  UnsignedInt,
  SignedInt,
  Float,
  Void,
  Composite,
};

struct SemanticTypeRef final {
  std::string fullName;
  std::vector<std::string> namespaceComponents;
  std::string shortName;
  std::uint32_t majorVersion{0};
  std::uint32_t minorVersion{0};
};

struct SemanticFieldType final {
  SemanticScalarCategory scalarCategory{SemanticScalarCategory::Void};
  CastMode castMode{CastMode::Saturated};
  std::uint32_t bitLength{0};
  ArrayKind arrayKind{ArrayKind::None};
  std::int64_t arrayCapacity{0}; // Resolved max element count for arrays.
  std::int64_t arrayLengthPrefixBits{0};
  std::int64_t alignmentBits{1};
  BitLengthSet bitLengthSet;
  std::optional<SemanticTypeRef> compositeType;
  bool compositeSealed{true};
  std::int64_t compositeExtentBits{0};
};

struct SemanticField final {
  std::string name;
  TypeExprAST type;
  bool isPadding{false};
  SemanticFieldType resolvedType;
  std::string sectionName;
  std::uint32_t unionOptionIndex{0};
  std::uint32_t unionTagBits{0};
};

struct SemanticConstant final {
  std::string name;
  TypeExprAST type;
  Value value;
};

struct SemanticSection final {
  bool isUnion{false};
  bool sealed{false};
  bool deprecated{false};
  std::optional<std::int64_t> extentBits;
  std::vector<SemanticField> fields;
  std::vector<SemanticConstant> constants;
  BitLengthSet offsetAtEnd;
  std::int64_t minBitLength{0};
  std::int64_t maxBitLength{0};
  bool fixedSize{true};
  std::int64_t serializationBufferSizeBits{0};
};

struct SemanticDefinition final {
  DiscoveredDefinition info;
  bool isService{false};
  SemanticSection request;
  std::optional<SemanticSection> response;
};

struct SemanticModule final {
  std::vector<SemanticDefinition> definitions;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_SEMANTICS_MODEL_H
