#include "llvmdsdl/CodeGen/RustEmitter.h"
#include "llvmdsdl/CodeGen/ArrayWirePlan.h"
#include "llvmdsdl/CodeGen/HelperBindingRender.h"
#include "llvmdsdl/CodeGen/HelperSymbolResolver.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/SerDesHelperDescriptors.h"
#include "llvmdsdl/CodeGen/SerDesStatementPlan.h"
#include "llvmdsdl/CodeGen/TypeStorage.h"
#include "llvmdsdl/CodeGen/WireLayoutFacts.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace llvmdsdl {
namespace {

bool isRustKeyword(const std::string &name) {
  static const std::set<std::string> kKeywords = {
      "as",       "break",    "const",   "continue", "crate", "else",
      "enum",     "extern",   "false",   "fn",       "for",   "if",
      "impl",     "in",       "let",     "loop",     "match", "mod",
      "move",     "mut",      "pub",     "ref",      "return", "self",
      "Self",     "static",   "struct",  "super",    "trait", "true",
      "type",     "unsafe",   "use",     "where",    "while", "async",
      "await",    "dyn",      "abstract", "become",   "box",   "do",
      "final",    "macro",    "override", "priv",     "try",   "typeof",
      "unsized",  "virtual",  "yield"};
  return kKeywords.contains(name);
}

std::string sanitizeRustIdent(std::string name) {
  if (name.empty()) {
    return "_";
  }
  for (char &c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      c = '_';
    }
  }
  if (std::isdigit(static_cast<unsigned char>(name.front()))) {
    name.insert(name.begin(), '_');
  }
  if (isRustKeyword(name)) {
    name += '_';
  }
  return name;
}

std::string toSnakeCase(const std::string &in) {
  std::string out;
  out.reserve(in.size() + 8);

  bool prevUnderscore = false;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const char c = in[i];
    const char prev = (i > 0) ? in[i - 1] : '\0';
    const char next = (i + 1 < in.size()) ? in[i + 1] : '\0';
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      if (!out.empty() && !prevUnderscore) {
        out.push_back('_');
        prevUnderscore = true;
      }
      continue;
    }

    if (std::isupper(static_cast<unsigned char>(c))) {
      const bool boundary =
          std::islower(static_cast<unsigned char>(prev)) ||
          (std::isupper(static_cast<unsigned char>(prev)) &&
           std::islower(static_cast<unsigned char>(next)));
      if (!out.empty() && !prevUnderscore && boundary) {
        out.push_back('_');
      }
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
      prevUnderscore = false;
    } else {
      out.push_back(c);
      prevUnderscore = (c == '_');
    }
  }

  if (out.empty()) {
    out = "_";
  }
  if (std::isdigit(static_cast<unsigned char>(out.front()))) {
    out.insert(out.begin(), '_');
  }
  out = sanitizeRustIdent(out);
  return out;
}

std::string toUpperSnake(const std::string &in) {
  auto s = toSnakeCase(in);
  for (char &c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string unsignedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "u8";
  case 16:
    return "u16";
  case 32:
    return "u32";
  default:
    return "u64";
  }
}

std::string signedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "i8";
  case 16:
    return "i16";
  case 32:
    return "i32";
  default:
    return "i64";
  }
}

std::string rustConstValue(const Value &value) {
  if (const auto *b = std::get_if<bool>(&value.data)) {
    return *b ? "true" : "false";
  }
  if (const auto *r = std::get_if<Rational>(&value.data)) {
    if (r->isInteger()) {
      return std::to_string(r->asInteger().value());
    }
    std::ostringstream out;
    out << "(" << r->numerator() << "f64 / " << r->denominator() << "f64)";
    return out.str();
  }
  if (const auto *s = std::get_if<std::string>(&value.data)) {
    std::string escaped;
    escaped.reserve(s->size() + 2);
    escaped.push_back('"');
    for (char c : *s) {
      if (c == '\\' || c == '"') {
        escaped.push_back('\\');
      }
      escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
  }
  return value.str();
}

llvm::Error writeFile(const std::filesystem::path &p, llvm::StringRef content) {
  std::error_code ec;
  llvm::raw_fd_ostream os(p.string(), ec, llvm::sys::fs::OF_Text);
  if (ec) {
    return llvm::createStringError(ec, "failed to open %s", p.string().c_str());
  }
  os << content;
  os.close();
  return llvm::Error::success();
}

void emitLine(std::ostringstream &out, const int indent, const std::string &line) {
  out << std::string(static_cast<std::size_t>(indent) * 4U, ' ') << line << '\n';
}

class EmitterContext final {
public:
  explicit EmitterContext(const SemanticModule &semantic) {
    for (const auto &def : semantic.definitions) {
      byKey_.emplace(loweredTypeKey(def.info.fullName, def.info.majorVersion,
                             def.info.minorVersion),
                     &def);
    }
  }

  const SemanticDefinition *find(const SemanticTypeRef &ref) const {
    const auto it = byKey_.find(
        loweredTypeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
    if (it == byKey_.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::string rustModuleName(const DiscoveredDefinition &info) const {
    return toSnakeCase(info.shortName) + "_" + std::to_string(info.majorVersion) +
           "_" + std::to_string(info.minorVersion);
  }

  std::string rustTypeName(const DiscoveredDefinition &info) const {
    std::string out;
    for (std::size_t i = 0; i < info.namespaceComponents.size(); ++i) {
      if (!out.empty()) {
        out += "_";
      }
      out += sanitizeRustIdent(info.namespaceComponents[i]);
    }
    if (!out.empty()) {
      out += "_";
    }
    out += sanitizeRustIdent(info.shortName);
    out += "_" + std::to_string(info.majorVersion) + "_" +
           std::to_string(info.minorVersion);
    return sanitizeRustIdent(out);
  }

  std::string rustTypeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return rustTypeName(def->info);
    }

    DiscoveredDefinition tmp;
    tmp.shortName = ref.shortName;
    tmp.namespaceComponents = ref.namespaceComponents;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    return rustTypeName(tmp);
  }

  std::string rustTypePath(const SemanticTypeRef &ref) const {
    std::ostringstream out;
    out << "crate";
    for (const auto &ns : ref.namespaceComponents) {
      out << "::" << sanitizeRustIdent(ns);
    }

    if (const auto *def = find(ref)) {
      out << "::" << rustModuleName(def->info) << "::" << rustTypeName(def->info);
      return out.str();
    }

    DiscoveredDefinition tmp;
    tmp.shortName = ref.shortName;
    tmp.namespaceComponents = ref.namespaceComponents;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    out << "::" << rustModuleName(tmp) << "::" << rustTypeName(tmp);
    return out.str();
  }

private:
  std::unordered_map<std::string, const SemanticDefinition *> byKey_;
};

std::string rustFieldBaseType(const SemanticFieldType &type,
                              const EmitterContext &ctx) {
  switch (type.scalarCategory) {
  case SemanticScalarCategory::Bool:
    return "bool";
  case SemanticScalarCategory::Byte:
  case SemanticScalarCategory::Utf8:
  case SemanticScalarCategory::UnsignedInt:
    return unsignedStorageType(type.bitLength);
  case SemanticScalarCategory::SignedInt:
    return signedStorageType(type.bitLength);
  case SemanticScalarCategory::Float:
    return type.bitLength == 64 ? "f64" : "f32";
  case SemanticScalarCategory::Void:
    return "u8";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      return ctx.rustTypeName(*type.compositeType);
    }
    return "u8";
  }
  return "u8";
}

std::string rustFieldType(const SemanticFieldType &type, const EmitterContext &ctx) {
  const auto base = rustFieldBaseType(type, ctx);
  if (type.arrayKind == ArrayKind::None) {
    return base;
  }
  return "crate::dsdl_runtime::DsdlVec<" + base + ">";
}

std::string defaultExpr(const SemanticFieldType &type, const EmitterContext &ctx) {
  if (type.arrayKind != ArrayKind::None) {
    return "crate::dsdl_runtime::DsdlVec::new()";
  }

  switch (type.scalarCategory) {
  case SemanticScalarCategory::Bool:
    return "false";
  case SemanticScalarCategory::Byte:
  case SemanticScalarCategory::Utf8:
  case SemanticScalarCategory::UnsignedInt:
    return "0";
  case SemanticScalarCategory::SignedInt:
    return "0";
  case SemanticScalarCategory::Float:
    return type.bitLength == 64 ? "0.0f64" : "0.0f32";
  case SemanticScalarCategory::Void:
    return "0";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      return ctx.rustTypeName(*type.compositeType) + "::default()";
    }
    return "0";
  }
  return "0";
}

class FunctionBodyEmitter final {
public:
  explicit FunctionBodyEmitter(const EmitterContext &ctx) : ctx_(ctx) {}

  void emitSerialize(std::ostringstream &out, const std::string &typeName,
                     const SemanticSection &section,
                     const LoweredSectionFacts *const sectionFacts) {
    (void)typeName;
    emitLine(out, 1,
             "pub fn serialize(&self, buffer: &mut [u8]) -> core::result::Result<usize, i8> {");
    emitLine(out, 2, "let mut offset_bits: usize = 0;");
    emitSerializeMlirHelperBindings(out, section, 2, sectionFacts);
    const auto fieldPlan = buildSectionStatementPlan(section, sectionFacts);
    if (const auto capacityHelperSymbol =
            resolveSectionCapacityCheckHelperSymbol(sectionFacts);
        !capacityHelperSymbol.empty()) {
      const auto capacityHelper = helperBindingName(capacityHelperSymbol);
      emitLine(out, 2,
               "let _err_capacity = " + capacityHelper +
                   "(buffer.len().saturating_mul(8) as i64);");
      emitLine(out, 2,
               "if _err_capacity != crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS {");
      emitLine(out, 3, "return Err(_err_capacity);");
      emitLine(out, 2, "}");
    } else {
      emitLine(out, 2,
               "if buffer.len().saturating_mul(8) < Self::SERIALIZATION_BUFFER_SIZE_BYTES.saturating_mul(8) {");
      emitLine(out, 3,
               "return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL);");
      emitLine(out, 2, "}");
    }

    if (section.isUnion) {
      emitSerializeUnion(out, section, fieldPlan.unionBranches, 2, sectionFacts);
    } else {
      for (const auto &step : fieldPlan.orderedFields) {
        const auto &field = *step.field;
        emitAlignSerialize(out, field.resolvedType.alignmentBits, 2);
        if (field.isPadding) {
          emitSerializePadding(out, field.resolvedType, 2);
        } else {
          const auto fieldRef = "self." + sanitizeRustIdent(field.name);
          emitSerializeAny(out, field.resolvedType, fieldRef, 2,
                           step.arrayLengthPrefixBits, step.fieldFacts);
        }
      }
    }

    emitAlignSerialize(out, 8, 2);
    emitLine(out, 2, "Ok(offset_bits / 8)");
    emitLine(out, 1, "}");
  }

  void emitDeserialize(std::ostringstream &out, const std::string &typeName,
                       const SemanticSection &section,
                       const LoweredSectionFacts *const sectionFacts) {
    (void)typeName;
    emitLine(out, 1,
             "pub fn deserialize(&mut self, buffer: &[u8]) -> core::result::Result<usize, i8> {");
    emitLine(out, 2, "let capacity_bytes = buffer.len();");
    emitLine(out, 2, "let capacity_bits = capacity_bytes.saturating_mul(8);\n");
    emitLine(out, 2, "let mut offset_bits: usize = 0;");
    emitDeserializeMlirHelperBindings(out, section, 2, sectionFacts);
    const auto fieldPlan = buildSectionStatementPlan(section, sectionFacts);

    if (section.isUnion) {
      emitDeserializeUnion(out, section, fieldPlan.unionBranches, 2,
                           sectionFacts);
    } else {
      for (const auto &step : fieldPlan.orderedFields) {
        const auto &field = *step.field;
        emitAlignDeserialize(out, field.resolvedType.alignmentBits, 2);
        if (field.isPadding) {
          emitDeserializePadding(out, field.resolvedType, 2);
        } else {
          const auto fieldRef = "self." + sanitizeRustIdent(field.name);
          emitDeserializeAny(out, field.resolvedType, fieldRef, 2,
                             step.arrayLengthPrefixBits, step.fieldFacts);
        }
      }
    }

    emitAlignDeserialize(out, 8, 2);
    emitLine(out, 2,
             "Ok(crate::dsdl_runtime::choose_min(offset_bits, capacity_bits) / 8)");
    emitLine(out, 1, "}");
    emitLine(out, 0, "");
    emitLine(out, 1,
             "pub fn deserialize_with_consumed(&mut self, buffer: &[u8]) -> (i8, usize) {");
    emitLine(out, 2, "match self.deserialize(buffer) {");
    emitLine(out, 3, "Ok(consumed) => (0, consumed),");
    emitLine(out, 3, "Err(rc) => (rc, buffer.len()),");
    emitLine(out, 2, "}");
    emitLine(out, 1, "}");
  }

private:
  const EmitterContext &ctx_;
  std::size_t id_{0};

  std::string nextName(const std::string &prefix) {
    return "_" + prefix + std::to_string(id_++) + "_";
  }

  std::string helperBindingName(const std::string &helperSymbol) const {
    return "mlir_" + sanitizeRustIdent(helperSymbol);
  }

  void emitSerializeMlirHelperBindings(
      std::ostringstream &out, const SemanticSection &section, const int indent,
      const LoweredSectionFacts *const sectionFacts) {
    const auto plan = buildSectionHelperBindingPlan(
        section, sectionFacts, HelperBindingDirection::Serialize);
    for (const auto &line : renderSectionHelperBindings(
             plan, HelperBindingRenderLanguage::Rust,
             ScalarBindingRenderDirection::Serialize,
             [this](const std::string &symbol) {
               return helperBindingName(symbol);
             },
             /*emitCapacityCheck=*/true)) {
      emitLine(out, indent, line);
    }
  }

  void emitDeserializeMlirHelperBindings(
      std::ostringstream &out, const SemanticSection &section, const int indent,
      const LoweredSectionFacts *const sectionFacts) {
    const auto plan = buildSectionHelperBindingPlan(
        section, sectionFacts, HelperBindingDirection::Deserialize);
    for (const auto &line : renderSectionHelperBindings(
             plan, HelperBindingRenderLanguage::Rust,
             ScalarBindingRenderDirection::Deserialize,
             [this](const std::string &symbol) {
               return helperBindingName(symbol);
             },
             /*emitCapacityCheck=*/false)) {
      emitLine(out, indent, line);
    }
  }

  void emitAlignSerialize(std::ostringstream &out, const std::int64_t alignmentBits,
                          const int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             "while (offset_bits % " + std::to_string(alignmentBits) +
                 "usize) != 0 {");
    emitLine(out, indent + 1,
             "let " + err +
                 " = crate::dsdl_runtime::set_bit(buffer, offset_bits, false);");
    emitLine(out, indent + 1,
             "if " + err + " < 0 { return Err(" + err + "); }");
    emitLine(out, indent + 1, "offset_bits += 1;");
    emitLine(out, indent, "}");
  }

  void emitAlignDeserialize(std::ostringstream &out, const std::int64_t alignmentBits,
                            const int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    emitLine(out, indent,
             "offset_bits = (offset_bits + " + std::to_string(alignmentBits - 1) +
                 ") & !" + std::to_string(alignmentBits - 1) + "usize;");
  }

  void emitSerializePadding(std::ostringstream &out, const SemanticFieldType &type,
                            const int indent) {
    if (type.bitLength == 0) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             "let " + err + " = crate::dsdl_runtime::set_uxx(buffer, offset_bits, 0, " +
                 std::to_string(type.bitLength) + "u8);");
    emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(type.bitLength) + ";");
  }

  void emitDeserializePadding(std::ostringstream &out,
                              const SemanticFieldType &type,
                              const int indent) {
    if (type.bitLength == 0) {
      return;
    }
    emitLine(out, indent,
             "offset_bits += " + std::to_string(type.bitLength) + ";");
  }

  void emitSerializeUnion(std::ostringstream &out, const SemanticSection &section,
                          const std::vector<PlannedFieldStep> &unionBranches,
                          const int indent,
                          const LoweredSectionFacts *const sectionFacts) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    const auto validateHelperSymbol =
        resolveSectionUnionTagValidateHelperSymbol(sectionFacts);
    const auto validateHelper = validateHelperSymbol.empty()
                                    ? std::string{}
                                    : helperBindingName(validateHelperSymbol);
    if (!validateHelper.empty()) {
      emitLine(out, indent,
               "let _err_union_tag = " + validateHelper + "(self._tag_ as i64);");
      emitLine(out, indent,
               "if _err_union_tag != crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS { return Err(_err_union_tag); }");
    }
    const auto tagHelperSymbol = resolveSectionUnionTagMaskHelperSymbol(
        sectionFacts, HelperBindingDirection::Serialize);
    const auto tagHelper = tagHelperSymbol.empty() ? std::string{}
                                                   : helperBindingName(tagHelperSymbol);
    std::string tagExpr = "self._tag_ as u64";
    if (!tagHelper.empty()) {
      tagExpr = tagHelper + "(" + tagExpr + ")";
    }

    const auto tagErr = nextName("err");
    emitLine(out, indent,
             "let " + tagErr + " = crate::dsdl_runtime::set_uxx(buffer, offset_bits, " +
                 tagExpr + ", " +
                 std::to_string(tagBits) + "u8);");
    emitLine(out, indent, "if " + tagErr + " < 0 { return Err(" + tagErr + "); }");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + ";");

    emitLine(out, indent, "match self._tag_ {");
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      emitLine(out, indent + 1,
               std::to_string(field.unionOptionIndex) + " => {");
      emitAlignSerialize(out, field.resolvedType.alignmentBits, indent + 2);
      emitSerializeAny(out, field.resolvedType,
                       "self." + sanitizeRustIdent(field.name), indent + 2,
                       step.arrayLengthPrefixBits, step.fieldFacts);
      emitLine(out, indent + 1, "}");
    }
    emitLine(out, indent + 1,
             "_ => return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG),");
    emitLine(out, indent, "}");
  }

  void emitDeserializeUnion(std::ostringstream &out,
                            const SemanticSection &section,
                            const std::vector<PlannedFieldStep> &unionBranches,
                            const int indent,
                            const LoweredSectionFacts *const sectionFacts) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    const auto rawTag = nextName("tag_raw");
    emitLine(out, indent,
             "let " + rawTag + " = crate::dsdl_runtime::get_u64(buffer, offset_bits, " +
                 std::to_string(tagBits) + "u8);");
    const auto tagHelperSymbol = resolveSectionUnionTagMaskHelperSymbol(
        sectionFacts, HelperBindingDirection::Deserialize);
    const auto tagHelper = tagHelperSymbol.empty() ? std::string{}
                                                   : helperBindingName(tagHelperSymbol);
    std::string tagExpr = rawTag;
    if (!tagHelper.empty()) {
      tagExpr = tagHelper + "(" + tagExpr + ")";
    }
    emitLine(out, indent,
             "self._tag_ = (" + tagExpr + ") as u8;");
    const auto validateHelperSymbol =
        resolveSectionUnionTagValidateHelperSymbol(sectionFacts);
    const auto validateHelper = validateHelperSymbol.empty()
                                    ? std::string{}
                                    : helperBindingName(validateHelperSymbol);
    if (!validateHelper.empty()) {
      emitLine(out, indent,
               "let _err_union_tag = " + validateHelper + "(self._tag_ as i64);");
      emitLine(out, indent,
               "if _err_union_tag != crate::dsdl_runtime::DSDL_RUNTIME_SUCCESS { return Err(_err_union_tag); }");
    }
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + ";");

    emitLine(out, indent, "match self._tag_ {");
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      emitLine(out, indent + 1,
               std::to_string(field.unionOptionIndex) + " => {");
      emitAlignDeserialize(out, field.resolvedType.alignmentBits, indent + 2);
      emitDeserializeAny(out, field.resolvedType,
                         "self." + sanitizeRustIdent(field.name), indent + 2,
                         step.arrayLengthPrefixBits, step.fieldFacts);
      emitLine(out, indent + 1, "}");
    }
    emitLine(out, indent + 1,
             "_ => return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG),");
    emitLine(out, indent, "}");
  }

  void emitSerializeAny(std::ostringstream &out, const SemanticFieldType &type,
                        const std::string &expr, const int indent,
                        const std::optional<std::uint32_t>
                            arrayLengthPrefixBitsOverride = std::nullopt,
                        const LoweredFieldFacts *const fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitSerializeArray(out, type, expr, indent,
                         arrayLengthPrefixBitsOverride, fieldFacts);
      return;
    }
    emitSerializeScalar(out, type, expr, indent, fieldFacts);
  }

  void emitDeserializeAny(std::ostringstream &out, const SemanticFieldType &type,
                          const std::string &expr, const int indent,
                          const std::optional<std::uint32_t>
                              arrayLengthPrefixBitsOverride = std::nullopt,
                          const LoweredFieldFacts *const fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitDeserializeArray(out, type, expr, indent,
                           arrayLengthPrefixBitsOverride, fieldFacts);
      return;
    }
    emitDeserializeScalar(out, type, expr, indent, fieldFacts);
  }

  void emitSerializeScalar(std::ostringstream &out, const SemanticFieldType &type,
                           const std::string &expr, const int indent,
                           const LoweredFieldFacts *const fieldFacts) {
    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool: {
      const auto err = nextName("err");
      emitLine(out, indent,
               "let " + err + " = crate::dsdl_runtime::set_bit(buffer, offset_bits, " +
                   expr + ");");
      emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
      emitLine(out, indent, "offset_bits += 1;");
      break;
    }
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      std::string valueExpr = expr + " as u64";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        valueExpr = helper + "(" + valueExpr + ")";
      } else if (type.castMode == CastMode::Saturated) {
        if (const auto maxVal = resolveUnsignedSaturationMax(type.bitLength)) {
          const auto sat = nextName("sat");
          emitLine(out, indent, "let mut " + sat + " = " + valueExpr + ";");
          emitLine(out, indent,
                   "if " + sat + " > " + std::to_string(*maxVal) + "u64 { " +
                       sat + " = " + std::to_string(*maxVal) + "u64; }");
          valueExpr = sat;
        }
      }

      const auto err = nextName("err");
      emitLine(out, indent,
               "let " + err + " = crate::dsdl_runtime::set_uxx(buffer, offset_bits, " +
                   valueExpr + ", " + std::to_string(type.bitLength) + "u8);");
      emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      std::string valueExpr = expr + " as i64";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        valueExpr = helper + "(" + valueExpr + ")";
      } else if (type.castMode == CastMode::Saturated) {
        if (const auto range = resolveSignedSaturationRange(type.bitLength)) {
          const auto sat = nextName("sat");
          emitLine(out, indent, "let mut " + sat + " = " + valueExpr + ";");
          emitLine(out, indent,
                   "if " + sat + " < " + std::to_string(range->first) +
                       "i64 { " + sat + " = " + std::to_string(range->first) +
                       "i64; }");
          emitLine(out, indent,
                   "if " + sat + " > " + std::to_string(range->second) +
                       "i64 { " + sat + " = " +
                       std::to_string(range->second) + "i64; }");
          valueExpr = sat;
        }
      }

      const auto err = nextName("err");
      emitLine(out, indent,
               "let " + err + " = crate::dsdl_runtime::set_ixx(buffer, offset_bits, " +
                   valueExpr + ", " + std::to_string(type.bitLength) + "u8);");
      emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::Float: {
      const auto err = nextName("err");
      std::string normalizedExpr = expr + " as f64";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        normalizedExpr = helper + "(" + normalizedExpr + ")";
      }
      std::string setCall;
      if (type.bitLength == 16U) {
        setCall = "crate::dsdl_runtime::set_f16(buffer, offset_bits, " +
                  normalizedExpr + " as f32)";
      } else if (type.bitLength == 32U) {
        setCall = "crate::dsdl_runtime::set_f32(buffer, offset_bits, " +
                  normalizedExpr + " as f32)";
      } else {
        setCall = "crate::dsdl_runtime::set_f64(buffer, offset_bits, " +
                  normalizedExpr + " as f64)";
      }
      emitLine(out, indent, "let " + err + " = " + setCall + ";");
      emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::Void:
      emitSerializePadding(out, type, indent);
      break;
    case SemanticScalarCategory::Composite:
      emitSerializeComposite(out, type, expr, indent, fieldFacts);
      break;
    }
  }

  void emitDeserializeScalar(std::ostringstream &out,
                             const SemanticFieldType &type,
                             const std::string &expr, const int indent,
                             const LoweredFieldFacts *const fieldFacts) {
    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool:
      emitLine(out, indent,
               expr + " = crate::dsdl_runtime::get_bit(buffer, offset_bits);");
      emitLine(out, indent, "offset_bits += 1;");
      break;
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      const std::string getter =
          "get_u" + std::string(scalarWidthSuffix(type.bitLength));
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto raw = nextName("raw");
        emitLine(out, indent,
                 "let " + raw + " = crate::dsdl_runtime::" + getter +
                     "(buffer, offset_bits, " + std::to_string(type.bitLength) +
                     "u8) as u64;");
        emitLine(out, indent,
                 expr + " = " + helper + "(" + raw + ") as " +
                     unsignedStorageType(type.bitLength) + ";");
      } else {
        emitLine(out, indent,
                 expr + " = crate::dsdl_runtime::" + getter +
                     "(buffer, offset_bits, " + std::to_string(type.bitLength) +
                     "u8) as " + unsignedStorageType(type.bitLength) + ";");
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      const std::string getter =
          "get_i" + std::string(scalarWidthSuffix(type.bitLength));
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto raw = nextName("raw");
        emitLine(out, indent,
                 "let " + raw + " = crate::dsdl_runtime::get_u64(buffer, offset_bits, " +
                     std::to_string(type.bitLength) + "u8) as i64;");
        emitLine(out, indent,
                 expr + " = " + helper + "(" + raw + ") as " +
                     signedStorageType(type.bitLength) + ";");
      } else {
        emitLine(out, indent,
                 expr + " = crate::dsdl_runtime::" + getter +
                     "(buffer, offset_bits, " + std::to_string(type.bitLength) +
                     "u8) as " + signedStorageType(type.bitLength) + ";");
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::Float: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (type.bitLength == 16U) {
        if (!helper.empty()) {
          emitLine(out, indent,
                   expr + " = " + helper +
                       "(crate::dsdl_runtime::get_f16(buffer, offset_bits) as f64) as f32;");
        } else {
          emitLine(out, indent,
                   expr + " = crate::dsdl_runtime::get_f16(buffer, offset_bits);");
        }
      } else if (type.bitLength == 32U) {
        if (!helper.empty()) {
          emitLine(out, indent,
                   expr + " = " + helper +
                       "(crate::dsdl_runtime::get_f32(buffer, offset_bits) as f64) as f32;");
        } else {
          emitLine(out, indent,
                   expr + " = crate::dsdl_runtime::get_f32(buffer, offset_bits);");
        }
      } else {
        if (!helper.empty()) {
          emitLine(out, indent,
                   expr + " = " + helper +
                       "(crate::dsdl_runtime::get_f64(buffer, offset_bits) as f64) as f64;");
        } else {
          emitLine(out, indent,
                   expr + " = crate::dsdl_runtime::get_f64(buffer, offset_bits);");
        }
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + ";");
      break;
    }
    case SemanticScalarCategory::Void:
      emitDeserializePadding(out, type, indent);
      break;
    case SemanticScalarCategory::Composite:
      emitDeserializeComposite(out, type, expr, indent, fieldFacts);
      break;
    }
  }

  void emitSerializeArray(std::ostringstream &out, const SemanticFieldType &type,
                          const std::string &expr, const int indent,
                          const std::optional<std::uint32_t>
                              arrayLengthPrefixBitsOverride,
                          const LoweredFieldFacts *const fieldFacts) {
    const auto arrayPlan = buildArrayWirePlan(
        type, fieldFacts, arrayLengthPrefixBitsOverride,
        HelperBindingDirection::Serialize);
    const bool variable = arrayPlan.variable;
    const auto prefixBits = arrayPlan.prefixBits;
    const auto &arrayDescriptor = arrayPlan.descriptor;

    if (type.arrayKind == ArrayKind::Fixed) {
      emitLine(out, indent,
               "if " + expr + ".len() != " + std::to_string(type.arrayCapacity) +
                   "usize { return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH); }");
    }

    if (variable) {
      std::string validateHelper;
      if (arrayDescriptor && !arrayDescriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayDescriptor->validateSymbol);
      }
      if (!validateHelper.empty()) {
        const auto validateRc = nextName("len_rc");
        emitLine(out, indent,
                 "let " + validateRc + " = " + validateHelper +
                     "(" + expr + ".len() as i64);");
        emitLine(out, indent,
                 "if " + validateRc + " < 0 { return Err(" + validateRc + "); }");
      } else {
        emitLine(out, indent,
                 "if " + expr + ".len() > " +
                     std::to_string(type.arrayCapacity) +
                     "usize { return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH); }");
      }
      std::string prefixExpr = expr + ".len() as u64";
      std::string serPrefixHelper;
      if (arrayDescriptor && !arrayDescriptor->prefixSymbol.empty()) {
        serPrefixHelper = helperBindingName(arrayDescriptor->prefixSymbol);
      }
      if (!serPrefixHelper.empty()) {
        prefixExpr = serPrefixHelper + "(" + prefixExpr + ")";
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "let " + err + " = crate::dsdl_runtime::set_uxx(buffer, offset_bits, " +
                   prefixExpr + ", " +
                   std::to_string(prefixBits) + "u8);");
      emitLine(out, indent, "if " + err + " < 0 { return Err(" + err + "); }");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(prefixBits) + ";");
    }

    const auto index = nextName("index");
    const auto count = variable ? (expr + ".len()")
                                : std::to_string(type.arrayCapacity) + "usize";

    emitLine(out, indent,
             "for " + index + " in 0.." + count + " {");

    const auto itemType = arrayElementType(type);

    if (itemType.scalarCategory == SemanticScalarCategory::Composite) {
      emitSerializeScalar(out, itemType, expr + "[" + index + "]", indent + 1,
                          fieldFacts);
    } else {
      emitSerializeScalar(out, itemType, expr + "[" + index + "]", indent + 1,
                          fieldFacts);
    }
    emitLine(out, indent, "}");
  }

  void emitDeserializeArray(std::ostringstream &out,
                            const SemanticFieldType &type,
                            const std::string &expr, const int indent,
                            const std::optional<std::uint32_t>
                                arrayLengthPrefixBitsOverride,
                            const LoweredFieldFacts *const fieldFacts) {
    const auto arrayPlan = buildArrayWirePlan(
        type, fieldFacts, arrayLengthPrefixBitsOverride,
        HelperBindingDirection::Deserialize);
    const bool variable = arrayPlan.variable;
    const auto prefixBits = arrayPlan.prefixBits;
    const auto &arrayDescriptor = arrayPlan.descriptor;
    const auto count = nextName("count");

    if (variable) {
      const auto rawCount = nextName("count_raw");
      emitLine(out, indent,
               "let " + rawCount + " = crate::dsdl_runtime::get_u64(buffer, offset_bits, " +
                   std::to_string(prefixBits) + "u8);");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(prefixBits) + ";");
      std::string countExpr = rawCount + " as usize";
      std::string deserPrefixHelper;
      if (arrayDescriptor && !arrayDescriptor->prefixSymbol.empty()) {
        deserPrefixHelper = helperBindingName(arrayDescriptor->prefixSymbol);
      }
      if (!deserPrefixHelper.empty()) {
        countExpr = deserPrefixHelper + "(" + rawCount + ") as usize";
      }
      emitLine(out, indent, "let " + count + " = " + countExpr + ";");
      std::string validateHelper;
      if (arrayDescriptor && !arrayDescriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayDescriptor->validateSymbol);
      }
      if (!validateHelper.empty()) {
        const auto validateRc = nextName("len_rc");
        emitLine(out, indent,
                 "let " + validateRc + " = " + validateHelper + "(" + count + " as i64);");
        emitLine(out, indent,
                 "if " + validateRc + " < 0 { return Err(" + validateRc + "); }");
      } else {
        emitLine(out, indent,
                 "if " + count + " > " + std::to_string(type.arrayCapacity) +
                     "usize { return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH); }");
      }
      emitLine(out, indent, expr + ".clear();");
      emitLine(out, indent, expr + ".reserve(" + count + ");");
    } else {
      emitLine(out, indent, expr + ".clear();");
      emitLine(out, indent,
               expr + ".reserve(" + std::to_string(type.arrayCapacity) + "usize);");
      emitLine(out, indent,
               "let " + count + " = " + std::to_string(type.arrayCapacity) +
                   "usize;");
    }

    const auto index = nextName("index");
    emitLine(out, indent,
             "for " + index + " in 0.." + count + " {");

    const auto itemType = arrayElementType(type);

    const auto itemVar = nextName("item");
    emitLine(out, indent + 1,
             "let mut " + itemVar + " = " + defaultExpr(itemType, ctx_) + ";");
    emitDeserializeScalar(out, itemType, itemVar, indent + 1, fieldFacts);
    emitLine(out, indent + 1, expr + ".push(" + itemVar + ");");
    emitLine(out, indent, "}");
  }

  void emitSerializeComposite(std::ostringstream &out,
                              const SemanticFieldType &type,
                              const std::string &expr, const int indent,
                              const LoweredFieldFacts *const fieldFacts) {
    const auto sizeVar = nextName("size_bytes");
    const auto errVar = nextName("err");

    if (!type.compositeSealed) {
      emitLine(out, indent, "offset_bits += 32;  // Delimiter header");
    }

    emitLine(out, indent,
             "let mut " + sizeVar + " = " +
                 std::to_string((type.bitLengthSet.max() + 7) / 8) + "usize;");
    if (!type.compositeSealed) {
      emitLine(out, indent,
               "let _remaining = buffer.len().saturating_sub(crate::dsdl_runtime::choose_min(offset_bits / 8, buffer.len()));");
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto validateRc = nextName("rc");
        emitLine(out, indent,
                 "let " + validateRc + " = " + helper +
                     "(" + sizeVar + " as i64, _remaining as i64);");
        emitLine(out, indent,
                 "if " + validateRc + " < 0 { return Err(" + validateRc + "); }");
      } else {
        emitLine(out, indent,
                 "if " + sizeVar +
                     " > _remaining { return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER); }");
      }
    }
    emitLine(out, indent,
             "let _start = crate::dsdl_runtime::choose_min(offset_bits / 8, buffer.len());");
    emitLine(out, indent,
             "let _end = _start.saturating_add(" + sizeVar + ").min(buffer.len());");
    emitLine(out, indent,
             "let " + errVar + " = " + expr + ".serialize(&mut buffer[_start.._end]);");
    emitLine(out, indent, "match " + errVar + " {");
    emitLine(out, indent + 1, "Ok(v) => " + sizeVar + " = v,");
    emitLine(out, indent + 1, "Err(e) => return Err(e),");
    emitLine(out, indent, "}");

    if (!type.compositeSealed) {
      const auto hdrErr = nextName("err");
      emitLine(out, indent,
               "let " + hdrErr + " = crate::dsdl_runtime::set_uxx(buffer, offset_bits - 32, " +
                   sizeVar + " as u64, 32u8);");
      emitLine(out, indent, "if " + hdrErr + " < 0 { return Err(" + hdrErr + "); }");
    }

    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8;");
  }

  void emitDeserializeComposite(std::ostringstream &out,
                                const SemanticFieldType &type,
                                const std::string &expr, const int indent,
                                const LoweredFieldFacts *const fieldFacts) {
    const auto sizeVar = nextName("size_bytes");

    if (!type.compositeSealed) {
      emitLine(out, indent,
               "let " + sizeVar + " = crate::dsdl_runtime::get_u32(buffer, offset_bits, 32u8) as usize;");
      emitLine(out, indent, "offset_bits += 32;");
      emitLine(out, indent,
               "let _remaining = capacity_bytes.saturating_sub(crate::dsdl_runtime::choose_min(offset_bits / 8, capacity_bytes));");
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto validateRc = nextName("rc");
        emitLine(out, indent,
                 "let " + validateRc + " = " + helper +
                     "(" + sizeVar + " as i64, _remaining as i64);");
        emitLine(out, indent,
                 "if " + validateRc + " < 0 { return Err(" + validateRc + "); }");
      } else {
        emitLine(out, indent,
                 "if " + sizeVar +
                     " > _remaining { return Err(-crate::dsdl_runtime::DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER); }");
      }
      emitLine(out, indent,
               "let _start = crate::dsdl_runtime::choose_min(offset_bits / 8, buffer.len());");
      emitLine(out, indent,
               "let _end = _start.saturating_add(" + sizeVar + ").min(buffer.len());");
      emitLine(out, indent,
               "if let Err(e) = " + expr + ".deserialize(&buffer[_start.._end]) { return Err(e); }");
      emitLine(out, indent, "offset_bits += " + sizeVar + " * 8;");
      return;
    }

    emitLine(out, indent,
             "let _start = crate::dsdl_runtime::choose_min(offset_bits / 8, buffer.len());");
    emitLine(out, indent,
             "let _slice = &buffer[_start..buffer.len()];");
    emitLine(out, indent,
             "let _consumed = match " + expr + ".deserialize(_slice) { Ok(v) => v, Err(e) => return Err(e) };\n");
    emitLine(out, indent, "offset_bits += _consumed * 8;");
  }
};

void collectDependencies(const SemanticSection &section,
                         std::set<std::string> &deps) {
  for (const auto &f : section.fields) {
    if (f.resolvedType.compositeType) {
      const auto &r = *f.resolvedType.compositeType;
      deps.insert(loweredTypeKey(r.fullName, r.majorVersion, r.minorVersion));
    }
  }
}

std::string rustConstType(const TypeExprAST &type) {
  const auto *prim = std::get_if<PrimitiveTypeExprAST>(&type.scalar);
  if (!prim) {
    return "i64";
  }
  switch (prim->kind) {
  case PrimitiveKind::Bool:
    return "bool";
  case PrimitiveKind::Float:
    return "f64";
  case PrimitiveKind::SignedInt:
    return "i64";
  case PrimitiveKind::UnsignedInt:
  case PrimitiveKind::Byte:
  case PrimitiveKind::Utf8:
    return "u64";
  }
  return "i64";
}

std::string rustConstType(const TypeExprAST &type, const Value &value) {
  if (std::holds_alternative<std::string>(value.data)) {
    return "&'static str";
  }
  if (std::holds_alternative<bool>(value.data)) {
    return "bool";
  }
  return rustConstType(type);
}

void emitSectionType(std::ostringstream &out, const std::string &typeName,
                     const SemanticSection &section, const EmitterContext &ctx,
                     const std::string &fullName,
                     std::uint32_t majorVersion, std::uint32_t minorVersion,
                     const LoweredSectionFacts *const sectionFacts) {
  emitLine(out, 0, "#[derive(Clone, Debug, PartialEq)]");
  emitLine(out, 0, "pub struct " + typeName + " {");

  std::size_t fieldCount = 0;
  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }
    ++fieldCount;
    emitLine(out, 1,
             "pub " + sanitizeRustIdent(field.name) + ": " +
                 rustFieldType(field.resolvedType, ctx) + ",");
  }

  if (section.isUnion) {
    emitLine(out, 1, "pub _tag_: u8,");
  }

  if (fieldCount == 0 && !section.isUnion) {
    emitLine(out, 1, "pub _dummy_: u8,");
  }
  emitLine(out, 0, "}\n");

  emitLine(out, 0, "impl Default for " + typeName + " {");
  emitLine(out, 1, "fn default() -> Self {");
  emitLine(out, 2, "Self {");
  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }
    emitLine(out, 3,
             sanitizeRustIdent(field.name) + ": " +
                 defaultExpr(field.resolvedType, ctx) + ",");
  }
  if (section.isUnion) {
    emitLine(out, 3, "_tag_: 0,");
  }
  if (fieldCount == 0 && !section.isUnion) {
    emitLine(out, 3, "_dummy_: 0,");
  }
  emitLine(out, 2, "}");
  emitLine(out, 1, "}");
  emitLine(out, 0, "}\n");

  emitLine(out, 0, "impl " + typeName + " {");
  emitLine(out, 1, "pub const FULL_NAME: &'static str = \"" + fullName + "\";");
  emitLine(out, 1,
           "pub const FULL_NAME_AND_VERSION: &'static str = \"" + fullName +
               "." + std::to_string(majorVersion) + "." +
               std::to_string(minorVersion) + "\";");
  emitLine(out, 1,
           "pub const EXTENT_BYTES: usize = " +
               std::to_string(section.extentBits.value_or(0) / 8) + ";");
  emitLine(out, 1,
           "pub const SERIALIZATION_BUFFER_SIZE_BYTES: usize = " +
               std::to_string((section.serializationBufferSizeBits + 7) / 8) + ";");
  if (section.isUnion) {
    std::size_t optionCount = 0;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        ++optionCount;
      }
    }
    emitLine(out, 1,
             "pub const UNION_OPTION_COUNT: usize = " +
                 std::to_string(optionCount) + ";");
  }

  for (const auto &c : section.constants) {
    emitLine(out, 1,
             "pub const " + toUpperSnake(c.name) + ": " +
                 rustConstType(c.type, c.value) +
                 " = " + rustConstValue(c.value) + ";");
  }
  out << "\n";

  FunctionBodyEmitter body(ctx);
  body.emitSerialize(out, typeName, section, sectionFacts);
  out << "\n";
  body.emitDeserialize(out, typeName, section, sectionFacts);
  out << "\n";

  emitLine(out, 1,
           "pub fn to_bytes(&self) -> core::result::Result<crate::dsdl_runtime::DsdlVec<u8>, i8> {");
  emitLine(out, 2,
           "let mut buffer = crate::dsdl_runtime::DsdlVec::<u8>::with_capacity(Self::SERIALIZATION_BUFFER_SIZE_BYTES);");
  emitLine(out, 2, "buffer.resize(Self::SERIALIZATION_BUFFER_SIZE_BYTES, 0u8);");
  emitLine(out, 2, "let used = self.serialize(&mut buffer)?;");
  emitLine(out, 2, "buffer.truncate(used);");
  emitLine(out, 2, "Ok(buffer)");
  emitLine(out, 1, "}\n");

  emitLine(out, 1,
           "pub fn from_bytes(buffer: &[u8]) -> core::result::Result<(Self, usize), i8> {");
  emitLine(out, 2, "let mut out = Self::default();");
  emitLine(out, 2, "let used = out.deserialize(buffer)?;");
  emitLine(out, 2, "Ok((out, used))");
  emitLine(out, 1, "}");
  emitLine(out, 0, "}\n");
}

const LoweredSectionFacts *
findLoweredSectionFacts(const LoweredFactsMap &loweredFacts,
                        const SemanticDefinition &def,
                        llvm::StringRef sectionKey) {
  const auto defIt = loweredFacts.find(
      loweredTypeKey(def.info.fullName, def.info.majorVersion, def.info.minorVersion));
  if (defIt == loweredFacts.end()) {
    return nullptr;
  }
  const auto sectionIt = defIt->second.find(sectionKey.str());
  if (sectionIt == defIt->second.end()) {
    return nullptr;
  }
  return &sectionIt->second;
}

std::string renderDefinitionFile(const SemanticDefinition &def,
                                 const EmitterContext &ctx,
                                 const LoweredFactsMap &loweredFacts) {
  std::ostringstream out;
  emitLine(out, 0, "#![allow(non_camel_case_types)]");
  emitLine(out, 0, "#![allow(non_snake_case)]");
  emitLine(out, 0, "#![allow(non_upper_case_globals)]\n");

  std::set<std::string> deps;
  collectDependencies(def.request, deps);
  if (def.response) {
    collectDependencies(*def.response, deps);
  }

  const auto selfKey = loweredTypeKey(def.info.fullName, def.info.majorVersion,
                               def.info.minorVersion);

  for (const auto &dep : deps) {
    if (dep == selfKey) {
      continue;
    }

    auto first = dep.find(':');
    auto second = dep.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      continue;
    }

    SemanticTypeRef ref;
    ref.fullName = dep.substr(0, first);
    ref.majorVersion = static_cast<std::uint32_t>(
        std::stoul(dep.substr(first + 1, second - first - 1)));
    ref.minorVersion =
        static_cast<std::uint32_t>(std::stoul(dep.substr(second + 1)));

    if (const auto *resolved = ctx.find(ref)) {
      ref.namespaceComponents = resolved->info.namespaceComponents;
      ref.shortName = resolved->info.shortName;
    }

    const auto typePath = ctx.rustTypePath(ref);
    const auto rustType = ctx.rustTypeName(ref);
    emitLine(out, 0, "use " + typePath + ";");
    (void)rustType;
  }
  if (!deps.empty()) {
    out << "\n";
  }

  const auto baseType = ctx.rustTypeName(def.info);

  if (!def.isService) {
    emitSectionType(out, baseType, def.request, ctx, def.info.fullName,
                    def.info.majorVersion, def.info.minorVersion,
                    findLoweredSectionFacts(loweredFacts, def, ""));
    return out.str();
  }

  const auto reqType = baseType + "_Request";
  const auto respType = baseType + "_Response";

  emitSectionType(out, reqType, def.request, ctx,
                  def.info.fullName + ".Request", def.info.majorVersion,
                  def.info.minorVersion,
                  findLoweredSectionFacts(loweredFacts, def, "request"));

  if (def.response) {
    out << "\n";
    emitSectionType(out, respType, *def.response, ctx,
                    def.info.fullName + ".Response", def.info.majorVersion,
                    def.info.minorVersion,
                    findLoweredSectionFacts(loweredFacts, def, "response"));
  }

  out << "\n";
  emitLine(out, 0, "pub type " + baseType + " = " + reqType + ";");

  return out.str();
}

llvm::Expected<std::string> loadRustRuntime() {
  const std::filesystem::path runtimePath =
      std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "rust" /
      "dsdl_runtime.rs";
  std::ifstream in(runtimePath.string());
  if (!in) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to read Rust runtime");
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

std::string renderCargoToml(const RustEmitOptions &options) {
  std::ostringstream out;
  out << "[package]\n";
  out << "name = \"" << options.crateName << "\"\n";
  out << "version = \"0.1.0\"\n";
  out << "edition = \"2021\"\n\n";

  out << "[lib]\n";
  out << "path = \"src/lib.rs\"\n\n";

  out << "[features]\n";
  out << "default = [\"std\"]\n";
  out << "std = []\n";
  out << "future-no-std-alloc = []\n";
  return out.str();
}

} // namespace

llvm::Error emitRust(const SemanticModule &semantic, mlir::ModuleOp module,
                     const RustEmitOptions &options,
                     DiagnosticEngine &diagnostics) {
  if (options.outDir.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  }
  LoweredFactsMap loweredFacts;
  if (!collectLoweredFactsFromMlir(semantic, module, diagnostics, "Rust",
                                   &loweredFacts)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "MLIR schema coverage validation failed for Rust emission");
  }
  if (options.profile != RustProfile::Std) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Rust no_std+alloc backend is not implemented yet; use --rust-profile std");
  }

  std::filesystem::path outRoot(options.outDir);
  std::filesystem::path srcRoot = outRoot / "src";
  std::filesystem::create_directories(srcRoot);

  if (options.emitCargoToml) {
    if (auto err = writeFile(outRoot / "Cargo.toml", renderCargoToml(options))) {
      return err;
    }
  }

  auto runtime = loadRustRuntime();
  if (!runtime) {
    return runtime.takeError();
  }
  if (auto err = writeFile(srcRoot / "dsdl_runtime.rs", *runtime)) {
    return err;
  }

  EmitterContext ctx(semantic);

  std::map<std::string, std::set<std::string>> dirToSubdirs;
  std::map<std::string, std::set<std::string>> dirToFiles;

  for (const auto &def : semantic.definitions) {
    std::vector<std::string> ns;
    ns.reserve(def.info.namespaceComponents.size());
    for (const auto &c : def.info.namespaceComponents) {
      ns.push_back(sanitizeRustIdent(c));
    }

    std::string dirRel;
    std::string parentRel;
    for (const auto &component : ns) {
      dirToSubdirs[parentRel].insert(component);
      if (!dirRel.empty()) {
        dirRel += "/";
      }
      dirRel += component;
      parentRel = dirRel;
    }

    const auto modName = ctx.rustModuleName(def.info);
    dirToFiles[dirRel].insert(modName);

    std::filesystem::path dir = srcRoot;
    if (!dirRel.empty()) {
      dir /= dirRel;
    }
    std::filesystem::create_directories(dir);

    if (auto err = writeFile(dir / (modName + ".rs"),
                             renderDefinitionFile(def, ctx, loweredFacts))) {
      return err;
    }
  }

  std::ostringstream lib;
  emitLine(lib, 0, "#![allow(non_camel_case_types)]");
  emitLine(lib, 0, "#![allow(non_snake_case)]");
  emitLine(lib, 0, "#![allow(non_upper_case_globals)]");
  emitLine(lib, 0, "\n// std-first backend. Future no_std + allocator mode is reserved via feature/profile seams.");
  emitLine(lib, 0, "#[cfg(feature = \"future-no-std-alloc\")]");
  emitLine(lib, 0,
           "compile_error!(\"feature 'future-no-std-alloc' is reserved but not implemented yet; generate with --rust-profile std\");");
  emitLine(lib, 0, "pub mod dsdl_runtime;");

  if (dirToSubdirs.contains("")) {
    for (const auto &sub : dirToSubdirs[""]) {
      emitLine(lib, 0, "pub mod " + sub + ";");
    }
  }
  if (dirToFiles.contains("")) {
    for (const auto &file : dirToFiles[""]) {
      emitLine(lib, 0, "pub mod " + file + ";");
    }
  }

  if (auto err = writeFile(srcRoot / "lib.rs", lib.str())) {
    return err;
  }

  std::set<std::string> dirs;
  for (const auto &[d, _] : dirToSubdirs) {
    if (!d.empty()) {
      dirs.insert(d);
    }
  }
  for (const auto &[d, _] : dirToFiles) {
    if (!d.empty()) {
      dirs.insert(d);
    }
  }

  for (const auto &dirRel : dirs) {
    std::ostringstream mod;
    if (dirToSubdirs.contains(dirRel)) {
      for (const auto &sub : dirToSubdirs[dirRel]) {
        emitLine(mod, 0, "pub mod " + sub + ";");
      }
    }
    if (dirToFiles.contains(dirRel)) {
      for (const auto &file : dirToFiles[dirRel]) {
        emitLine(mod, 0, "pub mod " + file + ";");
      }
    }

    std::filesystem::path dir = srcRoot / dirRel;
    std::filesystem::create_directories(dir);
    if (auto err = writeFile(dir / "mod.rs", mod.str())) {
      return err;
    }
  }

  return llvm::Error::success();
}

} // namespace llvmdsdl
