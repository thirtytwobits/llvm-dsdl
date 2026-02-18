#include "llvmdsdl/CodeGen/CppEmitter.h"
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

bool isCppKeyword(const std::string &name) {
  static const std::set<std::string> kKeywords = {
      "alignas",   "alignof",    "and",         "and_eq",   "asm",
      "atomic_cancel", "atomic_commit", "atomic_noexcept", "auto", "bitand",
      "bitor",     "bool",       "break",       "case",     "catch",
      "char",      "char8_t",    "char16_t",    "char32_t", "class",
      "compl",     "concept",    "const",       "consteval", "constexpr",
      "constinit", "const_cast", "continue",    "co_await", "co_return",
      "co_yield",  "decltype",   "default",     "delete",   "do",
      "double",    "dynamic_cast", "else",      "enum",     "explicit",
      "export",    "extern",     "false",       "float",    "for",
      "friend",    "goto",       "if",          "inline",   "int",
      "long",      "mutable",    "namespace",   "new",      "noexcept",
      "not",       "not_eq",     "nullptr",     "operator", "or",
      "or_eq",     "private",    "protected",   "public",   "register",
      "reinterpret_cast", "requires", "return", "short",    "signed",
      "sizeof",    "static",     "static_assert", "static_cast", "struct",
      "switch",    "template",   "this",        "thread_local", "throw",
      "true",      "try",        "typedef",     "typeid",   "typename",
      "union",     "unsigned",   "using",       "virtual",  "void",
      "volatile",  "wchar_t",    "while",       "xor",      "xor_eq"};
  return kKeywords.contains(name);
}

std::string sanitizeIdentifier(std::string name) {
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
  if (isCppKeyword(name)) {
    name += '_';
  }
  return name;
}

std::string sanitizeMacroToken(std::string token) {
  for (char &c : token) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      c = '_';
    } else {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
  }
  if (!token.empty() && std::isdigit(static_cast<unsigned char>(token.front()))) {
    token.insert(token.begin(), '_');
  }
  return token;
}

std::string headerFileName(const DiscoveredDefinition &info) {
  return llvm::formatv("{0}_{1}_{2}.hpp", info.shortName, info.majorVersion,
                       info.minorVersion)
      .str();
}

std::string headerGuard(const DiscoveredDefinition &info) {
  std::string g = "LLVMDSDL_CPP_" + info.fullName + "_" +
                  std::to_string(info.majorVersion) + "_" +
                  std::to_string(info.minorVersion) + "_HPP";
  for (char &c : g) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    } else {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
  }
  return g;
}

std::string valueToCppExpr(const Value &value) {
  if (const auto *b = std::get_if<bool>(&value.data)) {
    return *b ? "true" : "false";
  }
  if (const auto *r = std::get_if<Rational>(&value.data)) {
    if (r->isInteger()) {
      return std::to_string(r->asInteger().value());
    }
    std::ostringstream out;
    out << "((double)" << r->numerator() << "/(double)" << r->denominator()
        << ")";
    return out.str();
  }
  if (const auto *s = std::get_if<std::string>(&value.data)) {
    if (s->size() == 1) {
      const char c = (*s)[0];
      if (c == '\\' || c == '\'' ) {
        return std::string("'\\") + c + "'";
      }
      return std::string("'") + c + "'";
    }
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

std::string unsignedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "std::uint8_t";
  case 16:
    return "std::uint16_t";
  case 32:
    return "std::uint32_t";
  default:
    return "std::uint64_t";
  }
}

std::string signedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "std::int8_t";
  case 16:
    return "std::int16_t";
  case 32:
    return "std::int32_t";
  default:
    return "std::int64_t";
  }
}

std::string unsignedGetter(const std::uint32_t bitLength) {
  return "dsdl_runtime_get_u" + std::string(scalarWidthSuffix(bitLength));
}

std::string signedGetter(const std::uint32_t bitLength) {
  return "dsdl_runtime_get_i" + std::string(scalarWidthSuffix(bitLength));
}

std::string cppNamespacePath(const std::vector<std::string> &components) {
  std::string out;
  for (const auto &component : components) {
    if (!out.empty()) {
      out += "::";
    }
    out += sanitizeIdentifier(component);
  }
  return out;
}

void emitNamespaceOpen(std::ostringstream &out,
                       const std::vector<std::string> &components) {
  if (components.empty()) {
    return;
  }
  out << "namespace " << cppNamespacePath(components) << " {\n\n";
}

void emitNamespaceClose(std::ostringstream &out,
                        const std::vector<std::string> &components) {
  if (components.empty()) {
    return;
  }
  out << "\n} // namespace " << cppNamespacePath(components) << "\n";
}

class EmitterContext final {
public:
  explicit EmitterContext(const SemanticModule &semantic) {
    for (const auto &def : semantic.definitions) {
      const auto key = loweredTypeKey(def.info.fullName, def.info.majorVersion,
                               def.info.minorVersion);
      byKey_.emplace(key, &def);
      versionCountByFullName_[def.info.fullName] += 1U;
    }
  }

  const SemanticDefinition *find(const SemanticTypeRef &ref) const {
    const auto it = byKey_.find(loweredTypeKey(ref.fullName, ref.majorVersion,
                                        ref.minorVersion));
    if (it == byKey_.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::string cppTypeName(const DiscoveredDefinition &info) const {
    std::string out = sanitizeIdentifier(info.shortName);
    const auto it = versionCountByFullName_.find(info.fullName);
    if (it != versionCountByFullName_.end() && it->second > 1U) {
      out += "_" + std::to_string(info.majorVersion) + "_" +
             std::to_string(info.minorVersion);
    }
    return out;
  }

  std::string cppTypeName(const SemanticDefinition &def) const {
    return cppTypeName(def.info);
  }

  std::string cppTypeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return cppTypeName(*def);
    }

    std::string out = sanitizeIdentifier(ref.shortName);
    out += "_" + std::to_string(ref.majorVersion) + "_" +
           std::to_string(ref.minorVersion);
    return out;
  }

  std::string cppQualifiedTypeName(const SemanticDefinition &def) const {
    std::string out = "::";
    const auto ns = cppNamespacePath(def.info.namespaceComponents);
    if (!ns.empty()) {
      out += ns + "::";
    }
    out += cppTypeName(def);
    return out;
  }

  std::string cppQualifiedTypeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return cppQualifiedTypeName(*def);
    }

    std::string out = "::";
    const auto ns = cppNamespacePath(ref.namespaceComponents);
    if (!ns.empty()) {
      out += ns + "::";
    }
    out += cppTypeName(ref);
    return out;
  }

  std::string relativeHeaderPath(const SemanticDefinition &def) const {
    std::filesystem::path p;
    for (const auto &ns : def.info.namespaceComponents) {
      p /= ns;
    }
    p /= headerFileName(def.info);
    return p.generic_string();
  }

private:
  std::unordered_map<std::string, const SemanticDefinition *> byKey_;
  std::unordered_map<std::string, std::size_t> versionCountByFullName_;
};

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
  out << std::string(static_cast<std::size_t>(indent) * 2U, ' ') << line << '\n';
}

class FunctionBodyEmitter final {
public:
  explicit FunctionBodyEmitter(const EmitterContext &ctx, const bool pmrMode)
      : ctx_(ctx), pmrMode_(pmrMode) {}

  void emitSerializeFunction(std::ostringstream &out, const std::string &typeName,
                             const SemanticSection &section,
                             const LoweredSectionFacts *const sectionFacts) {
    emitLine(out, 0, "inline std::int8_t " + typeName +
                        "__serialize_(const " + typeName +
                        "* const obj, std::uint8_t* const buffer, std::size_t* const "
                        "inout_buffer_size_bytes" +
                        (pmrMode_ ? ", ::llvmdsdl::cpp::MemoryResource* const memory_resource" : "") +
                        ")");
    emitLine(out, 0, "{");
    emitLine(out, 1, "if ((obj == nullptr) || (buffer == nullptr) || (inout_buffer_size_bytes == nullptr)) {");
    emitLine(out, 2, "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_INVALID_ARGUMENT);");
    emitLine(out, 1, "}");
    if (pmrMode_) {
      emitLine(out, 1,
               "::llvmdsdl::cpp::MemoryResource* const effective_memory_resource = "
               "(memory_resource != nullptr) ? memory_resource : obj->_memory_resource;");
      emitLine(out, 1, "(void)effective_memory_resource;");
    }
    emitLine(out, 1, "const std::size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "std::size_t offset_bits = 0U;");
    emitSerializeMlirHelperBindings(out, section, 1, sectionFacts);
    const auto fieldPlan = buildSectionStatementPlan(section, sectionFacts);
    if (const auto capacityHelperSymbol =
            resolveSectionCapacityCheckHelperSymbol(sectionFacts);
        !capacityHelperSymbol.empty()) {
      const auto capacityHelper = helperBindingName(capacityHelperSymbol);
      const auto errCapacity = nextName("err_capacity");
      emitLine(out, 1,
               "const std::int8_t " + errCapacity + " = " + capacityHelper +
                   "(static_cast<std::int64_t>(capacity_bytes * 8U));");
      emitLine(out, 1, "if (" + errCapacity +
                           " != static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS)) {");
      emitLine(out, 2, "return " + errCapacity + ";");
      emitLine(out, 1, "}");
    } else {
      emitLine(out, 1, "if ((capacity_bytes * 8U) < " +
                          std::to_string(section.serializationBufferSizeBits) + "ULL) {");
      emitLine(out, 2,
               "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL);");
      emitLine(out, 1, "}");
    }

    if (section.isUnion) {
      emitSerializeUnion(out, section, fieldPlan.unionBranches, "obj", 1,
                         sectionFacts);
    } else {
      for (const auto &step : fieldPlan.orderedFields) {
        const auto &field = *step.field;
        emitAlignSerialize(out, field.resolvedType.alignmentBits, 1);
        if (field.isPadding) {
          emitSerializePadding(out, field.resolvedType, 1);
        } else {
          emitSerializeValue(out, field.resolvedType,
                             "obj->" + sanitizeIdentifier(field.name), 1,
                             step.arrayLengthPrefixBits, step.fieldFacts);
        }
      }
    }

    emitAlignSerialize(out, 8, 1);
    emitLine(out, 1, "*inout_buffer_size_bytes = static_cast<std::size_t>(offset_bits / 8U);");
    emitLine(out, 1, "return static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS);");
    emitLine(out, 0, "}");
    out << "\n";
  }

  void emitDeserializeFunction(std::ostringstream &out,
                               const std::string &typeName,
                               const SemanticSection &section,
                               const LoweredSectionFacts *const sectionFacts) {
    emitLine(out, 0, "inline std::int8_t " + typeName +
                        "__deserialize_(" + typeName +
                        "* const out_obj, const std::uint8_t* buffer, std::size_t* const "
                        "inout_buffer_size_bytes" +
                        (pmrMode_ ? ", ::llvmdsdl::cpp::MemoryResource* const memory_resource" : "") +
                        ")");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "if ((out_obj == nullptr) || (inout_buffer_size_bytes == nullptr) || ((buffer == nullptr) && (0U != *inout_buffer_size_bytes))) {");
    emitLine(out, 2, "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_INVALID_ARGUMENT);");
    emitLine(out, 1, "}");
    if (pmrMode_) {
      emitLine(out, 1,
               "::llvmdsdl::cpp::MemoryResource* const effective_memory_resource = "
               "(memory_resource != nullptr) ? memory_resource : out_obj->_memory_resource;");
      emitLine(out, 1, "(void)effective_memory_resource;");
      emitLine(out, 1,
               "if (effective_memory_resource != nullptr) { out_obj->set_memory_resource(effective_memory_resource); }");
    }
    emitLine(out, 1, "if (buffer == nullptr) {");
    emitLine(out, 2, "buffer = reinterpret_cast<const std::uint8_t*>(\"\");");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const std::size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "const std::size_t capacity_bits = capacity_bytes * 8U;");
    emitLine(out, 1, "std::size_t offset_bits = 0U;");
    emitDeserializeMlirHelperBindings(out, section, 1, sectionFacts);
    const auto fieldPlan = buildSectionStatementPlan(section, sectionFacts);

    if (section.isUnion) {
      emitDeserializeUnion(out, section, fieldPlan.unionBranches, "out_obj", 1,
                           sectionFacts);
    } else {
      for (const auto &step : fieldPlan.orderedFields) {
        const auto &field = *step.field;
        emitAlignDeserialize(out, field.resolvedType.alignmentBits, 1);
        if (field.isPadding) {
          emitDeserializePadding(out, field.resolvedType, 1);
        } else {
          emitDeserializeValue(out, field.resolvedType,
                               "out_obj->" + sanitizeIdentifier(field.name), 1,
                               step.arrayLengthPrefixBits, step.fieldFacts);
        }
      }
    }

    emitAlignDeserialize(out, 8, 1);
    emitLine(out, 1,
             "*inout_buffer_size_bytes = static_cast<std::size_t>(dsdl_runtime_choose_min(offset_bits, capacity_bits) / 8U);");
    emitLine(out, 1, "return static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS);");
    emitLine(out, 0, "}");
    out << "\n";
  }

private:
  const EmitterContext &ctx_;
  bool pmrMode_{false};
  std::size_t id_{0};

  std::string nextName(const std::string &prefix) {
    return "_" + prefix + std::to_string(id_++) + "_";
  }

  std::string helperBindingName(const std::string &helperSymbol) const {
    return "mlir_" + sanitizeIdentifier(helperSymbol);
  }

  void emitSerializeMlirHelperBindings(
      std::ostringstream &out, const SemanticSection &section, const int indent,
      const LoweredSectionFacts *const sectionFacts) {
    const auto plan = buildSectionHelperBindingPlan(
        section, sectionFacts, HelperBindingDirection::Serialize);
    for (const auto &line : renderSectionHelperBindings(
             plan, HelperBindingRenderLanguage::Cpp,
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
             plan, HelperBindingRenderLanguage::Cpp,
             ScalarBindingRenderDirection::Deserialize,
             [this](const std::string &symbol) {
               return helperBindingName(symbol);
             },
             /*emitCapacityCheck=*/false)) {
      emitLine(out, indent, line);
    }
  }

  std::string containerElementType(const SemanticFieldType &type) const {
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
      return type.bitLength == 64U ? "double" : "float";
    case SemanticScalarCategory::Void:
      return "std::uint8_t";
    case SemanticScalarCategory::Composite:
      if (type.compositeType) {
        return ctx_.cppQualifiedTypeName(*type.compositeType);
      }
      return "std::uint8_t";
    }
    return "std::uint8_t";
  }

  void emitAlignSerialize(std::ostringstream &out,
                          const std::int64_t alignmentBits,
                          const int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    const auto alignedOffset = nextName("aligned_offset");
    const auto bitIndex = nextName("align_bit");
    const auto err = nextName("err");
    emitLine(out, indent,
             "const std::size_t " + alignedOffset + " = (offset_bits + " +
                 std::to_string(alignmentBits - 1) + "U) & ~(std::size_t)" +
                 std::to_string(alignmentBits - 1) + "U;");
    emitLine(out, indent,
             "for (std::size_t " + bitIndex + " = offset_bits; " + bitIndex +
                 " < " + alignedOffset + "; ++" + bitIndex + ") {");
    emitLine(out, indent + 1,
             "const std::int8_t " + err +
                 " = dsdl_runtime_set_bit(buffer, capacity_bytes, " + bitIndex +
                 ", false);");
    emitLine(out, indent + 1, "if (" + err + " < 0) {");
    emitLine(out, indent + 2, "return " + err + ";");
    emitLine(out, indent + 1, "}");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offset_bits = " + alignedOffset + ";");
  }

  void emitAlignDeserialize(std::ostringstream &out,
                            const std::int64_t alignmentBits,
                            const int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    emitLine(out, indent,
             "offset_bits = (offset_bits + " + std::to_string(alignmentBits - 1) +
                 "U) & ~(std::size_t)" + std::to_string(alignmentBits - 1) + "U;");
  }

  void emitSerializePadding(std::ostringstream &out, const SemanticFieldType &type,
                            const int indent) {
    if (type.bitLength == 0) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             "const std::int8_t " + err +
                 " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, 0U, " +
                 std::to_string(type.bitLength) + "U);");
    emitLine(out, indent, "if (" + err + " < 0) {");
    emitLine(out, indent + 1, "return " + err + ";");
    emitLine(out, indent, "}");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(type.bitLength) + "U;");
  }

  void emitDeserializePadding(std::ostringstream &out,
                              const SemanticFieldType &type,
                              const int indent) {
    if (type.bitLength == 0) {
      return;
    }
    emitLine(out, indent,
             "offset_bits += " + std::to_string(type.bitLength) + "U;");
  }

  void emitSerializeUnion(
      std::ostringstream &out, const SemanticSection &section,
      const std::vector<PlannedFieldStep> &unionBranches,
      const std::string &objRef, const int indent,
      const LoweredSectionFacts *const sectionFacts) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    const auto validateHelperSymbol =
        resolveSectionUnionTagValidateHelperSymbol(sectionFacts);
    const auto validateHelper = validateHelperSymbol.empty()
                                    ? std::string{}
                                    : helperBindingName(validateHelperSymbol);
    if (!validateHelper.empty()) {
      const auto validateErr = nextName("err_union_tag");
      emitLine(out, indent,
               "const std::int8_t " + validateErr + " = " + validateHelper +
                   "(static_cast<std::int64_t>(" + objRef + "->_tag_));");
      emitLine(out, indent, "if (" + validateErr +
                           " != static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS)) {");
      emitLine(out, indent + 1, "return " + validateErr + ";");
      emitLine(out, indent, "}");
    }
    const auto tagHelperSymbol = resolveSectionUnionTagMaskHelperSymbol(
        sectionFacts, HelperBindingDirection::Serialize);
    const auto tagHelper = tagHelperSymbol.empty()
                               ? std::string{}
                               : helperBindingName(tagHelperSymbol);
    std::string tagExpr = "static_cast<std::uint64_t>(" + objRef + "->_tag_)";
    if (!tagHelper.empty()) {
      tagExpr = tagHelper + "(" + tagExpr + ")";
    }

    const auto tagErr = nextName("err");
    emitLine(out, indent,
             "const std::int8_t " + tagErr +
                 " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, " +
                 tagExpr + ", " +
                 std::to_string(tagBits) + "U);");
    emitLine(out, indent, "if (" + tagErr + " < 0) {");
    emitLine(out, indent + 1, "return " + tagErr + ";");
    emitLine(out, indent, "}");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + "U;");

    bool first = true;
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      const auto member = sanitizeIdentifier(field.name);
      emitLine(out, indent,
               std::string(first ? "if" : "else if") + " (" +
                   objRef + "->_tag_ == " +
                   std::to_string(field.unionOptionIndex) + "U) {");
      emitAlignSerialize(out, field.resolvedType.alignmentBits, indent + 1);
      emitSerializeValue(out, field.resolvedType,
                         objRef + "->" + member, indent + 1,
                         step.arrayLengthPrefixBits, step.fieldFacts);
      emitLine(out, indent, "}");
      first = false;
    }

    emitLine(out, indent, "else {");
    emitLine(out, indent + 1,
             "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG);");
    emitLine(out, indent, "}");
  }

  void emitDeserializeUnion(std::ostringstream &out,
                            const SemanticSection &section,
                            const std::vector<PlannedFieldStep> &unionBranches,
                            const std::string &objRef, const int indent,
                            const LoweredSectionFacts *const sectionFacts) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    const auto rawTag = nextName("tag_raw");
    emitLine(out, indent,
             "const std::uint64_t " + rawTag + " = static_cast<std::uint64_t>(" +
                 unsignedGetter(tagBits) +
                 "(buffer, capacity_bytes, offset_bits, " +
                 std::to_string(tagBits) + "U));");
    const auto tagHelperSymbol = resolveSectionUnionTagMaskHelperSymbol(
        sectionFacts, HelperBindingDirection::Deserialize);
    const auto tagHelper = tagHelperSymbol.empty()
                               ? std::string{}
                               : helperBindingName(tagHelperSymbol);
    std::string tagExpr = rawTag;
    if (!tagHelper.empty()) {
      tagExpr = tagHelper + "(" + tagExpr + ")";
    }

    emitLine(out, indent,
             objRef + "->_tag_ = static_cast<std::uint8_t>(" + tagExpr + ");");
    const auto validateHelperSymbol =
        resolveSectionUnionTagValidateHelperSymbol(sectionFacts);
    const auto validateHelper = validateHelperSymbol.empty()
                                    ? std::string{}
                                    : helperBindingName(validateHelperSymbol);
    if (!validateHelper.empty()) {
      const auto validateErr = nextName("err_union_tag");
      emitLine(out, indent,
               "const std::int8_t " + validateErr + " = " + validateHelper +
                   "(static_cast<std::int64_t>(" + objRef + "->_tag_));");
      emitLine(out, indent, "if (" + validateErr +
                           " != static_cast<std::int8_t>(DSDL_RUNTIME_SUCCESS)) {");
      emitLine(out, indent + 1, "return " + validateErr + ";");
      emitLine(out, indent, "}");
    }
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + "U;");

    bool first = true;
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      const auto member = sanitizeIdentifier(field.name);
      emitLine(out, indent,
               std::string(first ? "if" : "else if") + " (" +
                   objRef + "->_tag_ == " +
                   std::to_string(field.unionOptionIndex) + "U) {");
      emitAlignDeserialize(out, field.resolvedType.alignmentBits, indent + 1);
      emitDeserializeValue(out, field.resolvedType,
                           objRef + "->" + member, indent + 1,
                           step.arrayLengthPrefixBits, step.fieldFacts);
      emitLine(out, indent, "}");
      first = false;
    }

    emitLine(out, indent, "else {");
    emitLine(out, indent + 1,
             "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG);");
    emitLine(out, indent, "}");
  }

  void emitSerializeValue(std::ostringstream &out, const SemanticFieldType &type,
                          const std::string &expr, const int indent,
                          const std::optional<std::uint32_t>
                              arrayLengthPrefixBitsOverride = std::nullopt,
                          const LoweredFieldFacts *const fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitSerializeArray(out, type, expr, indent,
                         arrayLengthPrefixBitsOverride, fieldFacts);
      return;
    }

    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool: {
      const auto err = nextName("err");
      emitLine(out, indent,
               "const std::int8_t " + err +
                   " = dsdl_runtime_set_bit(buffer, capacity_bytes, offset_bits, " +
                   expr + ");");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offset_bits += 1U;");
      break;
    }
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      std::string valueExpr = "static_cast<std::uint64_t>(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        valueExpr = helper + "(" + valueExpr + ")";
      } else if (type.castMode == CastMode::Saturated) {
        if (const auto maxVal = resolveUnsignedSaturationMax(type.bitLength)) {
          const auto sat = nextName("sat");
          emitLine(
              out, indent,
              "std::uint64_t " + sat + " = static_cast<std::uint64_t>(" + expr +
                  ");");
          emitLine(out, indent,
                   "if (" + sat + " > " + std::to_string(*maxVal) + "ULL) {");
          emitLine(out, indent + 1,
                   sat + " = " + std::to_string(*maxVal) + "ULL;");
          emitLine(out, indent, "}");
          valueExpr = sat;
        }
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "const std::int8_t " + err +
                   " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, " +
                   valueExpr + ", " + std::to_string(type.bitLength) + "U);");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      std::string valueExpr = "static_cast<std::int64_t>(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        valueExpr = helper + "(" + valueExpr + ")";
      } else if (type.castMode == CastMode::Saturated) {
        if (const auto range = resolveSignedSaturationRange(type.bitLength)) {
          const auto sat = nextName("sat");
          emitLine(
              out, indent,
              "std::int64_t " + sat + " = static_cast<std::int64_t>(" + expr +
                  ");");
          emitLine(out, indent,
                   "if (" + sat + " < " + std::to_string(range->first) + "LL) {");
          emitLine(out, indent + 1,
                   sat + " = " + std::to_string(range->first) + "LL;");
          emitLine(out, indent, "}");
          emitLine(out, indent,
                   "if (" + sat + " > " + std::to_string(range->second) +
                       "LL) {");
          emitLine(out, indent + 1,
                   sat + " = " + std::to_string(range->second) + "LL;");
          emitLine(out, indent, "}");
          valueExpr = sat;
        }
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "const std::int8_t " + err +
                   " = dsdl_runtime_set_ixx(buffer, capacity_bytes, offset_bits, " +
                   valueExpr + ", " + std::to_string(type.bitLength) + "U);");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    }
    case SemanticScalarCategory::Float: {
      const auto err = nextName("err");
      std::string normalizedExpr = "static_cast<double>(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        normalizedExpr = helper + "(" + normalizedExpr + ")";
      }
      std::string call;
      if (type.bitLength == 16U) {
        call = "dsdl_runtime_set_f16(buffer, capacity_bytes, offset_bits, static_cast<float>(" +
               normalizedExpr + "))";
      } else if (type.bitLength == 32U) {
        call = "dsdl_runtime_set_f32(buffer, capacity_bytes, offset_bits, static_cast<float>(" +
               normalizedExpr + "))";
      } else {
        call = "dsdl_runtime_set_f64(buffer, capacity_bytes, offset_bits, static_cast<double>(" +
               normalizedExpr + "))";
      }
      emitLine(out, indent, "const std::int8_t " + err + " = " + call + ";");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
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

  void emitDeserializeValue(std::ostringstream &out, const SemanticFieldType &type,
                            const std::string &expr, const int indent,
                            const std::optional<std::uint32_t>
                                arrayLengthPrefixBitsOverride = std::nullopt,
                            const LoweredFieldFacts *const fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitDeserializeArray(out, type, expr, indent,
                           arrayLengthPrefixBitsOverride, fieldFacts);
      return;
    }

    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool:
      emitLine(out, indent,
               expr + " = dsdl_runtime_get_bit(buffer, capacity_bytes, offset_bits);");
      emitLine(out, indent, "offset_bits += 1U;");
      break;
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto raw = nextName("raw");
        emitLine(out, indent,
                 "const std::uint64_t " + raw + " = static_cast<std::uint64_t>(" +
                     unsignedGetter(type.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, " +
                     std::to_string(type.bitLength) + "U));");
        emitLine(out, indent,
                 expr + " = static_cast<" + unsignedStorageType(type.bitLength) +
                     ">(" + helper + "(" + raw + "));");
      } else {
        emitLine(out, indent,
                 expr + " = static_cast<" + unsignedStorageType(type.bitLength) + ">(" +
                     unsignedGetter(type.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, " +
                     std::to_string(type.bitLength) + "U));");
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto raw = nextName("raw");
        emitLine(out, indent,
                 "const std::int64_t " + raw + " = static_cast<std::int64_t>(" +
                     unsignedGetter(type.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, " +
                     std::to_string(type.bitLength) + "U));");
        emitLine(out, indent,
                 expr + " = static_cast<" + signedStorageType(type.bitLength) +
                     ">(" + helper + "(" + raw + "));");
      } else {
        emitLine(out, indent,
                 expr + " = static_cast<" + signedStorageType(type.bitLength) + ">(" +
                     signedGetter(type.bitLength) +
                     "(buffer, capacity_bytes, offset_bits, " +
                     std::to_string(type.bitLength) + "U));");
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
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
                   expr + " = static_cast<float>(" + helper +
                       "(static_cast<double>(dsdl_runtime_get_f16(buffer, capacity_bytes, offset_bits))));");
        } else {
          emitLine(out, indent,
                   expr + " = dsdl_runtime_get_f16(buffer, capacity_bytes, offset_bits);");
        }
      } else if (type.bitLength == 32U) {
        if (!helper.empty()) {
          emitLine(out, indent,
                   expr + " = static_cast<float>(" + helper +
                       "(static_cast<double>(dsdl_runtime_get_f32(buffer, capacity_bytes, offset_bits))));");
        } else {
          emitLine(out, indent,
                   expr + " = dsdl_runtime_get_f32(buffer, capacity_bytes, offset_bits);");
        }
      } else {
        if (!helper.empty()) {
          emitLine(out, indent,
                   expr + " = static_cast<double>(" + helper +
                       "(static_cast<double>(dsdl_runtime_get_f64(buffer, capacity_bytes, offset_bits))));");
        } else {
          emitLine(out, indent,
                   expr + " = dsdl_runtime_get_f64(buffer, capacity_bytes, offset_bits);");
        }
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
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
    const bool elementIsBool = type.scalarCategory == SemanticScalarCategory::Bool;
    const auto arrayPlan = buildArrayWirePlan(
        type, fieldFacts, arrayLengthPrefixBitsOverride,
        HelperBindingDirection::Serialize);
    const bool variable = arrayPlan.variable;
    const auto prefixBits = arrayPlan.prefixBits;
    const auto &arrayDescriptor = arrayPlan.descriptor;

    if (variable) {
      std::string validateHelper;
      if (arrayDescriptor && !arrayDescriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayDescriptor->validateSymbol);
      }
      if (!validateHelper.empty()) {
        const auto validateRc = nextName("len_rc");
        emitLine(out, indent,
                 "const std::int8_t " + validateRc + " = " + validateHelper +
                     "(static_cast<std::int64_t>(" + expr + ".size()));");
        emitLine(out, indent, "if (" + validateRc + " < 0) {");
        emitLine(out, indent + 1, "return " + validateRc + ";");
        emitLine(out, indent, "}");
      } else {
        emitLine(out, indent,
                 "if (" + expr + ".size() > " +
                     std::to_string(type.arrayCapacity) + "U) {");
        emitLine(out, indent + 1,
                 "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH);");
        emitLine(out, indent, "}");
      }

      std::string prefixExpr = "static_cast<std::uint64_t>(" + expr + ".size())";
      std::string serPrefixHelper;
      if (arrayDescriptor && !arrayDescriptor->prefixSymbol.empty()) {
        serPrefixHelper = helperBindingName(arrayDescriptor->prefixSymbol);
      }
      if (!serPrefixHelper.empty()) {
        prefixExpr = serPrefixHelper + "(" + prefixExpr + ")";
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "const std::int8_t " + err +
                   " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, " +
                   prefixExpr + ", " +
                   std::to_string(prefixBits) + "U);");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(prefixBits) + "U;");
    }

    if (elementIsBool && !variable) {
      const auto source = expr + ".data()";
      const auto countExpr = std::to_string(type.arrayCapacity) + "U";
      emitLine(out, indent,
               "dsdl_runtime_copy_bits(&buffer[0], offset_bits, " + countExpr +
                   ", " + source + ", 0U);");
      emitLine(out, indent, "offset_bits += " + countExpr + ";");
      return;
    }

    const auto index = nextName("index");
    const auto bound = variable ? (expr + ".size()")
                                : std::to_string(type.arrayCapacity) + "U";
    const auto accessPrefix = expr;

    emitLine(out, indent,
             "for (std::size_t " + index + " = 0U; " + index + " < " + bound +
                 "; ++" + index + ") {");
    const auto elementType = arrayElementType(type);
    emitSerializeValue(out, elementType,
                       accessPrefix + "[" + index + "]", indent + 1,
                       std::nullopt, fieldFacts);
    emitLine(out, indent, "}");
  }

  void emitDeserializeArray(std::ostringstream &out,
                            const SemanticFieldType &type,
                            const std::string &expr, const int indent,
                            const std::optional<std::uint32_t>
                                arrayLengthPrefixBitsOverride,
                            const LoweredFieldFacts *const fieldFacts) {
    const bool elementIsBool = type.scalarCategory == SemanticScalarCategory::Bool;
    const auto arrayPlan = buildArrayWirePlan(
        type, fieldFacts, arrayLengthPrefixBitsOverride,
        HelperBindingDirection::Deserialize);
    const bool variable = arrayPlan.variable;
    const auto prefixBits = arrayPlan.prefixBits;
    const auto &arrayDescriptor = arrayPlan.descriptor;
    std::string countExpr;

    if (variable) {
      const auto rawCountVar = nextName("count_raw");
      emitLine(out, indent,
               "const std::uint64_t " + rawCountVar + " = static_cast<std::uint64_t>(" +
                   unsignedGetter(prefixBits) +
                   "(buffer, capacity_bytes, offset_bits, " +
                   std::to_string(prefixBits) + "U));");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(prefixBits) + "U;");
      std::string countRawExpr = rawCountVar;
      std::string deserPrefixHelper;
      if (arrayDescriptor && !arrayDescriptor->prefixSymbol.empty()) {
        deserPrefixHelper = helperBindingName(arrayDescriptor->prefixSymbol);
      }
      if (!deserPrefixHelper.empty()) {
        countRawExpr = deserPrefixHelper + "(" + countRawExpr + ")";
      }
      const auto countVar = nextName("count");
      emitLine(out, indent,
               "const std::size_t " + countVar + " = static_cast<std::size_t>(" +
                   countRawExpr + ");");

      std::string validateHelper;
      if (arrayDescriptor && !arrayDescriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayDescriptor->validateSymbol);
      }
      if (!validateHelper.empty()) {
        const auto validateRc = nextName("len_rc");
        emitLine(out, indent,
                 "const std::int8_t " + validateRc + " = " + validateHelper +
                     "(static_cast<std::int64_t>(" + countVar + "));");
        emitLine(out, indent, "if (" + validateRc + " < 0) {");
        emitLine(out, indent + 1, "return " + validateRc + ";");
        emitLine(out, indent, "}");
      } else {
        emitLine(out, indent,
                 "if (" + countVar + " > " +
                     std::to_string(type.arrayCapacity) + "U) {");
        emitLine(out, indent + 1,
                 "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH);");
        emitLine(out, indent, "}");
      }
      if (pmrMode_) {
        const auto tmpVar = nextName("tmp");
        emitLine(out, indent,
                 "std::pmr::vector<" + containerElementType(type) + "> " + tmpVar +
                     "(effective_memory_resource != nullptr ? effective_memory_resource : ::llvmdsdl::cpp::default_memory_resource());");
        emitLine(out, indent, tmpVar + ".resize(" + countVar + ");");
        emitLine(out, indent, expr + " = std::move(" + tmpVar + ");");
      } else {
        emitLine(out, indent, expr + ".resize(" + countVar + ");");
      }

      if (pmrMode_ && type.scalarCategory == SemanticScalarCategory::Composite) {
        const auto initIndex = nextName("i");
        emitLine(out, indent,
                 "for (std::size_t " + initIndex + " = 0U; " + initIndex + " < " +
                     countVar + "; ++" + initIndex + ") {");
        emitLine(out, indent + 1,
                 expr + "[" + initIndex + "].set_memory_resource(effective_memory_resource);");
        emitLine(out, indent, "}");
      }

      countExpr = countVar;
    } else {
      countExpr = std::to_string(type.arrayCapacity) + "U";
    }

    if (elementIsBool && !variable) {
      const auto target = expr + ".data()";
      emitLine(out, indent,
               "dsdl_runtime_get_bits(" + target + ", &buffer[0], capacity_bytes, "
               "offset_bits, " + countExpr + ");");
      emitLine(out, indent, "offset_bits += " + countExpr + ";");
      return;
    }

    const auto index = nextName("index");
    const auto bound = variable ? countExpr : std::to_string(type.arrayCapacity) + "U";
    const auto accessPrefix = expr;

    emitLine(out, indent,
             "for (std::size_t " + index + " = 0U; " + index + " < " + bound +
                 "; ++" + index + ") {");
    const auto elementType = arrayElementType(type);
    emitDeserializeValue(out, elementType,
                         accessPrefix + "[" + index + "]", indent + 1,
                         std::nullopt, fieldFacts);
    emitLine(out, indent, "}");
  }

  void emitSerializeComposite(std::ostringstream &out,
                              const SemanticFieldType &type,
                              const std::string &expr, const int indent,
                              const LoweredFieldFacts *const fieldFacts) {
    if (!type.compositeType) {
      emitLine(out, indent,
               "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_INVALID_ARGUMENT);");
      return;
    }

    const auto nestedType = ctx_.cppQualifiedTypeName(*type.compositeType);
    auto sizeVar = nextName("size_bytes");
    auto errVar = nextName("err");

    if (!type.compositeSealed) {
      emitLine(out, indent,
               "offset_bits += 32U;  // Delimiter header");
    }

    emitLine(out, indent,
             "std::size_t " + sizeVar + " = " +
                 std::to_string((type.bitLengthSet.max() + 7) / 8) + "U;");
    if (!type.compositeSealed) {
      const auto remaining = nextName("remaining");
      emitLine(out, indent,
               "const std::size_t " + remaining +
                   " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, capacity_bytes);");
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto validateRc = nextName("rc");
        emitLine(out, indent,
                 "const std::int8_t " + validateRc + " = " + helper +
                     "(static_cast<std::int64_t>(" + sizeVar + "), static_cast<std::int64_t>(" +
                     remaining + "));");
        emitLine(out, indent, "if (" + validateRc + " < 0) {");
        emitLine(out, indent + 1, "return " + validateRc + ";");
        emitLine(out, indent, "}");
      } else {
        emitLine(out, indent, "if (" + sizeVar + " > " + remaining + ") {");
        emitLine(out, indent + 1,
                 "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER);");
        emitLine(out, indent, "}");
      }
    }
    emitLine(out, indent,
             "std::int8_t " + errVar + " = " + nestedType +
                 "__serialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                 sizeVar + (pmrMode_ ? ", effective_memory_resource" : "") + ");");
    emitLine(out, indent, "if (" + errVar + " < 0) {");
    emitLine(out, indent + 1, "return " + errVar + ";");
    emitLine(out, indent, "}");

    if (!type.compositeSealed) {
      auto hdrErr = nextName("err");
      emitLine(out, indent,
               "const std::int8_t " + hdrErr +
                   " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits - 32U, "
                   "static_cast<std::uint64_t>(" +
                   sizeVar + "), 32U);");
      emitLine(out, indent, "if (" + hdrErr + " < 0) {");
      emitLine(out, indent + 1, "return " + hdrErr + ";");
      emitLine(out, indent, "}");
    }

    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
  }

  void emitDeserializeComposite(std::ostringstream &out,
                                const SemanticFieldType &type,
                                const std::string &expr, const int indent,
                                const LoweredFieldFacts *const fieldFacts) {
    if (!type.compositeType) {
      emitLine(out, indent,
               "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_INVALID_ARGUMENT);");
      return;
    }

    const auto nestedType = ctx_.cppQualifiedTypeName(*type.compositeType);
    auto sizeVar = nextName("size_bytes");
    auto errVar = nextName("err");

    if (!type.compositeSealed) {
      emitLine(out, indent,
               "std::size_t " + sizeVar +
                   " = static_cast<std::size_t>(dsdl_runtime_get_u32(buffer, capacity_bytes, offset_bits, 32U));");
      emitLine(out, indent, "offset_bits += 32U;");
      emitLine(out, indent,
               "const std::size_t _remaining_" + std::to_string(id_) +
                   " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, capacity_bytes);");
      const auto remVar = "_remaining_" + std::to_string(id_);
      ++id_;
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (!helper.empty()) {
        const auto validateRc = nextName("rc");
        emitLine(out, indent,
                 "const std::int8_t " + validateRc + " = " + helper +
                     "(static_cast<std::int64_t>(" + sizeVar + "), static_cast<std::int64_t>(" +
                     remVar + "));");
        emitLine(out, indent, "if (" + validateRc + " < 0) {");
        emitLine(out, indent + 1, "return " + validateRc + ";");
        emitLine(out, indent, "}");
      } else {
        emitLine(out, indent, "if (" + sizeVar + " > " + remVar + ") {");
        emitLine(out, indent + 1,
                 "return static_cast<std::int8_t>(-DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER);");
        emitLine(out, indent, "}");
      }
      const auto consumed = nextName("consumed");
      emitLine(out, indent, "std::size_t " + consumed + " = " + sizeVar + ";");
      emitLine(out, indent,
               "const std::int8_t " + errVar + " = " + nestedType +
                   "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                   consumed + (pmrMode_ ? ", effective_memory_resource" : "") + ");");
      emitLine(out, indent, "if (" + errVar + " < 0) {");
      emitLine(out, indent + 1, "return " + errVar + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
      return;
    }

    emitLine(out, indent,
             "std::size_t " + sizeVar +
                 " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, capacity_bytes);");
    emitLine(out, indent,
             "const std::int8_t " + errVar + " = " + nestedType +
                 "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                 sizeVar + (pmrMode_ ? ", effective_memory_resource" : "") + ");");
    emitLine(out, indent, "if (" + errVar + " < 0) {");
    emitLine(out, indent + 1, "return " + errVar + ";");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
  }
};

std::string cppTypeFromFieldType(const SemanticFieldType &type,
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
    if (type.bitLength == 64U) {
      return "double";
    }
    return "float";
  case SemanticScalarCategory::Void:
    return "std::uint8_t";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      return ctx.cppQualifiedTypeName(*type.compositeType);
    }
    return "std::uint8_t";
  }
  return "std::uint8_t";
}

void collectSectionDependencies(const SemanticSection &section,
                                std::set<std::string> &out) {
  for (const auto &field : section.fields) {
    if (field.resolvedType.compositeType) {
      const auto &ref = *field.resolvedType.compositeType;
      out.insert(loweredTypeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
    }
  }
}

void emitArrayMetadata(std::ostringstream &out, const std::string &typeName,
                       const SemanticSection &section) {
  for (const auto &field : section.fields) {
    if (field.isPadding || field.resolvedType.arrayKind == ArrayKind::None) {
      continue;
    }
    const auto fieldName = sanitizeMacroToken(field.name);
    emitLine(out, 1,
             "static constexpr std::size_t " + fieldName + "_ARRAY_CAPACITY = " +
                 std::to_string(field.resolvedType.arrayCapacity) + "U;");
    emitLine(out, 1,
             "static constexpr bool " + fieldName + "_ARRAY_IS_VARIABLE_LENGTH = " +
                 std::string(isVariableArray(field.resolvedType.arrayKind) ? "true" : "false") + ";");
  }
}

void emitFunctionPrototypes(std::ostringstream &out, const std::string &typeName,
                            const bool pmrMode) {
  emitLine(out, 0, "struct " + typeName + ";");
  emitLine(out, 0,
           "inline std::int8_t " + typeName +
               "__serialize_(const " + typeName +
               "* obj, std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes" +
               (pmrMode ? ", ::llvmdsdl::cpp::MemoryResource* memory_resource" : "") +
               ");");
  emitLine(out, 0,
           "inline std::int8_t " + typeName +
               "__deserialize_(" + typeName +
               "* out_obj, const std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes" +
               (pmrMode ? ", ::llvmdsdl::cpp::MemoryResource* memory_resource" : "") +
               ");");
  out << "\n";
}

void emitSectionStruct(std::ostringstream &out, const std::string &typeName,
                       const std::string &fullName,
                       std::uint32_t majorVersion,
                       std::uint32_t minorVersion,
                       const SemanticSection &section,
                       const EmitterContext &ctx,
                       const bool pmrMode) {
  emitLine(out, 0, "struct " + typeName + " {");

  std::size_t emitted = 0;
  std::vector<std::string> variableArrayMembers;
  std::vector<std::string> compositeScalarMembers;
  std::vector<std::string> compositeFixedArrayMembers;
  std::vector<std::string> compositeVariableArrayMembers;

  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }

    const auto member = sanitizeIdentifier(field.name);
    const auto baseType = cppTypeFromFieldType(field.resolvedType, ctx);

    if (field.resolvedType.arrayKind == ArrayKind::None) {
      emitLine(out, 1, baseType + " " + member + "{};");
      if (pmrMode &&
          field.resolvedType.scalarCategory == SemanticScalarCategory::Composite) {
        compositeScalarMembers.push_back(member);
      }
      ++emitted;
      continue;
    }

    if (field.resolvedType.arrayKind == ArrayKind::Fixed) {
      if (field.resolvedType.scalarCategory == SemanticScalarCategory::Bool) {
        emitLine(out, 1,
                 "std::array<std::uint8_t, (" +
                     std::to_string(field.resolvedType.arrayCapacity) +
                     "U + 7U) / 8U> " + member + "{};");
      } else {
        emitLine(out, 1,
                 "std::array<" + baseType + ", " +
                     std::to_string(field.resolvedType.arrayCapacity) + "U> " + member +
                     "{};");
      }
      if (pmrMode &&
          field.resolvedType.scalarCategory == SemanticScalarCategory::Composite) {
        compositeFixedArrayMembers.push_back(member);
      }
      ++emitted;
      continue;
    }

    if (pmrMode) {
      emitLine(out, 1,
               "std::pmr::vector<" +
                   std::string(field.resolvedType.scalarCategory ==
                                       SemanticScalarCategory::Bool
                                   ? "bool"
                                   : baseType) +
                   "> " + member + "{};");
      variableArrayMembers.push_back(member);
      if (field.resolvedType.scalarCategory == SemanticScalarCategory::Composite) {
        compositeVariableArrayMembers.push_back(member);
      }
    } else {
      emitLine(out, 1,
               "std::vector<" +
                   std::string(field.resolvedType.scalarCategory ==
                                       SemanticScalarCategory::Bool
                                   ? "bool"
                                   : baseType) +
                   "> " + member + "{};");
    }
    ++emitted;
  }

  if (section.isUnion) {
    emitLine(out, 1, "std::uint8_t _tag_{0U};");
    ++emitted;
  }

  if (pmrMode) {
    emitLine(out, 1,
             "::llvmdsdl::cpp::MemoryResource* _memory_resource{::llvmdsdl::cpp::default_memory_resource()};");
    emitLine(out, 1, typeName + "() = default;");
    emitLine(out, 1,
             "explicit " + typeName +
                 "(::llvmdsdl::cpp::MemoryResource* memory_resource) { set_memory_resource(memory_resource); }");
    emitLine(out, 1,
             "void set_memory_resource(::llvmdsdl::cpp::MemoryResource* memory_resource) {");
    emitLine(out, 2,
             "_memory_resource = (memory_resource != nullptr) ? memory_resource : ::llvmdsdl::cpp::default_memory_resource();");
    for (const auto &member : variableArrayMembers) {
      emitLine(out, 2,
               member + " = decltype(" + member + ")(_memory_resource);");
    }
    for (const auto &member : compositeScalarMembers) {
      emitLine(out, 2, member + ".set_memory_resource(_memory_resource);");
    }
    for (const auto &member : compositeFixedArrayMembers) {
      const auto i = sanitizeIdentifier(member + "_index");
      emitLine(out, 2,
               "for (std::size_t " + i + " = 0U; " + i + " < " + member +
                   ".size(); ++" + i + ") {");
      emitLine(out, 3, member + "[" + i + "].set_memory_resource(_memory_resource);");
      emitLine(out, 2, "}");
    }
    for (const auto &member : compositeVariableArrayMembers) {
      const auto i = sanitizeIdentifier(member + "_index");
      emitLine(out, 2,
               "for (std::size_t " + i + " = 0U; " + i + " < " + member +
                   ".size(); ++" + i + ") {");
      emitLine(out, 3, member + "[" + i + "].set_memory_resource(_memory_resource);");
      emitLine(out, 2, "}");
    }
    emitLine(out, 1, "}");
    ++emitted;
  }

  if (emitted == 0) {
    emitLine(out, 1, "std::uint8_t _dummy_{0U};");
  }

  emitLine(out, 1,
           "static constexpr const char* FULL_NAME = \"" + fullName + "\";");
  emitLine(out, 1,
           "static constexpr const char* FULL_NAME_AND_VERSION = \"" + fullName +
               "." + std::to_string(majorVersion) + "." +
               std::to_string(minorVersion) + "\";");
  emitLine(out, 1,
           "static constexpr std::size_t EXTENT_BYTES = " +
               std::to_string(section.extentBits.value_or(0) / 8) + "U;");
  emitLine(out, 1,
           "static constexpr std::size_t SERIALIZATION_BUFFER_SIZE_BYTES = " +
               std::to_string((section.serializationBufferSizeBits + 7) / 8) + "U;");
  if (section.isUnion) {
    std::size_t optionCount = 0;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        ++optionCount;
      }
    }
    emitLine(out, 1,
             "static constexpr std::size_t UNION_OPTION_COUNT = " +
                 std::to_string(optionCount) + "U;");
  }

  for (const auto &c : section.constants) {
    emitLine(out, 1,
             "static constexpr auto " + sanitizeMacroToken(c.name) + " = " +
                 valueToCppExpr(c.value) + ";");
  }

  emitArrayMetadata(out, typeName, section);

  emitLine(out, 1,
           "[[nodiscard]] inline std::int8_t serialize(std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes) const {");
  if (pmrMode) {
    emitLine(out, 2,
             "return " + typeName + "__serialize_(this, buffer, inout_buffer_size_bytes, _memory_resource);");
  } else {
    emitLine(out, 2,
             "return " + typeName + "__serialize_(this, buffer, inout_buffer_size_bytes);");
  }
  emitLine(out, 1, "}");

  emitLine(out, 1,
           "[[nodiscard]] inline std::int8_t deserialize(const std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes) {");
  if (pmrMode) {
    emitLine(out, 2,
             "return " + typeName + "__deserialize_(this, buffer, inout_buffer_size_bytes, _memory_resource);");
  } else {
    emitLine(out, 2,
             "return " + typeName + "__deserialize_(this, buffer, inout_buffer_size_bytes);");
  }
  emitLine(out, 1, "}");

  if (pmrMode) {
    emitLine(out, 1,
             "[[nodiscard]] inline std::int8_t serialize(std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes, ::llvmdsdl::cpp::MemoryResource* memory_resource) const {");
    emitLine(out, 2,
             "return " + typeName + "__serialize_(this, buffer, inout_buffer_size_bytes, memory_resource);");
    emitLine(out, 1, "}");

    emitLine(out, 1,
             "[[nodiscard]] inline std::int8_t deserialize(const std::uint8_t* buffer, std::size_t* inout_buffer_size_bytes, ::llvmdsdl::cpp::MemoryResource* memory_resource) {");
    emitLine(out, 2,
             "return " + typeName + "__deserialize_(this, buffer, inout_buffer_size_bytes, memory_resource);");
    emitLine(out, 1, "}");
  }

  emitLine(out, 0, "};");
  out << "\n";
}

void emitSection(std::ostringstream &out, const EmitterContext &ctx,
                 const SemanticDefinition &def, const std::string &typeName,
                 const std::string &fullName, const SemanticSection &section,
                 const bool pmrMode,
                 const LoweredSectionFacts *const sectionFacts) {
  (void)def;
  emitFunctionPrototypes(out, typeName, pmrMode);
  emitSectionStruct(out, typeName, fullName, def.info.majorVersion,
                    def.info.minorVersion, section, ctx, pmrMode);

  FunctionBodyEmitter bodyEmitter(ctx, pmrMode);
  bodyEmitter.emitSerializeFunction(out, typeName, section, sectionFacts);
  bodyEmitter.emitDeserializeFunction(out, typeName, section, sectionFacts);
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

llvm::Expected<std::string> loadCRuntimeHeader() {
  const std::filesystem::path absoluteRuntimeHeader =
      std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "dsdl_runtime.h";
  std::ifstream in(absoluteRuntimeHeader.string());
  if (!in) {
    in.open("runtime/dsdl_runtime.h");
  }
  if (!in) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to read runtime header");
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

llvm::Expected<std::string> loadCppRuntimeHeader() {
  const std::filesystem::path absoluteRuntimeHeader =
      std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "cpp" /
      "dsdl_runtime.hpp";
  std::ifstream in(absoluteRuntimeHeader.string());
  if (!in) {
    in.open("runtime/cpp/dsdl_runtime.hpp");
  }
  if (!in) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to read C++ runtime header");
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

std::string renderHeader(const SemanticDefinition &def, const EmitterContext &ctx,
                         const bool pmrMode,
                         const LoweredFactsMap &loweredFacts) {
  std::ostringstream out;
  const auto guard = headerGuard(def.info);
  const auto baseTypeName = ctx.cppTypeName(def);

  std::set<std::string> dependencies;
  collectSectionDependencies(def.request, dependencies);
  if (def.response) {
    collectSectionDependencies(*def.response, dependencies);
  }

  out << "#ifndef " << guard << "\n";
  out << "#define " << guard << "\n\n";
  out << "#include <array>\n";
  out << "#include <cstddef>\n";
  out << "#include <cstdint>\n";
  out << "#include <utility>\n";
  out << "#include <vector>\n";
  if (pmrMode) {
    out << "#include <memory_resource>\n";
  }
  out << "#include \"dsdl_runtime.hpp\"\n";

  for (const auto &depKey : dependencies) {
    auto split0 = depKey.find(':');
    auto split1 = depKey.find(':', split0 + 1);
    if (split0 == std::string::npos || split1 == std::string::npos) {
      continue;
    }
    SemanticTypeRef ref;
    ref.fullName = depKey.substr(0, split0);
    ref.majorVersion = static_cast<std::uint32_t>(
        std::stoul(depKey.substr(split0 + 1, split1 - split0 - 1)));
    ref.minorVersion = static_cast<std::uint32_t>(
        std::stoul(depKey.substr(split1 + 1)));
    if (const auto *dep = ctx.find(ref)) {
      out << "#include \"" << ctx.relativeHeaderPath(*dep) << "\"\n";
    }
  }
  out << "\n";
  emitNamespaceOpen(out, def.info.namespaceComponents);

  if (def.isService) {
    const auto requestType = baseTypeName + "__Request";
    const auto responseType = baseTypeName + "__Response";

    emitLine(out, 0,
             "inline constexpr const char* " + baseTypeName +
                 "_FULL_NAME = \"" + def.info.fullName + "\";");
    emitLine(out, 0,
             "inline constexpr const char* " + baseTypeName +
                 "_FULL_NAME_AND_VERSION = \"" + def.info.fullName + "." +
                 std::to_string(def.info.majorVersion) + "." +
                 std::to_string(def.info.minorVersion) + "\";");
    out << "\n";

    emitSection(out, ctx, def, requestType, def.info.fullName + ".Request",
                def.request, pmrMode,
                findLoweredSectionFacts(loweredFacts, def, "request"));
    if (def.response) {
      emitSection(out, ctx, def, responseType, def.info.fullName + ".Response",
                  *def.response, pmrMode,
                  findLoweredSectionFacts(loweredFacts, def, "response"));
    }

    emitLine(out, 0, "using " + baseTypeName + " = " + requestType + ";");
    emitLine(out, 0,
             "inline constexpr std::size_t " + baseTypeName +
                 "_EXTENT_BYTES = " + requestType + "::EXTENT_BYTES;");
    emitLine(out, 0,
             "inline constexpr std::size_t " + baseTypeName +
                 "_SERIALIZATION_BUFFER_SIZE_BYTES = " + requestType +
                 "::SERIALIZATION_BUFFER_SIZE_BYTES;");
    out << "\n";

    emitLine(out, 0,
             "inline std::int8_t " + baseTypeName +
                 "__serialize_(const " + baseTypeName +
                 "* const obj, std::uint8_t* const buffer, std::size_t* const "
                 "inout_buffer_size_bytes" +
                 (pmrMode ? ", ::llvmdsdl::cpp::MemoryResource* const memory_resource" : "") +
                 ")");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "return " + requestType + "__serialize_(reinterpret_cast<const " +
                 requestType + "*>(obj), buffer, inout_buffer_size_bytes" +
                 (pmrMode ? ", memory_resource" : "") + ");");
    emitLine(out, 0, "}");
    out << "\n";

    emitLine(out, 0,
             "inline std::int8_t " + baseTypeName +
                 "__deserialize_(" + baseTypeName +
                 "* const out_obj, const std::uint8_t* buffer, std::size_t* const "
                 "inout_buffer_size_bytes" +
                 (pmrMode ? ", ::llvmdsdl::cpp::MemoryResource* const memory_resource" : "") +
                 ")");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "return " + requestType + "__deserialize_(reinterpret_cast<" +
                 requestType + "*>(out_obj), buffer, inout_buffer_size_bytes" +
                 (pmrMode ? ", memory_resource" : "") + ");");
    emitLine(out, 0, "}");
  } else {
    emitSection(out, ctx, def, baseTypeName, def.info.fullName, def.request,
                pmrMode, findLoweredSectionFacts(loweredFacts, def, ""));
  }

  emitNamespaceClose(out, def.info.namespaceComponents);
  out << "\n#endif /* " << guard << " */\n";
  return out.str();
}

llvm::Error emitProfile(const SemanticModule &semantic,
                        const std::filesystem::path &outRoot,
                        const bool pmrMode,
                        const LoweredFactsMap &loweredFacts) {
  std::filesystem::create_directories(outRoot);

  auto cRuntime = loadCRuntimeHeader();
  if (!cRuntime) {
    return cRuntime.takeError();
  }
  if (auto err = writeFile(outRoot / "dsdl_runtime.h", *cRuntime)) {
    return err;
  }

  auto cppRuntime = loadCppRuntimeHeader();
  if (!cppRuntime) {
    return cppRuntime.takeError();
  }
  if (auto err = writeFile(outRoot / "dsdl_runtime.hpp", *cppRuntime)) {
    return err;
  }

  EmitterContext ctx(semantic);
  for (const auto &def : semantic.definitions) {
    std::filesystem::path dir = outRoot;
    for (const auto &ns : def.info.namespaceComponents) {
      dir /= ns;
    }
    std::filesystem::create_directories(dir);

    if (auto err = writeFile(dir / headerFileName(def.info),
                             renderHeader(def, ctx, pmrMode, loweredFacts))) {
      return err;
    }
  }

  return llvm::Error::success();
}

} // namespace

llvm::Error emitCpp(const SemanticModule &semantic, mlir::ModuleOp module,
                    const CppEmitOptions &options,
                    DiagnosticEngine &diagnostics) {
  if (options.outDir.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  }
  LoweredFactsMap loweredFacts;
  if (!collectLoweredFactsFromMlir(semantic, module, diagnostics, "C++",
                                   &loweredFacts)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "MLIR schema coverage validation failed for C++ emission");
  }

  std::filesystem::path outRoot(options.outDir);
  std::filesystem::create_directories(outRoot);

  if (options.profile == CppProfile::Std) {
    return emitProfile(semantic, outRoot, false, loweredFacts);
  }
  if (options.profile == CppProfile::Pmr) {
    return emitProfile(semantic, outRoot, true, loweredFacts);
  }

  if (auto err = emitProfile(semantic, outRoot / "std", false, loweredFacts)) {
    return err;
  }
  return emitProfile(semantic, outRoot / "pmr", true, loweredFacts);
}

} // namespace llvmdsdl
