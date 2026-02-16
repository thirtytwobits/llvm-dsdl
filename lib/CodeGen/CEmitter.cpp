#include "llvmdsdl/CodeGen/CEmitter.h"

#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

namespace llvmdsdl {
namespace {

std::string typeKey(const std::string &name, std::uint32_t major,
                    std::uint32_t minor) {
  return name + ":" + std::to_string(major) + ":" + std::to_string(minor);
}

bool isCKeyword(const std::string &name) {
  static const std::set<std::string> kKeywords = {
      "auto",       "break",      "case",      "char",      "const",
      "continue",   "default",    "do",        "double",    "else",
      "enum",       "extern",     "float",     "for",       "goto",
      "if",         "inline",     "int",       "long",      "register",
      "restrict",   "return",     "short",     "signed",    "sizeof",
      "static",     "struct",     "switch",    "typedef",   "union",
      "unsigned",   "void",       "volatile",  "while",     "_Alignas",
      "_Alignof",   "_Atomic",    "_Bool",     "_Complex",  "_Generic",
      "_Imaginary", "_Noreturn",  "_Static_assert", "_Thread_local", "true",
      "false"};
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
  if (isCKeyword(name)) {
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
  return llvm::formatv("{0}_{1}_{2}.h", info.shortName, info.majorVersion,
                       info.minorVersion)
      .str();
}

std::string cTypeNameFromInfo(const DiscoveredDefinition &info) {
  std::string out;
  for (std::size_t i = 0; i < info.namespaceComponents.size(); ++i) {
    if (i > 0) {
      out += "__";
    }
    out += sanitizeIdentifier(info.namespaceComponents[i]);
  }
  if (!out.empty()) {
    out += "__";
  }
  out += sanitizeIdentifier(info.shortName);
  return out;
}

std::string headerGuard(const DiscoveredDefinition &info) {
  std::string g = "LLVMDSDL_" + info.fullName + "_" +
                  std::to_string(info.majorVersion) + "_" +
                  std::to_string(info.minorVersion) + "_H";
  for (char &c : g) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    } else {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
  }
  return g;
}

std::string valueToCExpr(const Value &value) {
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

std::uint32_t scalarStorageBits(const std::uint32_t bitLength) {
  if (bitLength <= 8) {
    return 8;
  }
  if (bitLength <= 16) {
    return 16;
  }
  if (bitLength <= 32) {
    return 32;
  }
  return 64;
}

std::string unsignedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "uint8_t";
  case 16:
    return "uint16_t";
  case 32:
    return "uint32_t";
  default:
    return "uint64_t";
  }
}

std::string signedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "int8_t";
  case 16:
    return "int16_t";
  case 32:
    return "int32_t";
  default:
    return "int64_t";
  }
}

std::string unsignedGetter(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "dsdl_runtime_get_u8";
  case 16:
    return "dsdl_runtime_get_u16";
  case 32:
    return "dsdl_runtime_get_u32";
  default:
    return "dsdl_runtime_get_u64";
  }
}

std::string signedGetter(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "dsdl_runtime_get_i8";
  case 16:
    return "dsdl_runtime_get_i16";
  case 32:
    return "dsdl_runtime_get_i32";
  default:
    return "dsdl_runtime_get_i64";
  }
}

bool isVariableArray(const ArrayKind k) {
  return k == ArrayKind::VariableInclusive || k == ArrayKind::VariableExclusive;
}

class EmitterContext final {
public:
  explicit EmitterContext(const SemanticModule &semantic) {
    for (const auto &def : semantic.definitions) {
      const auto key = typeKey(def.info.fullName, def.info.majorVersion,
                               def.info.minorVersion);
      byKey_.emplace(key, &def);
    }
  }

  const SemanticDefinition *find(const SemanticTypeRef &ref) const {
    const auto it = byKey_.find(typeKey(ref.fullName, ref.majorVersion,
                                        ref.minorVersion));
    if (it == byKey_.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::string cTypeName(const SemanticDefinition &def) const {
    return cTypeNameFromInfo(def.info);
  }

  std::string cTypeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return cTypeName(*def);
    }

    DiscoveredDefinition tmp;
    tmp.fullName = ref.fullName;
    tmp.shortName = ref.shortName;
    tmp.namespaceComponents = ref.namespaceComponents;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    return cTypeNameFromInfo(tmp);
  }

  std::string relativeHeaderPath(const SemanticDefinition &def) const {
    std::filesystem::path p;
    for (const auto &ns : def.info.namespaceComponents) {
      p /= ns;
    }
    p /= headerFileName(def.info);
    return p.generic_string();
  }

  std::string relativeHeaderPath(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return relativeHeaderPath(*def);
    }
    std::filesystem::path p;
    for (const auto &ns : ref.namespaceComponents) {
      p /= ns;
    }
    DiscoveredDefinition tmp;
    tmp.shortName = ref.shortName;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    p /= headerFileName(tmp);
    return p.generic_string();
  }

private:
  std::unordered_map<std::string, const SemanticDefinition *> byKey_;
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
  explicit FunctionBodyEmitter(const EmitterContext &ctx) : ctx_(ctx) {}

  void emitSerializeFunction(std::ostringstream &out, const std::string &typeName,
                             const SemanticSection &section) {
    emitLine(out, 0, "static inline int8_t " + typeName +
                        "__serialize_(const " + typeName +
                        "* const obj, uint8_t* const buffer, size_t* const "
                        "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1, "if ((obj == NULL) || (buffer == NULL) || (inout_buffer_size_bytes == NULL)) {");
    emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "if ((capacity_bytes * 8U) < " +
                        std::to_string(section.serializationBufferSizeBits) + "ULL) {");
    emitLine(out, 2,
             "return -(int8_t)DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "size_t offset_bits = 0U;");

    if (section.isUnion) {
      emitSerializeUnion(out, section, "obj", 1);
    } else {
      for (const auto &field : section.fields) {
        emitAlign(out, field.resolvedType.alignmentBits, 1);
        if (field.isPadding) {
          emitSerializePadding(out, field.resolvedType, 1);
        } else {
          emitSerializeValue(out, field.resolvedType,
                             "obj->" + sanitizeIdentifier(field.name), 1);
        }
      }
    }

    emitAlign(out, 8, 1);
    emitLine(out, 1, "*inout_buffer_size_bytes = (size_t)(offset_bits / 8U);");
    emitLine(out, 1, "return (int8_t)DSDL_RUNTIME_SUCCESS;");
    emitLine(out, 0, "}");
    out << "\n";
  }

  void emitDeserializeFunction(std::ostringstream &out,
                               const std::string &typeName,
                               const SemanticSection &section) {
    emitLine(out, 0, "static inline int8_t " + typeName +
                        "__deserialize_(" + typeName +
                        "* const out_obj, const uint8_t* buffer, size_t* const "
                        "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "if ((out_obj == NULL) || (inout_buffer_size_bytes == NULL) || ((buffer == NULL) && (0U != *inout_buffer_size_bytes))) {");
    emitLine(out, 2, "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (buffer == NULL) {");
    emitLine(out, 2, "buffer = (const uint8_t*)\"\";");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const size_t capacity_bytes = *inout_buffer_size_bytes;");
    emitLine(out, 1, "const size_t capacity_bits = capacity_bytes * 8U;");
    emitLine(out, 1, "size_t offset_bits = 0U;");

    if (section.isUnion) {
      emitDeserializeUnion(out, section, "out_obj", 1);
    } else {
      for (const auto &field : section.fields) {
        emitAlign(out, field.resolvedType.alignmentBits, 1);
        if (field.isPadding) {
          emitDeserializePadding(out, field.resolvedType, 1);
        } else {
          emitDeserializeValue(out, field.resolvedType,
                               "out_obj->" + sanitizeIdentifier(field.name), 1);
        }
      }
    }

    emitAlign(out, 8, 1);
    emitLine(out, 1,
             "*inout_buffer_size_bytes = (size_t)(dsdl_runtime_choose_min(offset_bits, capacity_bits) / 8U);");
    emitLine(out, 1, "return (int8_t)DSDL_RUNTIME_SUCCESS;");
    emitLine(out, 0, "}");
    out << "\n";
  }

private:
  const EmitterContext &ctx_;
  std::size_t id_{0};

  std::string nextName(const std::string &prefix) {
    return "_" + prefix + std::to_string(id_++) + "_";
  }

  void emitAlign(std::ostringstream &out, const std::int64_t alignmentBits,
                 const int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    emitLine(out, indent,
             "offset_bits = (offset_bits + " + std::to_string(alignmentBits - 1) +
                 "U) & ~(size_t)" + std::to_string(alignmentBits - 1) + "U;");
  }

  void emitSerializePadding(std::ostringstream &out, const SemanticFieldType &type,
                            const int indent) {
    if (type.bitLength == 0) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             "const int8_t " + err +
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

  void emitSerializeUnion(std::ostringstream &out, const SemanticSection &section,
                          const std::string &objRef, const int indent) {
    std::uint32_t tagBits = 8;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        tagBits = std::max<std::uint32_t>(8U, f.unionTagBits);
        break;
      }
    }

    const auto tagErr = nextName("err");
    emitLine(out, indent,
             "const int8_t " + tagErr +
                 " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, " +
                 "(uint64_t)(" + objRef + "->_tag_), " +
                 std::to_string(tagBits) + "U);");
    emitLine(out, indent, "if (" + tagErr + " < 0) {");
    emitLine(out, indent + 1, "return " + tagErr + ";");
    emitLine(out, indent, "}");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + "U;");

    bool first = true;
    for (const auto &field : section.fields) {
      if (field.isPadding) {
        continue;
      }
      const auto member = sanitizeIdentifier(field.name);
      emitLine(out, indent,
               std::string(first ? "if" : "else if") + " (" +
                   objRef + "->_tag_ == " +
                   std::to_string(field.unionOptionIndex) + "U) {");
      emitAlign(out, field.resolvedType.alignmentBits, indent + 1);
      emitSerializeValue(out, field.resolvedType,
                         objRef + "->" + member, indent + 1);
      emitLine(out, indent, "}");
      first = false;
    }

    emitLine(out, indent, "else {");
    emitLine(out, indent + 1,
             "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG;");
    emitLine(out, indent, "}");
  }

  void emitDeserializeUnion(std::ostringstream &out,
                            const SemanticSection &section,
                            const std::string &objRef, const int indent) {
    std::uint32_t tagBits = 8;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        tagBits = std::max<std::uint32_t>(8U, f.unionTagBits);
        break;
      }
    }

    emitLine(out, indent,
             objRef + "->_tag_ = (uint8_t)" + unsignedGetter(tagBits) +
                 "(buffer, capacity_bytes, offset_bits, " +
                 std::to_string(tagBits) + "U);");
    emitLine(out, indent,
             "offset_bits += " + std::to_string(tagBits) + "U;");

    bool first = true;
    for (const auto &field : section.fields) {
      if (field.isPadding) {
        continue;
      }
      const auto member = sanitizeIdentifier(field.name);
      emitLine(out, indent,
               std::string(first ? "if" : "else if") + " (" +
                   objRef + "->_tag_ == " +
                   std::to_string(field.unionOptionIndex) + "U) {");
      emitAlign(out, field.resolvedType.alignmentBits, indent + 1);
      emitDeserializeValue(out, field.resolvedType,
                           objRef + "->" + member, indent + 1);
      emitLine(out, indent, "}");
      first = false;
    }

    emitLine(out, indent, "else {");
    emitLine(out, indent + 1,
             "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG;");
    emitLine(out, indent, "}");
  }

  void emitSerializeValue(std::ostringstream &out, const SemanticFieldType &type,
                          const std::string &expr, const int indent) {
    if (type.arrayKind != ArrayKind::None) {
      emitSerializeArray(out, type, expr, indent);
      return;
    }

    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool: {
      const auto err = nextName("err");
      emitLine(out, indent,
               "const int8_t " + err +
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
      std::string valueExpr = "(uint64_t)(" + expr + ")";
      if (type.castMode == CastMode::Saturated && type.bitLength < 64U) {
        const auto sat = nextName("sat");
        const auto maxVal = (1ULL << type.bitLength) - 1ULL;
        emitLine(out, indent,
                 "uint64_t " + sat + " = (uint64_t)(" + expr + ");");
        emitLine(out, indent,
                 "if (" + sat + " > " + std::to_string(maxVal) + "ULL) {");
        emitLine(out, indent + 1,
                 sat + " = " + std::to_string(maxVal) + "ULL;");
        emitLine(out, indent, "}");
        valueExpr = sat;
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "const int8_t " + err +
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
      std::string valueExpr = "(int64_t)(" + expr + ")";
      if (type.castMode == CastMode::Saturated && type.bitLength < 64U &&
          type.bitLength > 0U) {
        const auto sat = nextName("sat");
        const auto minVal = -(1LL << (type.bitLength - 1U));
        const auto maxVal = (1LL << (type.bitLength - 1U)) - 1LL;
        emitLine(out, indent,
                 "int64_t " + sat + " = (int64_t)(" + expr + ");");
        emitLine(out, indent,
                 "if (" + sat + " < " + std::to_string(minVal) + "LL) {");
        emitLine(out, indent + 1,
                 sat + " = " + std::to_string(minVal) + "LL;");
        emitLine(out, indent, "}");
        emitLine(out, indent,
                 "if (" + sat + " > " + std::to_string(maxVal) + "LL) {");
        emitLine(out, indent + 1,
                 sat + " = " + std::to_string(maxVal) + "LL;");
        emitLine(out, indent, "}");
        valueExpr = sat;
      }
      const auto err = nextName("err");
      emitLine(out, indent,
               "const int8_t " + err +
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
      std::string call;
      if (type.bitLength == 16U) {
        call = "dsdl_runtime_set_f16(buffer, capacity_bytes, offset_bits, (float)(" +
               expr + "))";
      } else if (type.bitLength == 32U) {
        call = "dsdl_runtime_set_f32(buffer, capacity_bytes, offset_bits, (float)(" +
               expr + "))";
      } else {
        call = "dsdl_runtime_set_f64(buffer, capacity_bytes, offset_bits, (double)(" +
               expr + "))";
      }
      emitLine(out, indent, "const int8_t " + err + " = " + call + ";");
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
      emitSerializeComposite(out, type, expr, indent);
      break;
    }
  }

  void emitDeserializeValue(std::ostringstream &out, const SemanticFieldType &type,
                            const std::string &expr, const int indent) {
    if (type.arrayKind != ArrayKind::None) {
      emitDeserializeArray(out, type, expr, indent);
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
    case SemanticScalarCategory::UnsignedInt:
      emitLine(out, indent,
               expr + " = (" + unsignedStorageType(type.bitLength) + ")" +
                   unsignedGetter(type.bitLength) +
                   "(buffer, capacity_bytes, offset_bits, " +
                   std::to_string(type.bitLength) + "U);");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    case SemanticScalarCategory::SignedInt:
      emitLine(out, indent,
               expr + " = (" + signedStorageType(type.bitLength) + ")" +
                   signedGetter(type.bitLength) +
                   "(buffer, capacity_bytes, offset_bits, " +
                   std::to_string(type.bitLength) + "U);");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    case SemanticScalarCategory::Float:
      if (type.bitLength == 16U) {
        emitLine(out, indent,
                 expr + " = dsdl_runtime_get_f16(buffer, capacity_bytes, offset_bits);");
      } else if (type.bitLength == 32U) {
        emitLine(out, indent,
                 expr + " = dsdl_runtime_get_f32(buffer, capacity_bytes, offset_bits);");
      } else {
        emitLine(out, indent,
                 expr + " = dsdl_runtime_get_f64(buffer, capacity_bytes, offset_bits);");
      }
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.bitLength) + "U;");
      break;
    case SemanticScalarCategory::Void:
      emitDeserializePadding(out, type, indent);
      break;
    case SemanticScalarCategory::Composite:
      emitDeserializeComposite(out, type, expr, indent);
      break;
    }
  }

  void emitSerializeArray(std::ostringstream &out, const SemanticFieldType &type,
                          const std::string &expr, const int indent) {
    const bool elementIsBool = type.scalarCategory == SemanticScalarCategory::Bool;
    const bool variable = isVariableArray(type.arrayKind);

    if (variable) {
      emitLine(out, indent,
               "if (" + expr + ".count > " + std::to_string(type.arrayCapacity) +
                   "U) {");
      emitLine(out, indent + 1,
               "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH;");
      emitLine(out, indent, "}");

      const auto err = nextName("err");
      emitLine(out, indent,
               "const int8_t " + err +
                   " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits, "
                   "(uint64_t)" +
                   expr + ".count, " +
                   std::to_string(type.arrayLengthPrefixBits) + "U);");
      emitLine(out, indent, "if (" + err + " < 0) {");
      emitLine(out, indent + 1, "return " + err + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.arrayLengthPrefixBits) + "U;");
    }

    if (elementIsBool) {
      const auto source = variable ? ("&" + expr + ".bitpacked[0]") : ("&" + expr + "[0]");
      const auto countExpr = variable ? (expr + ".count")
                                      : std::to_string(type.arrayCapacity) + "U";
      emitLine(out, indent,
               "dsdl_runtime_copy_bits(&buffer[0], offset_bits, " + countExpr +
                   ", " + source + ", 0U);");
      emitLine(out, indent, "offset_bits += " + countExpr + ";");
      return;
    }

    const auto index = nextName("index");
    const auto bound = variable ? (expr + ".count")
                                : std::to_string(type.arrayCapacity) + "U";
    const auto accessPrefix = variable ? (expr + ".elements") : expr;

    emitLine(out, indent,
             "for (size_t " + index + " = 0U; " + index + " < " + bound +
                 "; ++" + index + ") {");
    SemanticFieldType elementType = type;
    elementType.arrayKind = ArrayKind::None;
    elementType.arrayCapacity = 0;
    elementType.arrayLengthPrefixBits = 0;
    emitSerializeValue(out, elementType,
                       accessPrefix + "[" + index + "]", indent + 1);
    emitLine(out, indent, "}");
  }

  void emitDeserializeArray(std::ostringstream &out,
                            const SemanticFieldType &type,
                            const std::string &expr, const int indent) {
    const bool elementIsBool = type.scalarCategory == SemanticScalarCategory::Bool;
    const bool variable = isVariableArray(type.arrayKind);

    if (variable) {
      emitLine(out, indent,
               expr + ".count = (size_t)" +
                   unsignedGetter(static_cast<std::uint32_t>(type.arrayLengthPrefixBits)) +
                   "(buffer, capacity_bytes, offset_bits, " +
                   std::to_string(type.arrayLengthPrefixBits) + "U);");
      emitLine(out, indent,
               "offset_bits += " + std::to_string(type.arrayLengthPrefixBits) + "U;");
      emitLine(out, indent,
               "if (" + expr + ".count > " + std::to_string(type.arrayCapacity) +
                   "U) {");
      emitLine(out, indent + 1,
               "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_ARRAY_LENGTH;");
      emitLine(out, indent, "}");
    }

    if (elementIsBool) {
      const auto target = variable ? ("&" + expr + ".bitpacked[0]") : ("&" + expr + "[0]");
      const auto countExpr = variable ? (expr + ".count")
                                      : std::to_string(type.arrayCapacity) + "U";
      emitLine(out, indent,
               "dsdl_runtime_get_bits(" + target + ", &buffer[0], capacity_bytes, "
               "offset_bits, " + countExpr + ");");
      emitLine(out, indent, "offset_bits += " + countExpr + ";");
      return;
    }

    const auto index = nextName("index");
    const auto bound = variable ? (expr + ".count")
                                : std::to_string(type.arrayCapacity) + "U";
    const auto accessPrefix = variable ? (expr + ".elements") : expr;

    emitLine(out, indent,
             "for (size_t " + index + " = 0U; " + index + " < " + bound +
                 "; ++" + index + ") {");
    SemanticFieldType elementType = type;
    elementType.arrayKind = ArrayKind::None;
    elementType.arrayCapacity = 0;
    elementType.arrayLengthPrefixBits = 0;
    emitDeserializeValue(out, elementType,
                         accessPrefix + "[" + index + "]", indent + 1);
    emitLine(out, indent, "}");
  }

  void emitSerializeComposite(std::ostringstream &out,
                              const SemanticFieldType &type,
                              const std::string &expr, const int indent) {
    if (!type.compositeType) {
      emitLine(out, indent,
               "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
      return;
    }

    const auto nestedType = ctx_.cTypeName(*type.compositeType);
    auto sizeVar = nextName("size_bytes");
    auto errVar = nextName("err");

    if (!type.compositeSealed) {
      emitLine(out, indent,
               "offset_bits += 32U;  // Delimiter header");
    }

    emitLine(out, indent,
             "size_t " + sizeVar + " = " +
                 std::to_string((type.bitLengthSet.max() + 7) / 8) + "U;");
    emitLine(out, indent,
             "int8_t " + errVar + " = " + nestedType +
                 "__serialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                 sizeVar + ");");
    emitLine(out, indent, "if (" + errVar + " < 0) {");
    emitLine(out, indent + 1, "return " + errVar + ";");
    emitLine(out, indent, "}");

    if (!type.compositeSealed) {
      auto hdrErr = nextName("err");
      emitLine(out, indent,
               "const int8_t " + hdrErr +
                   " = dsdl_runtime_set_uxx(buffer, capacity_bytes, offset_bits - 32U, "
                   "(uint64_t)" +
                   sizeVar + ", 32U);");
      emitLine(out, indent, "if (" + hdrErr + " < 0) {");
      emitLine(out, indent + 1, "return " + hdrErr + ";");
      emitLine(out, indent, "}");
    }

    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
  }

  void emitDeserializeComposite(std::ostringstream &out,
                                const SemanticFieldType &type,
                                const std::string &expr, const int indent) {
    if (!type.compositeType) {
      emitLine(out, indent,
               "return -(int8_t)DSDL_RUNTIME_ERROR_INVALID_ARGUMENT;");
      return;
    }

    const auto nestedType = ctx_.cTypeName(*type.compositeType);
    auto sizeVar = nextName("size_bytes");
    auto errVar = nextName("err");

    if (!type.compositeSealed) {
      emitLine(out, indent,
               "size_t " + sizeVar +
                   " = (size_t)dsdl_runtime_get_u32(buffer, capacity_bytes, offset_bits, 32U);");
      emitLine(out, indent, "offset_bits += 32U;");
      emitLine(out, indent,
               "const size_t _remaining_" + std::to_string(id_) +
                   " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, capacity_bytes);");
      const auto remVar = "_remaining_" + std::to_string(id_);
      ++id_;
      emitLine(out, indent, "if (" + sizeVar + " > " + remVar + ") {");
      emitLine(out, indent + 1,
               "return -(int8_t)DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_DELIMITER_HEADER;");
      emitLine(out, indent, "}");
      const auto consumed = nextName("consumed");
      emitLine(out, indent, "size_t " + consumed + " = " + sizeVar + ";");
      emitLine(out, indent,
               "const int8_t " + errVar + " = " + nestedType +
                   "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                   consumed + ");");
      emitLine(out, indent, "if (" + errVar + " < 0) {");
      emitLine(out, indent + 1, "return " + errVar + ";");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
      return;
    }

    emitLine(out, indent,
             "size_t " + sizeVar +
                 " = capacity_bytes - dsdl_runtime_choose_min(offset_bits / 8U, capacity_bytes);");
    emitLine(out, indent,
             "const int8_t " + errVar + " = " + nestedType +
                 "__deserialize_(&" + expr + ", &buffer[offset_bits / 8U], &" +
                 sizeVar + ");");
    emitLine(out, indent, "if (" + errVar + " < 0) {");
    emitLine(out, indent + 1, "return " + errVar + ";");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8U;");
  }
};

std::string cTypeFromFieldType(const SemanticFieldType &type,
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
    return "uint8_t";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      return ctx.cTypeName(*type.compositeType);
    }
    return "uint8_t";
  }
  return "uint8_t";
}

void collectSectionDependencies(const SemanticSection &section,
                                std::set<std::string> &out) {
  for (const auto &field : section.fields) {
    if (field.resolvedType.compositeType) {
      const auto &ref = *field.resolvedType.compositeType;
      out.insert(typeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
    }
  }
}

void emitArrayMacros(std::ostringstream &out, const std::string &typeName,
                     const SemanticSection &section) {
  for (const auto &field : section.fields) {
    if (field.isPadding || field.resolvedType.arrayKind == ArrayKind::None) {
      continue;
    }
    const auto fieldName = sanitizeMacroToken(field.name);
    emitLine(out, 0,
             "#define " + typeName + "_" + fieldName + "_ARRAY_CAPACITY_ " +
                 std::to_string(field.resolvedType.arrayCapacity) + "U");
    emitLine(out, 0,
             "#define " + typeName + "_" + fieldName +
                 "_ARRAY_IS_VARIABLE_LENGTH_ " +
                 (isVariableArray(field.resolvedType.arrayKind) ? "true" :
                                                             "false"));
  }
  if (!section.fields.empty()) {
    out << "\n";
  }
}

void emitSectionTypedef(std::ostringstream &out, const std::string &typeName,
                        const SemanticSection &section,
                        const EmitterContext &ctx) {
  emitLine(out, 0, "typedef struct " + typeName + " {");

  std::size_t emitted = 0;
  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }

    const auto cMember = sanitizeIdentifier(field.name);
    const auto baseType = cTypeFromFieldType(field.resolvedType, ctx);

    if (field.resolvedType.arrayKind == ArrayKind::None) {
      emitLine(out, 1, baseType + " " + cMember + ";");
      ++emitted;
      continue;
    }

    if (field.resolvedType.arrayKind == ArrayKind::Fixed) {
      if (field.resolvedType.scalarCategory == SemanticScalarCategory::Bool) {
        emitLine(out, 1,
                 "uint8_t " + cMember + "[(" +
                     std::to_string(field.resolvedType.arrayCapacity) +
                     "U + 7U) / 8U];");
      } else {
        emitLine(out, 1,
                 baseType + " " + cMember + "[" +
                     std::to_string(field.resolvedType.arrayCapacity) + "U];");
      }
      ++emitted;
      continue;
    }

    emitLine(out, 1, "struct {");
    if (field.resolvedType.scalarCategory == SemanticScalarCategory::Bool) {
      emitLine(out, 2,
               "uint8_t bitpacked[(" +
                   std::to_string(field.resolvedType.arrayCapacity) +
                   "U + 7U) / 8U];");
    } else {
      emitLine(out, 2,
               baseType + " elements[" +
                   std::to_string(field.resolvedType.arrayCapacity) + "U];");
    }
    emitLine(out, 2, "size_t count;");
    emitLine(out, 1, "} " + cMember + ";");
    ++emitted;
  }

  if (section.isUnion) {
    emitLine(out, 1, "uint8_t _tag_;");
    ++emitted;
  }

  if (emitted == 0) {
    emitLine(out, 1, "uint8_t _dummy_;" );
  }

  emitLine(out, 0, "} " + typeName + ";");
  out << "\n";

  if (section.isUnion) {
    std::size_t optionCount = 0;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        ++optionCount;
      }
    }
    emitLine(out, 0,
             "#define " + typeName + "_UNION_OPTION_COUNT_ " +
                 std::to_string(optionCount) + "U");
    out << "\n";
  }
}

void emitSectionConstants(std::ostringstream &out, const std::string &typeName,
                          const SemanticSection &section) {
  for (const auto &c : section.constants) {
    emitLine(out, 0,
             "#define " + typeName + "_" + sanitizeMacroToken(c.name) + " (" +
                 valueToCExpr(c.value) + ")");
  }
  if (!section.constants.empty()) {
    out << "\n";
  }
}

void emitSectionMetadata(std::ostringstream &out, const std::string &typeName,
                         const std::string &fullName,
                         std::uint32_t majorVersion,
                         std::uint32_t minorVersion,
                         const SemanticSection &section) {
  emitLine(out, 0,
           "#define " + typeName + "_FULL_NAME_ \"" + fullName + "\"");
  emitLine(out, 0,
           "#define " + typeName + "_FULL_NAME_AND_VERSION_ \"" + fullName +
               "." + std::to_string(majorVersion) + "." +
               std::to_string(minorVersion) + "\"");
  emitLine(out, 0,
           "#define " + typeName + "_EXTENT_BYTES_ " +
               std::to_string(section.extentBits.value_or(0) / 8) + "UL");
  emitLine(out, 0,
           "#define " + typeName + "_SERIALIZATION_BUFFER_SIZE_BYTES_ " +
               std::to_string((section.serializationBufferSizeBits + 7) / 8) +
               "UL");
  out << "\n";
}

void emitSection(std::ostringstream &out, const EmitterContext &ctx,
                 const SemanticDefinition &def, const std::string &typeName,
                 const std::string &fullName, const SemanticSection &section) {
  emitSectionMetadata(out, typeName, fullName, def.info.majorVersion,
                      def.info.minorVersion, section);
  emitSectionConstants(out, typeName, section);
  emitArrayMacros(out, typeName, section);
  emitSectionTypedef(out, typeName, section, ctx);

  FunctionBodyEmitter bodyEmitter(ctx);
  bodyEmitter.emitSerializeFunction(out, typeName, section);
  bodyEmitter.emitDeserializeFunction(out, typeName, section);
}

llvm::Expected<std::string> loadRuntimeHeader() {
  const std::filesystem::path absoluteRuntimeHeader =
      std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "dsdl_runtime.h";
  std::ifstream in(absoluteRuntimeHeader.string());
  if (!in) {
    // Fallback for environments where compile-time source definitions are
    // unavailable or altered.
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

std::string renderHeader(const SemanticDefinition &def, const EmitterContext &ctx) {
  std::ostringstream out;
  const auto guard = headerGuard(def.info);
  const auto baseTypeName = ctx.cTypeName(def);

  std::set<std::string> dependencies;
  collectSectionDependencies(def.request, dependencies);
  if (def.response) {
    collectSectionDependencies(*def.response, dependencies);
  }

  out << "#ifndef " << guard << "\n";
  out << "#define " << guard << "\n\n";
  out << "#include <stddef.h>\n";
  out << "#include <stdint.h>\n";
  out << "#include <stdbool.h>\n";
  out << "#include \"dsdl_runtime.h\"\n";

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

  if (def.isService) {
    const auto requestType = baseTypeName + "__Request";
    const auto responseType = baseTypeName + "__Response";

    emitLine(out, 0,
             "#define " + baseTypeName + "_FULL_NAME_ \"" + def.info.fullName +
                 "\"");
    emitLine(out, 0,
             "#define " + baseTypeName + "_FULL_NAME_AND_VERSION_ \"" +
                 def.info.fullName + "." +
                 std::to_string(def.info.majorVersion) + "." +
                 std::to_string(def.info.minorVersion) + "\"");
    out << "\n";

    emitSection(out, ctx, def, requestType, def.info.fullName + ".Request",
                def.request);
    if (def.response) {
      emitSection(out, ctx, def, responseType, def.info.fullName + ".Response",
                  *def.response);
    }

    emitLine(out, 0, "typedef " + requestType + " " + baseTypeName + ";");
    emitLine(out, 0,
             "#define " + baseTypeName + "_EXTENT_BYTES_ " + requestType +
                 "_EXTENT_BYTES_");
    emitLine(out, 0,
             "#define " + baseTypeName + "_SERIALIZATION_BUFFER_SIZE_BYTES_ " +
                 requestType + "_SERIALIZATION_BUFFER_SIZE_BYTES_");
    out << "\n";

    emitLine(out, 0,
             "static inline int8_t " + baseTypeName +
                 "__serialize_(const " + baseTypeName +
                 "* const obj, uint8_t* const buffer, size_t* const "
                 "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "return " + requestType +
                 "__serialize_((const " + requestType + "*)obj, buffer, "
                 "inout_buffer_size_bytes);");
    emitLine(out, 0, "}");
    out << "\n";

    emitLine(out, 0,
             "static inline int8_t " + baseTypeName +
                 "__deserialize_(" + baseTypeName +
                 "* const out_obj, const uint8_t* buffer, size_t* const "
                 "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1,
             "return " + requestType + "__deserialize_((" + requestType +
                 "*)out_obj, buffer, inout_buffer_size_bytes);");
    emitLine(out, 0, "}");
  } else {
    emitSection(out, ctx, def, baseTypeName, def.info.fullName, def.request);
  }

  out << "#endif /* " << guard << " */\n";
  return out.str();
}

} // namespace

llvm::Error emitC(const SemanticModule &semantic, mlir::ModuleOp module,
                  const CEmitOptions &options,
                  DiagnosticEngine &diagnostics) {
  if (options.outDir.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  }

  std::error_code ec;
  llvm::sys::fs::create_directories(options.outDir, true);

  if (options.emitImplTranslationUnit) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(createLowerDSDLSerializationPass());
    pm.addPass(createConvertDSDLToEmitCPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSCFToEmitC());
    pm.addPass(mlir::createConvertArithToEmitC());
    pm.addPass(mlir::createConvertFuncToEmitC());

    if (mlir::failed(pm.run(module))) {
      diagnostics.error({"<mlir>", 1, 1}, "EmitC lowering pipeline failed");
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "EmitC lowering pipeline failed");
    }

    std::string emitted;
    llvm::raw_string_ostream emittedStream(emitted);
    if (mlir::failed(mlir::emitc::translateToCpp(module, emittedStream,
                                                 options.declareVariablesAtTop))) {
      diagnostics.error({"<mlir>", 1, 1}, "EmitC translation failed");
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "EmitC translation failed");
    }

    std::filesystem::path outRoot(options.outDir);
    if (auto err = writeFile(outRoot / "generated_impl.c", emitted)) {
      return err;
    }
  }

  std::filesystem::path outRoot(options.outDir);
  if (options.emitHeaderOnly) {
    auto runtimeHeader = loadRuntimeHeader();
    if (!runtimeHeader) {
      return runtimeHeader.takeError();
    }
    if (auto err = writeFile(outRoot / "dsdl_runtime.h", *runtimeHeader)) {
      return err;
    }

    EmitterContext ctx(semantic);
    for (const auto &def : semantic.definitions) {
      std::filesystem::path dir = outRoot;
      for (const auto &ns : def.info.namespaceComponents) {
        dir /= ns;
      }
      std::filesystem::create_directories(dir);

      if (auto err =
              writeFile(dir / headerFileName(def.info), renderHeader(def, ctx))) {
        return err;
      }
    }
  }

  return llvm::Error::success();
}

} // namespace llvmdsdl
