#include "llvmdsdl/CodeGen/TsEmitter.h"

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace llvmdsdl {
namespace {

bool isTsKeyword(const std::string &name) {
  static const std::set<std::string> kKeywords = {
      "break", "case",     "catch",   "class",    "const", "continue",
      "debugger", "default", "delete", "do",       "else",  "enum",
      "export", "extends",  "false",   "finally",  "for",   "function",
      "if",     "import",   "in",      "instanceof", "new",   "null",
      "return", "super",    "switch",  "this",     "throw", "true",
      "try",    "typeof",   "var",     "void",     "while", "with",
      "as",     "implements", "interface", "let",   "package", "private",
      "protected", "public", "static",  "yield",    "any",   "boolean",
      "number", "string",   "symbol",  "type",     "from",  "of"};
  return kKeywords.contains(name);
}

std::string sanitizeTsIdent(std::string name) {
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
  if (isTsKeyword(name)) {
    name += "_";
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
      out.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
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
  return sanitizeTsIdent(out);
}

std::string toPascalCase(const std::string &in) {
  std::string out;
  out.reserve(in.size() + 8);

  bool upperNext = true;
  for (char c : in) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      upperNext = true;
      continue;
    }
    if (c == '_') {
      upperNext = true;
      continue;
    }
    if (upperNext) {
      out.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
      upperNext = false;
    } else {
      out.push_back(c);
    }
  }

  if (out.empty()) {
    out = "X";
  }
  return sanitizeTsIdent(out);
}

std::string toUpperSnake(const std::string &in) {
  auto out = toSnakeCase(in);
  for (char &c : out) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string tsConstValue(const Value &value) {
  if (const auto *b = std::get_if<bool>(&value.data)) {
    return *b ? "true" : "false";
  }
  if (const auto *r = std::get_if<Rational>(&value.data)) {
    if (r->isInteger()) {
      return std::to_string(r->asInteger().value());
    }
    std::ostringstream out;
    out << "(" << r->numerator() << " / " << r->denominator() << ")";
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
  out << std::string(static_cast<std::size_t>(indent) * 2U, ' ') << line << '\n';
}

class EmitterContext final {
public:
  explicit EmitterContext(const SemanticModule &semantic) {
    for (const auto &def : semantic.definitions) {
      byKey_.emplace(
          loweredTypeKey(def.info.fullName, def.info.majorVersion,
                         def.info.minorVersion),
          &def);
    }
  }

  const SemanticDefinition *find(const SemanticTypeRef &ref) const {
    const auto it =
        byKey_.find(loweredTypeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
    if (it == byKey_.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::string namespacePath(const DiscoveredDefinition &info) const {
    std::string out;
    for (const auto &component : info.namespaceComponents) {
      if (!out.empty()) {
        out += "/";
      }
      out += toSnakeCase(component);
    }
    return out;
  }

  std::string typeName(const DiscoveredDefinition &info) const {
    return toPascalCase(info.shortName) + "_" +
           std::to_string(info.majorVersion) + "_" +
           std::to_string(info.minorVersion);
  }

  std::string typeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return typeName(def->info);
    }

    DiscoveredDefinition tmp;
    tmp.shortName = ref.shortName;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    return typeName(tmp);
  }

  std::string fileStem(const DiscoveredDefinition &info) const {
    return toSnakeCase(info.shortName) + "_" +
           std::to_string(info.majorVersion) + "_" +
           std::to_string(info.minorVersion);
  }

  std::filesystem::path relativeFilePath(const DiscoveredDefinition &info) const {
    std::filesystem::path rel;
    const auto nsPath = namespacePath(info);
    if (!nsPath.empty()) {
      rel /= nsPath;
    }
    rel /= fileStem(info) + ".ts";
    return rel;
  }

  std::filesystem::path relativeFilePath(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return relativeFilePath(def->info);
    }

    std::filesystem::path rel;
    for (const auto &component : ref.namespaceComponents) {
      rel /= toSnakeCase(component);
    }
    rel /= toSnakeCase(ref.shortName) + "_" + std::to_string(ref.majorVersion) +
           "_" + std::to_string(ref.minorVersion) + ".ts";
    return rel;
  }

private:
  std::unordered_map<std::string, const SemanticDefinition *> byKey_;
};

std::string tsFieldBaseType(const SemanticFieldType &type,
                            const EmitterContext &ctx) {
  switch (type.scalarCategory) {
  case SemanticScalarCategory::Bool:
    return "boolean";
  case SemanticScalarCategory::Utf8:
    return "string";
  case SemanticScalarCategory::Byte:
  case SemanticScalarCategory::UnsignedInt:
  case SemanticScalarCategory::SignedInt:
  case SemanticScalarCategory::Float:
  case SemanticScalarCategory::Void:
    return "number";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      return ctx.typeName(*type.compositeType);
    }
    return "unknown";
  }
  return "unknown";
}

std::string tsFieldType(const SemanticFieldType &type, const EmitterContext &ctx) {
  const auto base = tsFieldBaseType(type, ctx);
  if (type.arrayKind == ArrayKind::None) {
    return base;
  }
  return "Array<" + base + ">";
}

std::string relativeImportPath(const std::filesystem::path &fromFile,
                               const std::filesystem::path &toFile) {
  const auto fromDir = fromFile.parent_path();
  auto rel = toFile.lexically_relative(fromDir);
  std::string importPath = rel.generic_string();
  if (importPath.size() >= 3 && importPath.ends_with(".ts")) {
    importPath.resize(importPath.size() - 3);
  }
  if (!importPath.empty() && importPath.front() != '.') {
    importPath = "./" + importPath;
  }
  return importPath;
}

std::string moduleAliasFromPath(const std::string &modulePath) {
  std::string alias;
  alias.reserve(modulePath.size() + 8);
  for (char c : modulePath) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      alias.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    } else {
      alias.push_back('_');
    }
  }
  return sanitizeTsIdent(alias.empty() ? "module" : alias);
}

struct ImportSpec final {
  std::string modulePath;
  std::string typeName;
};

enum class TsRuntimeFieldKind {
  Bool,
  Unsigned,
  Signed,
};

struct TsRuntimeFieldPlan final {
  std::string fieldName;
  TsRuntimeFieldKind kind{TsRuntimeFieldKind::Unsigned};
  CastMode castMode{CastMode::Saturated};
  std::int64_t bitLength{0};
  std::int64_t bitOffset{0};
  bool isArray{false};
  std::int64_t arrayCapacity{0};
};

struct TsRuntimeSectionPlan final {
  std::vector<TsRuntimeFieldPlan> fields;
  std::int64_t totalBits{0};
};

std::optional<TsRuntimeSectionPlan>
buildTsRuntimeSectionPlan(const SemanticSection &section) {
  if (section.isUnion || !section.fixedSize) {
    return std::nullopt;
  }

  TsRuntimeSectionPlan plan;
  std::int64_t bitOffset = 0;
  for (const auto &field : section.fields) {
    const auto fieldBits = static_cast<std::int64_t>(field.resolvedType.bitLength);
    if (fieldBits < 0) {
      return std::nullopt;
    }

    if (field.isPadding) {
      bitOffset += fieldBits;
      continue;
    }

    bool isArray = false;
    std::int64_t arrayCapacity = 1;
    if (field.resolvedType.arrayKind == ArrayKind::None) {
      // Scalar field (non-array).
    } else if (field.resolvedType.arrayKind == ArrayKind::Fixed) {
      if (field.resolvedType.arrayCapacity <= 0) {
        return std::nullopt;
      }
      isArray = true;
      arrayCapacity = field.resolvedType.arrayCapacity;
    } else {
      // Variable arrays are not yet covered by this runtime path.
      return std::nullopt;
    }

    TsRuntimeFieldKind kind = TsRuntimeFieldKind::Unsigned;
    switch (field.resolvedType.scalarCategory) {
    case SemanticScalarCategory::Bool:
      if (fieldBits != 1) {
        return std::nullopt;
      }
      kind = TsRuntimeFieldKind::Bool;
      break;
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::UnsignedInt:
      kind = TsRuntimeFieldKind::Unsigned;
      break;
    case SemanticScalarCategory::SignedInt:
      kind = TsRuntimeFieldKind::Signed;
      break;
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::Float:
    case SemanticScalarCategory::Void:
    case SemanticScalarCategory::Composite:
      return std::nullopt;
    }

    if (fieldBits <= 0 || fieldBits > 32) {
      return std::nullopt;
    }

    TsRuntimeFieldPlan fieldPlan;
    fieldPlan.fieldName = sanitizeTsIdent(toSnakeCase(field.name));
    fieldPlan.kind = kind;
    fieldPlan.castMode = field.resolvedType.castMode;
    fieldPlan.bitLength = fieldBits;
    fieldPlan.bitOffset = bitOffset;
    fieldPlan.isArray = isArray;
    fieldPlan.arrayCapacity = arrayCapacity;
    plan.fields.push_back(fieldPlan);

    bitOffset += fieldBits * arrayCapacity;
  }

  plan.totalBits = bitOffset;
  return plan;
}

std::vector<ImportSpec>
collectCompositeImports(const SemanticSection &section,
                        const DiscoveredDefinition &owner,
                        const EmitterContext &ctx) {
  std::map<std::string, std::set<std::string>> importsByModule;
  const auto ownerPath = ctx.relativeFilePath(owner);

  for (const auto &field : section.fields) {
    if (field.isPadding || !field.resolvedType.compositeType) {
      continue;
    }
    const auto &ref = *field.resolvedType.compositeType;
    const auto targetPath = ctx.relativeFilePath(ref);
    if (targetPath == ownerPath) {
      continue;
    }
    const auto modulePath = relativeImportPath(ownerPath, targetPath);
    importsByModule[modulePath].insert(ctx.typeName(ref));
  }

  std::vector<ImportSpec> out;
  for (const auto &[modulePath, names] : importsByModule) {
    for (const auto &name : names) {
      out.push_back({modulePath, name});
    }
  }
  return out;
}

void emitSectionConstants(std::ostringstream &out, const std::string &prefix,
                          const SemanticSection &section) {
  for (const auto &constant : section.constants) {
    const auto constName = toUpperSnake(prefix) + "_" + toUpperSnake(constant.name);
    emitLine(out, 0,
             "export const " + constName + " = " + tsConstValue(constant.value) + ";");
  }
}

void emitStructSectionType(std::ostringstream &out, const std::string &typeName,
                           const SemanticSection &section,
                           const EmitterContext &ctx) {
  emitLine(out, 0, "export interface " + typeName + " {");
  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }
    const auto fieldName = sanitizeTsIdent(toSnakeCase(field.name));
    emitLine(out, 1,
             fieldName + ": " + tsFieldType(field.resolvedType, ctx) + ";");
  }
  emitLine(out, 0, "}");
}

void emitUnionSectionType(std::ostringstream &out, const std::string &typeName,
                          const SemanticSection &section,
                          const EmitterContext &ctx) {
  std::vector<const SemanticField *> options;
  for (const auto &field : section.fields) {
    if (!field.isPadding) {
      options.push_back(&field);
    }
  }

  if (options.empty()) {
    emitLine(out, 0, "export type " + typeName + " = { _tag: number };");
    return;
  }

  emitLine(out, 0, "export type " + typeName + " =");
  for (std::size_t i = 0; i < options.size(); ++i) {
    const auto *field = options[i];
    const auto fieldName = sanitizeTsIdent(toSnakeCase(field->name));
    std::ostringstream variant;
    variant << "{ _tag: " << field->unionOptionIndex << "; " << fieldName << ": "
            << tsFieldType(field->resolvedType, ctx) << "; }";
    const auto prefix = i == 0 ? "  | " : "  | ";
    emitLine(out, 0,
             prefix + variant.str() + (i + 1 == options.size() ? ";" : ""));
  }
}

void emitSectionType(std::ostringstream &out, const std::string &typeName,
                     const SemanticSection &section, const EmitterContext &ctx) {
  if (section.isUnion) {
    emitUnionSectionType(out, typeName, section, ctx);
  } else {
    emitStructSectionType(out, typeName, section, ctx);
  }
}

std::string tsRuntimeSerializeFn(const std::string &typeName) {
  return "serialize" + typeName;
}

std::string tsRuntimeDeserializeFn(const std::string &typeName) {
  return "deserialize" + typeName;
}

void emitTsRuntimeUnsupportedStubs(std::ostringstream &out,
                                   const std::string &typeName) {
  const auto serializeFn = tsRuntimeSerializeFn(typeName);
  const auto deserializeFn = tsRuntimeDeserializeFn(typeName);
  emitLine(out, 0, "export function " + serializeFn + "(_value: " + typeName +
                       "): Uint8Array {");
  emitLine(out, 1,
           "throw new Error(\"TypeScript runtime path is not yet available for this DSDL type\");");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function " + deserializeFn +
                       "(_bytes: Uint8Array): { value: " + typeName +
                       "; consumed: number } {");
  emitLine(out, 1,
           "throw new Error(\"TypeScript runtime path is not yet available for this DSDL type\");");
  emitLine(out, 0, "}");
}

void emitTsRuntimeFunctions(std::ostringstream &out, const std::string &typeName,
                            const TsRuntimeSectionPlan &plan) {
  const auto serializeFn = tsRuntimeSerializeFn(typeName);
  const auto deserializeFn = tsRuntimeDeserializeFn(typeName);
  const auto byteLength = (plan.totalBits + 7) / 8;

  emitLine(out, 0, "export function " + serializeFn + "(value: " + typeName +
                       "): Uint8Array {");
  emitLine(out, 1, "const out = new Uint8Array(" + std::to_string(byteLength) + ");");
  for (const auto &field : plan.fields) {
    const auto off = std::to_string(field.bitOffset);
    const auto bits = std::to_string(field.bitLength);
    const auto saturating = field.castMode == CastMode::Saturated ? "true" : "false";

    if (!field.isArray) {
      if (field.kind == TsRuntimeFieldKind::Bool) {
        emitLine(out, 1, "dsdlRuntime.setBit(out, " + off + ", value." +
                             field.fieldName + ");");
      } else if (field.kind == TsRuntimeFieldKind::Unsigned) {
        emitLine(out, 1, "dsdlRuntime.writeUnsigned(out, " + off + ", " + bits +
                             ", value." + field.fieldName + ", " + saturating + ");");
      } else {
        emitLine(out, 1, "dsdlRuntime.writeSigned(out, " + off + ", " + bits +
                             ", value." + field.fieldName + ", " + saturating + ");");
      }
    } else {
      const auto cap = std::to_string(field.arrayCapacity);
      const auto fieldArr = field.fieldName + "Array";
      emitLine(out, 1, "const " + fieldArr + " = value." + field.fieldName + ";");
      emitLine(out, 1, "if (!Array.isArray(" + fieldArr + ") || " + fieldArr +
                           ".length !== " + cap + ") {");
      emitLine(out, 2, "throw new Error(\"field '" + field.fieldName +
                           "' expects exactly " + cap + " elements\");");
      emitLine(out, 1, "}");
      emitLine(out, 1,
               "for (let i = 0; i < " + cap + "; ++i) {");
      const auto elemOff = off + " + (i * " + bits + ")";
      if (field.kind == TsRuntimeFieldKind::Bool) {
        emitLine(out, 2, "dsdlRuntime.setBit(out, " + elemOff + ", " + fieldArr +
                             "[i]);");
      } else if (field.kind == TsRuntimeFieldKind::Unsigned) {
        emitLine(out, 2, "dsdlRuntime.writeUnsigned(out, " + elemOff + ", " + bits +
                             ", " + fieldArr + "[i], " + saturating + ");");
      } else {
        emitLine(out, 2, "dsdlRuntime.writeSigned(out, " + elemOff + ", " + bits +
                             ", " + fieldArr + "[i], " + saturating + ");");
      }
      emitLine(out, 1, "}");
    }
  }
  emitLine(out, 1, "return out;");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");

  emitLine(out, 0, "export function " + deserializeFn +
                       "(bytes: Uint8Array): { value: " + typeName +
                       "; consumed: number } {");
  emitLine(out, 1, "const value = {} as " + typeName + ";");
  for (const auto &field : plan.fields) {
    const auto off = std::to_string(field.bitOffset);
    const auto bits = std::to_string(field.bitLength);
    if (!field.isArray) {
      if (field.kind == TsRuntimeFieldKind::Bool) {
        emitLine(out, 1, "value." + field.fieldName + " = dsdlRuntime.getBit(bytes, " +
                             off + ");");
      } else if (field.kind == TsRuntimeFieldKind::Unsigned) {
        emitLine(out, 1, "value." + field.fieldName +
                             " = dsdlRuntime.readUnsigned(bytes, " + off + ", " +
                             bits + ");");
      } else {
        emitLine(out, 1, "value." + field.fieldName +
                             " = dsdlRuntime.readSigned(bytes, " + off + ", " +
                             bits + ");");
      }
      continue;
    }

    const auto cap = std::to_string(field.arrayCapacity);
    const auto elemOff = off + " + (i * " + bits + ")";
    const auto fieldArr = field.fieldName + "Array";
    const auto arrayElemType =
        field.kind == TsRuntimeFieldKind::Bool ? "boolean" : "number";
    emitLine(out, 1, "const " + fieldArr + ": Array<" + arrayElemType +
                         "> = new Array(" + cap + ");");
    emitLine(out, 1, "for (let i = 0; i < " + cap + "; ++i) {");
    if (field.kind == TsRuntimeFieldKind::Bool) {
      emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.getBit(bytes, " + elemOff + ");");
    } else if (field.kind == TsRuntimeFieldKind::Unsigned) {
      emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.readUnsigned(bytes, " + elemOff +
                           ", " + bits + ");");
    } else {
      emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.readSigned(bytes, " + elemOff +
                           ", " + bits + ");");
    }
    emitLine(out, 1, "}");
    emitLine(out, 1, "value." + field.fieldName + " = " + fieldArr + ";");
  }
  emitLine(out, 1,
           "const consumed = Math.min(bytes.length, dsdlRuntime.byteLengthForBits(" +
               std::to_string(plan.totalBits) + "));");
  emitLine(out, 1, "return { value, consumed };");
  emitLine(out, 0, "}");
}

std::string renderDefinitionFile(const SemanticDefinition &def,
                                 const EmitterContext &ctx) {
  std::ostringstream out;
  emitLine(out, 0, "// Generated by llvmdsdl (TypeScript experimental backend).");

  const auto requestRuntimePlan = buildTsRuntimeSectionPlan(def.request);
  std::optional<TsRuntimeSectionPlan> responseRuntimePlan;
  if (def.response) {
    responseRuntimePlan = buildTsRuntimeSectionPlan(*def.response);
  }
  const bool hasSupportedRuntimePlan =
      requestRuntimePlan.has_value() || responseRuntimePlan.has_value();

  std::map<std::string, std::set<std::string>> importsByModule;
  for (const auto &imp : collectCompositeImports(def.request, def.info, ctx)) {
    importsByModule[imp.modulePath].insert(imp.typeName);
  }
  if (def.response) {
    for (const auto &imp : collectCompositeImports(*def.response, def.info, ctx)) {
      importsByModule[imp.modulePath].insert(imp.typeName);
    }
  }

  if (hasSupportedRuntimePlan) {
    const auto ownerPath = ctx.relativeFilePath(def.info);
    const auto runtimePath =
        relativeImportPath(ownerPath, std::filesystem::path("dsdl_runtime.ts"));
    emitLine(out, 0, "import * as dsdlRuntime from \"" + runtimePath + "\";");
  }

  for (const auto &[modulePath, names] : importsByModule) {
    std::string importNames;
    for (const auto &name : names) {
      if (!importNames.empty()) {
        importNames += ", ";
      }
      importNames += name;
    }
    emitLine(out, 0,
             "import type { " + importNames + " } from \"" + modulePath + "\";");
  }
  if (!importsByModule.empty()) {
    emitLine(out, 0, "");
  }

  const auto baseType = ctx.typeName(def.info);
  emitLine(out, 0, "export const DSDL_FULL_NAME = \"" + def.info.fullName + "\";");
  emitLine(out, 0,
           "export const DSDL_VERSION_MAJOR = " +
               std::to_string(def.info.majorVersion) + ";");
  emitLine(out, 0,
           "export const DSDL_VERSION_MINOR = " +
               std::to_string(def.info.minorVersion) + ";");
  emitLine(out, 0, "");

  if (!def.isService) {
    emitSectionType(out, baseType, def.request, ctx);
    emitLine(out, 0, "");
    emitSectionConstants(out, baseType, def.request);
    emitLine(out, 0, "");
    if (requestRuntimePlan) {
      emitTsRuntimeFunctions(out, baseType, *requestRuntimePlan);
    } else {
      emitTsRuntimeUnsupportedStubs(out, baseType);
    }
    return out.str();
  }

  const auto reqType = baseType + "_Request";
  const auto respType = baseType + "_Response";

  emitSectionType(out, reqType, def.request, ctx);
  emitLine(out, 0, "");
  emitSectionConstants(out, reqType, def.request);
  emitLine(out, 0, "");
  if (requestRuntimePlan) {
    emitTsRuntimeFunctions(out, reqType, *requestRuntimePlan);
  } else {
    emitTsRuntimeUnsupportedStubs(out, reqType);
  }
  emitLine(out, 0, "");

  if (def.response) {
    emitSectionType(out, respType, *def.response, ctx);
    emitLine(out, 0, "");
    emitSectionConstants(out, respType, *def.response);
    emitLine(out, 0, "");
    if (responseRuntimePlan) {
      emitTsRuntimeFunctions(out, respType, *responseRuntimePlan);
    } else {
      emitTsRuntimeUnsupportedStubs(out, respType);
    }
    emitLine(out, 0, "");
  }

  emitLine(out, 0, "export type " + baseType + " = " + reqType + ";");
  return out.str();
}

std::string renderPackageJson(const TsEmitOptions &options) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"name\": \"" << options.moduleName << "\",\n";
  out << "  \"version\": \"0.1.0\",\n";
  out << "  \"type\": \"module\"\n";
  out << "}\n";
  return out.str();
}

std::string renderTsRuntimeModule() {
  std::ostringstream out;
  emitLine(out, 0, "// Generated by llvmdsdl TypeScript runtime scaffold.");
  emitLine(out, 0, "");
  emitLine(out, 0, "function toBigIntValue(value: number | bigint): bigint {");
  emitLine(out, 1, "if (typeof value === \"bigint\") {");
  emitLine(out, 2, "return value;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "if (!Number.isFinite(value)) {");
  emitLine(out, 2, "return 0n;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return BigInt(Math.trunc(value));");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "function maskBits(lenBits: number): bigint {");
  emitLine(out, 1, "if (lenBits <= 0) {");
  emitLine(out, 2, "return 0n;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return (1n << BigInt(lenBits)) - 1n;");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function byteLengthForBits(totalBits: number): number {");
  emitLine(out, 1, "if (totalBits <= 0) {");
  emitLine(out, 2, "return 0;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return Math.floor((totalBits + 7) / 8);");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "function setRawBit(buf: Uint8Array, offBits: number, bit: boolean): void {");
  emitLine(out, 1, "const byteIndex = Math.floor(offBits / 8);");
  emitLine(out, 1, "const bitIndex = offBits % 8;");
  emitLine(out, 1, "if (byteIndex < 0 || byteIndex >= buf.length) {");
  emitLine(out, 2, "throw new Error(\"serialization buffer too small\");");
  emitLine(out, 1, "}");
  emitLine(out, 1, "const mask = 1 << bitIndex;");
  emitLine(out, 1, "if (bit) {");
  emitLine(out, 2, "buf[byteIndex] = (buf[byteIndex] | mask) & 0xff;");
  emitLine(out, 1, "} else {");
  emitLine(out, 2, "buf[byteIndex] = (buf[byteIndex] & (~mask)) & 0xff;");
  emitLine(out, 1, "}");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "function getRawBit(buf: Uint8Array, offBits: number): boolean {");
  emitLine(out, 1, "const byteIndex = Math.floor(offBits / 8);");
  emitLine(out, 1, "const bitIndex = offBits % 8;");
  emitLine(out, 1, "if (byteIndex < 0 || byteIndex >= buf.length) {");
  emitLine(out, 2, "return false;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return ((buf[byteIndex] >> bitIndex) & 1) === 1;");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "function writeUnsignedBits(");
  emitLine(out, 1, "buf: Uint8Array,");
  emitLine(out, 1, "offBits: number,");
  emitLine(out, 1, "lenBits: number,");
  emitLine(out, 1, "value: bigint");
  emitLine(out, 0, "): void {");
  emitLine(out, 1, "for (let i = 0; i < lenBits; ++i) {");
  emitLine(out, 2, "const bit = ((value >> BigInt(i)) & 1n) === 1n;");
  emitLine(out, 2, "setRawBit(buf, offBits + i, bit);");
  emitLine(out, 1, "}");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "function readUnsignedBits(buf: Uint8Array, offBits: number, lenBits: number): bigint {");
  emitLine(out, 1, "let out = 0n;");
  emitLine(out, 1, "for (let i = 0; i < lenBits; ++i) {");
  emitLine(out, 2, "if (getRawBit(buf, offBits + i)) {");
  emitLine(out, 3, "out |= (1n << BigInt(i));");
  emitLine(out, 2, "}");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return out;");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function setBit(buf: Uint8Array, offBits: number, value: boolean): void {");
  emitLine(out, 1, "setRawBit(buf, offBits, !!value);");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function getBit(buf: Uint8Array, offBits: number): boolean {");
  emitLine(out, 1, "return getRawBit(buf, offBits);");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function writeUnsigned(");
  emitLine(out, 1, "buf: Uint8Array,");
  emitLine(out, 1, "offBits: number,");
  emitLine(out, 1, "lenBits: number,");
  emitLine(out, 1, "value: number | bigint,");
  emitLine(out, 1, "saturating: boolean");
  emitLine(out, 0, "): void {");
  emitLine(out, 1, "if (lenBits <= 0) {");
  emitLine(out, 2, "return;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "const max = maskBits(lenBits);");
  emitLine(out, 1, "let inValue = toBigIntValue(value);");
  emitLine(out, 1, "if (saturating) {");
  emitLine(out, 2, "if (inValue < 0n) {");
  emitLine(out, 3, "inValue = 0n;");
  emitLine(out, 2, "} else if (inValue > max) {");
  emitLine(out, 3, "inValue = max;");
  emitLine(out, 2, "}");
  emitLine(out, 1, "} else {");
  emitLine(out, 2, "inValue = BigInt.asUintN(lenBits, inValue);");
  emitLine(out, 1, "}");
  emitLine(out, 1, "writeUnsignedBits(buf, offBits, lenBits, inValue);");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function writeSigned(");
  emitLine(out, 1, "buf: Uint8Array,");
  emitLine(out, 1, "offBits: number,");
  emitLine(out, 1, "lenBits: number,");
  emitLine(out, 1, "value: number | bigint,");
  emitLine(out, 1, "saturating: boolean");
  emitLine(out, 0, "): void {");
  emitLine(out, 1, "if (lenBits <= 0) {");
  emitLine(out, 2, "return;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "let inValue = toBigIntValue(value);");
  emitLine(out, 1, "if (saturating) {");
  emitLine(out, 2, "const min = -(1n << BigInt(lenBits - 1));");
  emitLine(out, 2, "const max = (1n << BigInt(lenBits - 1)) - 1n;");
  emitLine(out, 2, "if (inValue < min) {");
  emitLine(out, 3, "inValue = min;");
  emitLine(out, 2, "} else if (inValue > max) {");
  emitLine(out, 3, "inValue = max;");
  emitLine(out, 2, "}");
  emitLine(out, 1, "}");
  emitLine(out, 1, "writeUnsignedBits(buf, offBits, lenBits, BigInt.asUintN(lenBits, inValue));");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function readUnsigned(buf: Uint8Array, offBits: number, lenBits: number): number {");
  emitLine(out, 1, "if (lenBits <= 0) {");
  emitLine(out, 2, "return 0;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return Number(readUnsignedBits(buf, offBits, lenBits));");
  emitLine(out, 0, "}");
  emitLine(out, 0, "");
  emitLine(out, 0, "export function readSigned(buf: Uint8Array, offBits: number, lenBits: number): number {");
  emitLine(out, 1, "if (lenBits <= 0) {");
  emitLine(out, 2, "return 0;");
  emitLine(out, 1, "}");
  emitLine(out, 1, "const raw = readUnsignedBits(buf, offBits, lenBits);");
  emitLine(out, 1, "const signBit = 1n << BigInt(lenBits - 1);");
  emitLine(out, 1, "if ((raw & signBit) !== 0n) {");
  emitLine(out, 2, "const signed = raw - (1n << BigInt(lenBits));");
  emitLine(out, 2, "return Number(signed);");
  emitLine(out, 1, "}");
  emitLine(out, 1, "return Number(raw);");
  emitLine(out, 0, "}");
  return out.str();
}

} // namespace

llvm::Error emitTs(const SemanticModule &semantic, mlir::ModuleOp module,
                   const TsEmitOptions &options,
                   DiagnosticEngine &diagnostics) {
  if (options.outDir.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  }

  LoweredFactsMap loweredFacts;
  if (!collectLoweredFactsFromMlir(semantic, module, diagnostics, "TypeScript",
                                   &loweredFacts,
                                   options.optimizeLoweredSerDes)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "MLIR schema coverage validation failed for TypeScript emission");
  }

  std::filesystem::path outRoot(options.outDir);
  std::filesystem::create_directories(outRoot);

  if (options.emitPackageJson) {
    if (auto err = writeFile(outRoot / "package.json", renderPackageJson(options))) {
      return err;
    }
  }
  if (auto err = writeFile(outRoot / "dsdl_runtime.ts", renderTsRuntimeModule())) {
    return err;
  }

  EmitterContext ctx(semantic);

  std::vector<const SemanticDefinition *> ordered;
  ordered.reserve(semantic.definitions.size());
  for (const auto &def : semantic.definitions) {
    ordered.push_back(&def);
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto *lhs, const auto *rhs) {
    if (lhs->info.fullName != rhs->info.fullName) {
      return lhs->info.fullName < rhs->info.fullName;
    }
    if (lhs->info.majorVersion != rhs->info.majorVersion) {
      return lhs->info.majorVersion < rhs->info.majorVersion;
    }
    return lhs->info.minorVersion < rhs->info.minorVersion;
  });

  std::vector<std::filesystem::path> generatedRelativePaths;
  generatedRelativePaths.reserve(ordered.size());

  for (const auto *def : ordered) {
    const auto relPath = ctx.relativeFilePath(def->info);
    generatedRelativePaths.push_back(relPath);

    const auto fullPath = outRoot / relPath;
    std::filesystem::create_directories(fullPath.parent_path());

    if (auto err = writeFile(fullPath, renderDefinitionFile(*def, ctx))) {
      return err;
    }
  }

  std::ostringstream index;
  emitLine(index, 0, "// Generated by llvmdsdl (TypeScript experimental backend).");
  std::map<std::string, unsigned> aliasUseCount;
  for (const auto &relPath : generatedRelativePaths) {
    std::string modulePath = relPath.generic_string();
    if (modulePath.size() >= 3 && modulePath.ends_with(".ts")) {
      modulePath.resize(modulePath.size() - 3);
    }
    std::string alias = moduleAliasFromPath(modulePath);
    unsigned &useCount = aliasUseCount[alias];
    if (useCount > 0) {
      alias += "_" + std::to_string(useCount);
    }
    ++useCount;

    emitLine(index, 0,
             "export * as " + alias + " from \"./" + modulePath + "\";");
  }

  if (auto err = writeFile(outRoot / "index.ts", index.str())) {
    return err;
  }

  return llvm::Error::success();
}

} // namespace llvmdsdl
