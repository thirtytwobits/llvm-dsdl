#include "llvmdsdl/CodeGen/GoEmitter.h"

#include "llvmdsdl/CodeGen/ArrayWirePlan.h"
#include "llvmdsdl/CodeGen/HelperBindingRender.h"
#include "llvmdsdl/CodeGen/HelperSymbolResolver.h"
#include "llvmdsdl/CodeGen/LoweredRenderIR.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/TypeStorage.h"
#include "llvmdsdl/CodeGen/WireLayoutFacts.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Error.h"
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

bool isGoKeyword(const std::string &name) {
  static const std::set<std::string> kKeywords = {
      "break",   "default", "func",      "interface", "select", "case",
      "defer",   "go",      "map",       "struct",    "chan",   "else",
      "goto",    "package", "switch",    "const",     "fallthrough",
      "if",      "range",   "type",      "continue",  "for",    "import",
      "return",  "var"};
  return kKeywords.contains(name);
}

std::string sanitizeGoIdent(std::string name) {
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
  if (isGoKeyword(name)) {
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
  return sanitizeGoIdent(out);
}

std::string toUpperSnake(const std::string &in) {
  auto out = toSnakeCase(in);
  for (char &c : out) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string toExportedIdent(const std::string &in) {
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
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
      upperNext = false;
    } else {
      out.push_back(c);
    }
  }
  if (out.empty()) {
    out = "X";
  }
  out = sanitizeGoIdent(out);
  if (!std::isupper(static_cast<unsigned char>(out.front()))) {
    out.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(out.front())));
  }
  return out;
}

std::string packagePathFromComponents(const std::vector<std::string> &components) {
  std::string out;
  for (const auto &c : components) {
    if (!out.empty()) {
      out += "/";
    }
    out += toSnakeCase(c);
  }
  return out;
}

std::string packageNameFromPath(const std::string &path) {
  if (path.empty()) {
    return "rootdsdl";
  }
  const auto split = path.find_last_of('/');
  const auto leaf = split == std::string::npos ? path : path.substr(split + 1);
  auto out = sanitizeGoIdent(leaf);
  if (out.empty()) {
    out = "rootdsdl";
  }
  return out;
}

std::string unsignedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "uint8";
  case 16:
    return "uint16";
  case 32:
    return "uint32";
  default:
    return "uint64";
  }
}

std::string signedStorageType(const std::uint32_t bitLength) {
  switch (scalarStorageBits(bitLength)) {
  case 8:
    return "int8";
  case 16:
    return "int16";
  case 32:
    return "int32";
  default:
    return "int64";
  }
}

std::string goConstValue(const Value &value) {
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

  std::string packagePath(const DiscoveredDefinition &info) const {
    return packagePathFromComponents(info.namespaceComponents);
  }

  std::string packagePath(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return packagePath(def->info);
    }
    return packagePathFromComponents(ref.namespaceComponents);
  }

  std::string goTypeName(const DiscoveredDefinition &info) const {
    return toExportedIdent(info.shortName) + "_" +
           std::to_string(info.majorVersion) + "_" +
           std::to_string(info.minorVersion);
  }

  std::string goTypeName(const SemanticTypeRef &ref) const {
    if (const auto *def = find(ref)) {
      return goTypeName(def->info);
    }
    DiscoveredDefinition tmp;
    tmp.shortName = ref.shortName;
    tmp.majorVersion = ref.majorVersion;
    tmp.minorVersion = ref.minorVersion;
    return goTypeName(tmp);
  }

  std::string goFileName(const DiscoveredDefinition &info) const {
    return toSnakeCase(info.shortName) + "_" + std::to_string(info.majorVersion) +
           "_" + std::to_string(info.minorVersion) + ".go";
  }

private:
  std::unordered_map<std::string, const SemanticDefinition *> byKey_;
};

void collectSectionDependencies(const SemanticSection &section,
                                std::set<std::string> &out) {
  for (const auto &field : section.fields) {
    if (field.resolvedType.compositeType) {
      const auto &ref = *field.resolvedType.compositeType;
      out.insert(loweredTypeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
    }
  }
}

std::map<std::string, std::string>
computeImportAliases(const SemanticDefinition &def, const EmitterContext &ctx) {
  std::set<std::string> deps;
  collectSectionDependencies(def.request, deps);
  if (def.response) {
    collectSectionDependencies(*def.response, deps);
  }

  const std::string currentPath = ctx.packagePath(def.info);
  std::map<std::string, std::string> out;
  std::set<std::string> usedAliases;

  for (const auto &dep : deps) {
    const auto first = dep.find(':');
    const auto second = dep.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
      continue;
    }
    SemanticTypeRef ref;
    ref.fullName = dep.substr(0, first);
    ref.majorVersion = static_cast<std::uint32_t>(
        std::stoul(dep.substr(first + 1, second - first - 1)));
    ref.minorVersion = static_cast<std::uint32_t>(std::stoul(dep.substr(second + 1)));
    if (const auto *resolved = ctx.find(ref)) {
      ref.namespaceComponents = resolved->info.namespaceComponents;
      ref.shortName = resolved->info.shortName;
    }

    const auto depPath = ctx.packagePath(ref);
    if (depPath.empty() || depPath == currentPath) {
      continue;
    }
    auto alias = "pkg_" + sanitizeGoIdent(llvm::join(ref.namespaceComponents, "_"));
    if (alias == "pkg_") {
      alias = "pkg_dep";
    }
    std::size_t suffix = 1;
    const auto baseAlias = alias;
    while (usedAliases.contains(alias)) {
      alias = baseAlias + "_" + std::to_string(suffix++);
    }
    usedAliases.insert(alias);
    out.emplace(depPath, alias);
  }

  return out;
}

std::string goBaseFieldType(const SemanticFieldType &type, const EmitterContext &ctx,
                            const std::string &currentPackagePath,
                            const std::map<std::string, std::string> &importAliases) {
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
    return type.bitLength == 64 ? "float64" : "float32";
  case SemanticScalarCategory::Void:
    return "uint8";
  case SemanticScalarCategory::Composite:
    if (type.compositeType) {
      const auto depPath = ctx.packagePath(*type.compositeType);
      const auto depType = ctx.goTypeName(*type.compositeType);
      if (depPath.empty() || depPath == currentPackagePath) {
        return depType;
      }
      const auto it = importAliases.find(depPath);
      if (it != importAliases.end()) {
        return it->second + "." + depType;
      }
      return depType;
    }
    return "uint8";
  }
  return "uint8";
}

std::string goFieldType(const SemanticFieldType &type, const EmitterContext &ctx,
                        const std::string &currentPackagePath,
                        const std::map<std::string, std::string> &importAliases) {
  const auto base = goBaseFieldType(type, ctx, currentPackagePath, importAliases);
  if (type.arrayKind == ArrayKind::None) {
    return base;
  }
  if (type.arrayKind == ArrayKind::Fixed) {
    return "[" + std::to_string(type.arrayCapacity) + "]" + base;
  }
  return "[]" + base;
}

const LoweredSectionFacts *
findLoweredSectionFacts(const LoweredFactsMap &facts, const SemanticDefinition &def,
                        llvm::StringRef sectionName) {
  const auto key =
      loweredTypeKey(def.info.fullName, def.info.majorVersion, def.info.minorVersion);
  const auto defIt = facts.find(key);
  if (defIt == facts.end()) {
    return nullptr;
  }
  const auto sectionKey = sectionName.empty() ? std::string{} : sectionName.str();
  const auto secIt = defIt->second.find(sectionKey);
  if (secIt == defIt->second.end()) {
    return nullptr;
  }
  return &secIt->second;
}

class FunctionBodyEmitter final {
public:
  FunctionBodyEmitter(
      const EmitterContext &ctx, std::string currentPackagePath,
      const std::map<std::string, std::string> &importAliases)
      : ctx_(ctx), currentPackagePath_(std::move(currentPackagePath)),
        importAliases_(importAliases) {}

  void emitSerializeFunction(std::ostringstream &out, const std::string &typeName,
                             const SemanticSection &section,
                             const LoweredSectionFacts *sectionFacts) {
    emitLine(out, 0,
             "func (obj *" + typeName + ") Serialize(buffer []byte) (int8, int) {");
    emitLine(out, 1, "if obj == nil {");
    emitLine(out, 2, "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
    emitLine(out, 1, "}");
    emitLine(out, 1, "offsetBits := 0");
    const auto renderIR = buildLoweredBodyRenderIR(
        section, sectionFacts, HelperBindingDirection::Serialize);
    emitSerializeMlirHelperBindings(out, renderIR.helperBindings, 1);

    if (renderIR.helperBindings.capacityCheck) {
      const auto capacityHelper =
          helperBindingName(renderIR.helperBindings.capacityCheck->symbol);
      emitLine(out, 1,
               "if rc := " + capacityHelper + "(int64(len(buffer) * 8)); rc != "
               "dsdlruntime.DSDL_RUNTIME_SUCCESS {");
      emitLine(out, 2, "return rc, 0");
      emitLine(out, 1, "}");
    } else {
      emitLine(out, 1,
               "if len(buffer)*8 < " +
                   std::to_string(section.serializationBufferSizeBits) + " {");
      emitLine(
          out, 2,
          "return -dsdlruntime.DSDL_RUNTIME_ERROR_SERIALIZATION_BUFFER_TOO_SMALL, 0");
      emitLine(out, 1, "}");
    }

    for (const auto &step : renderIR.steps) {
      switch (step.kind) {
      case LoweredRenderStepKind::UnionDispatch:
        emitSerializeUnion(out, section, step.unionBranches, 1, sectionFacts,
                           renderIR.helperBindings);
        break;
      case LoweredRenderStepKind::Field: {
        const auto *const field = step.fieldStep.field;
        if (field == nullptr) {
          continue;
        }
        emitAlignSerialize(out, field->resolvedType.alignmentBits, 1);
        emitSerializeAny(out, field->resolvedType,
                         "obj." + toExportedIdent(field->name), 1,
                         step.fieldStep.arrayLengthPrefixBits,
                         step.fieldStep.fieldFacts);
        break;
      }
      case LoweredRenderStepKind::Padding: {
        const auto *const field = step.fieldStep.field;
        if (field == nullptr) {
          continue;
        }
        emitAlignSerialize(out, field->resolvedType.alignmentBits, 1);
        emitSerializePadding(out, field->resolvedType, 1);
        break;
      }
      }
    }

    emitAlignSerialize(out, 8, 1);
    emitLine(out, 1, "return dsdlruntime.DSDL_RUNTIME_SUCCESS, offsetBits / 8");
    emitLine(out, 0, "}");
  }

  void emitDeserializeFunction(std::ostringstream &out,
                               const std::string &typeName,
                               const SemanticSection &section,
                               const LoweredSectionFacts *sectionFacts) {
    emitLine(out, 0, "func (obj *" + typeName +
                         ") Deserialize(buffer []byte) (int8, int) {");
    emitLine(out, 1, "if obj == nil {");
    emitLine(out, 2, "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
    emitLine(out, 1, "}");
    emitLine(out, 1, "capacityBytes := len(buffer)");
    emitLine(out, 1, "capacityBits := capacityBytes * 8");
    emitLine(out, 1, "offsetBits := 0");
    const auto renderIR = buildLoweredBodyRenderIR(
        section, sectionFacts, HelperBindingDirection::Deserialize);
    emitDeserializeMlirHelperBindings(out, renderIR.helperBindings, 1);

    for (const auto &step : renderIR.steps) {
      switch (step.kind) {
      case LoweredRenderStepKind::UnionDispatch:
        emitDeserializeUnion(out, section, step.unionBranches, 1, sectionFacts,
                             renderIR.helperBindings);
        break;
      case LoweredRenderStepKind::Field: {
        const auto *const field = step.fieldStep.field;
        if (field == nullptr) {
          continue;
        }
        emitAlignDeserialize(out, field->resolvedType.alignmentBits, 1);
        emitDeserializeAny(out, field->resolvedType,
                           "obj." + toExportedIdent(field->name), 1,
                           step.fieldStep.arrayLengthPrefixBits,
                           step.fieldStep.fieldFacts);
        break;
      }
      case LoweredRenderStepKind::Padding: {
        const auto *const field = step.fieldStep.field;
        if (field == nullptr) {
          continue;
        }
        emitAlignDeserialize(out, field->resolvedType.alignmentBits, 1);
        emitDeserializePadding(out, field->resolvedType, 1);
        break;
      }
      }
    }

    emitAlignDeserialize(out, 8, 1);
    emitLine(out, 1,
             "consumedBits := dsdlruntime.ChooseMin(offsetBits, capacityBits)");
    emitLine(out, 1, "return dsdlruntime.DSDL_RUNTIME_SUCCESS, consumedBits / 8");
    emitLine(out, 0, "}");
  }

private:
  const EmitterContext &ctx_;
  std::string currentPackagePath_;
  const std::map<std::string, std::string> &importAliases_;
  std::size_t id_{0};

  std::string nextName(const std::string &prefix) {
    return "_" + prefix + std::to_string(id_++) + "_";
  }

  std::string helperBindingName(const std::string &helperSymbol) const {
    return "mlir_" + sanitizeGoIdent(helperSymbol);
  }

  void emitSerializeMlirHelperBindings(
      std::ostringstream &out, const SectionHelperBindingPlan &plan,
      const int indent) {
    for (const auto &line : renderSectionHelperBindings(
             plan, HelperBindingRenderLanguage::Go,
             ScalarBindingRenderDirection::Serialize,
             [this](const std::string &symbol) {
               return helperBindingName(symbol);
             },
             /*emitCapacityCheck=*/true)) {
      emitLine(out, indent, line);
    }
  }

  void emitDeserializeMlirHelperBindings(
      std::ostringstream &out, const SectionHelperBindingPlan &plan,
      const int indent) {
    for (const auto &line : renderSectionHelperBindings(
             plan, HelperBindingRenderLanguage::Go,
             ScalarBindingRenderDirection::Deserialize,
             [this](const std::string &symbol) {
               return helperBindingName(symbol);
             },
             /*emitCapacityCheck=*/false)) {
      emitLine(out, indent, line);
    }
  }

  void emitAlignSerialize(std::ostringstream &out, std::int64_t alignmentBits,
                          int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             "for (offsetBits % " + std::to_string(alignmentBits) + ") != 0 {");
    emitLine(out, indent + 1,
             err + " := dsdlruntime.SetBit(buffer, offsetBits, false)");
    emitLine(out, indent + 1, "if " + err + " < 0 {");
    emitLine(out, indent + 2, "return " + err + ", 0");
    emitLine(out, indent + 1, "}");
    emitLine(out, indent + 1, "offsetBits++");
    emitLine(out, indent, "}");
  }

  void emitAlignDeserialize(std::ostringstream &out, std::int64_t alignmentBits,
                            int indent) {
    if (alignmentBits <= 1) {
      return;
    }
    emitLine(out, indent,
             "offsetBits = (offsetBits + " + std::to_string(alignmentBits - 1) +
                 ") & ^" + std::to_string(alignmentBits - 1));
  }

  void emitSerializePadding(std::ostringstream &out, const SemanticFieldType &type,
                            int indent) {
    if (type.bitLength == 0) {
      return;
    }
    const auto err = nextName("err");
    emitLine(out, indent,
             err + " := dsdlruntime.SetUxx(buffer, offsetBits, 0, " +
                 std::to_string(type.bitLength) + ")");
    emitLine(out, indent, "if " + err + " < 0 {");
    emitLine(out, indent + 1, "return " + err + ", 0");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
  }

  void emitDeserializePadding(std::ostringstream &out, const SemanticFieldType &type,
                              int indent) {
    if (type.bitLength == 0) {
      return;
    }
    emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
  }

  void emitSerializeUnion(std::ostringstream &out, const SemanticSection &section,
                          const std::vector<PlannedFieldStep> &unionBranches,
                          int indent, const LoweredSectionFacts *sectionFacts,
                          const SectionHelperBindingPlan &helperBindings) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    if (helperBindings.unionTagValidate) {
      const auto validateHelper =
          helperBindingName(helperBindings.unionTagValidate->symbol);
      emitLine(out, indent,
               "if rc := " + validateHelper +
                   "(int64(obj.Tag)); rc != dsdlruntime.DSDL_RUNTIME_SUCCESS {");
      emitLine(out, indent + 1, "return rc, 0");
      emitLine(out, indent, "}");
    }

    std::string tagExpr = "uint64(obj.Tag)";
    if (helperBindings.unionTagMask) {
      tagExpr =
          helperBindingName(helperBindings.unionTagMask->symbol) + "(" + tagExpr + ")";
    }
    const auto tagErr = nextName("err");
    emitLine(out, indent,
             tagErr + " := dsdlruntime.SetUxx(buffer, offsetBits, " + tagExpr +
                 ", " + std::to_string(tagBits) + ")");
    emitLine(out, indent, "if " + tagErr + " < 0 {");
    emitLine(out, indent + 1, "return " + tagErr + ", 0");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offsetBits += " + std::to_string(tagBits));

    emitLine(out, indent, "switch obj.Tag {");
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      emitLine(out, indent, "case " + std::to_string(field.unionOptionIndex) + ":");
      emitAlignSerialize(out, field.resolvedType.alignmentBits, indent + 1);
      emitSerializeAny(out, field.resolvedType,
                       "obj."+toExportedIdent(field.name), indent + 1,
                       step.arrayLengthPrefixBits, step.fieldFacts);
    }
    emitLine(out, indent, "default:");
    emitLine(out, indent + 1,
             "return -dsdlruntime.DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG, "
             "0");
    emitLine(out, indent, "}");
  }

  void emitDeserializeUnion(std::ostringstream &out, const SemanticSection &section,
                            const std::vector<PlannedFieldStep> &unionBranches,
                            int indent,
                            const LoweredSectionFacts *sectionFacts,
                            const SectionHelperBindingPlan &helperBindings) {
    const auto tagBits = resolveUnionTagBits(section, sectionFacts);
    const auto rawTag = nextName("tag");
    emitLine(out, indent,
             rawTag + " := dsdlruntime.GetU64(buffer, offsetBits, " +
                 std::to_string(tagBits) + ")");
    std::string tagExpr = rawTag;
    if (helperBindings.unionTagMask) {
      tagExpr =
          helperBindingName(helperBindings.unionTagMask->symbol) + "(" + tagExpr + ")";
    }
    emitLine(out, indent, "obj.Tag = uint8(" + tagExpr + ")");

    if (helperBindings.unionTagValidate) {
      const auto validateHelper =
          helperBindingName(helperBindings.unionTagValidate->symbol);
      emitLine(out, indent,
               "if rc := " + validateHelper +
                   "(int64(obj.Tag)); rc != dsdlruntime.DSDL_RUNTIME_SUCCESS {");
      emitLine(out, indent + 1, "return rc, 0");
      emitLine(out, indent, "}");
    }
    emitLine(out, indent, "offsetBits += " + std::to_string(tagBits));

    emitLine(out, indent, "switch obj.Tag {");
    for (const auto &step : unionBranches) {
      const auto &field = *step.field;
      emitLine(out, indent, "case " + std::to_string(field.unionOptionIndex) + ":");
      emitAlignDeserialize(out, field.resolvedType.alignmentBits, indent + 1);
      emitDeserializeAny(out, field.resolvedType,
                         "obj."+toExportedIdent(field.name), indent + 1,
                         step.arrayLengthPrefixBits, step.fieldFacts);
    }
    emitLine(out, indent, "default:");
    emitLine(out, indent + 1,
             "return -dsdlruntime.DSDL_RUNTIME_ERROR_REPRESENTATION_BAD_UNION_TAG, "
             "0");
    emitLine(out, indent, "}");
  }

  void emitSerializeAny(
      std::ostringstream &out, const SemanticFieldType &type,
      const std::string &expr, int indent,
      std::optional<std::uint32_t> arrayLengthPrefixBitsOverride = std::nullopt,
      const LoweredFieldFacts *fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitSerializeArray(out, type, expr, indent, arrayLengthPrefixBitsOverride,
                         fieldFacts);
      return;
    }
    emitSerializeScalar(out, type, expr, indent, fieldFacts);
  }

  void emitDeserializeAny(
      std::ostringstream &out, const SemanticFieldType &type,
      const std::string &expr, int indent,
      std::optional<std::uint32_t> arrayLengthPrefixBitsOverride = std::nullopt,
      const LoweredFieldFacts *fieldFacts = nullptr) {
    if (type.arrayKind != ArrayKind::None) {
      emitDeserializeArray(out, type, expr, indent,
                           arrayLengthPrefixBitsOverride, fieldFacts);
      return;
    }
    emitDeserializeScalar(out, type, expr, indent, fieldFacts);
  }

  void emitSerializeScalar(std::ostringstream &out, const SemanticFieldType &type,
                           const std::string &expr, int indent,
                           const LoweredFieldFacts *fieldFacts) {
    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool: {
      const auto err = nextName("err");
      emitLine(out, indent,
               err + " := dsdlruntime.SetBit(buffer, offsetBits, " + expr + ")");
      emitLine(out, indent, "if " + err + " < 0 {");
      emitLine(out, indent + 1, "return " + err + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offsetBits += 1");
      break;
    }
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      std::string valueExpr = "uint64(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      valueExpr = helper + "(" + valueExpr + ")";
      const auto err = nextName("err");
      emitLine(out, indent,
               err + " := dsdlruntime.SetUxx(buffer, offsetBits, " + valueExpr +
                   ", " + std::to_string(type.bitLength) + ")");
      emitLine(out, indent, "if " + err + " < 0 {");
      emitLine(out, indent + 1, "return " + err + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      std::string valueExpr = "int64(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      valueExpr = helper + "(" + valueExpr + ")";

      const auto err = nextName("err");
      emitLine(out, indent,
               err + " := dsdlruntime.SetIxx(buffer, offsetBits, " + valueExpr +
                   ", " + std::to_string(type.bitLength) + ")");
      emitLine(out, indent, "if " + err + " < 0 {");
      emitLine(out, indent + 1, "return " + err + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
      break;
    }
    case SemanticScalarCategory::Float: {
      std::string valueExpr = "float64(" + expr + ")";
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Serialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      valueExpr = helper + "(" + valueExpr + ")";
      const auto err = nextName("err");
      if (type.bitLength == 16U) {
        emitLine(out, indent,
                 err + " := dsdlruntime.SetF16(buffer, offsetBits, float32(" +
                     valueExpr + "))");
      } else if (type.bitLength == 32U) {
        emitLine(out, indent,
                 err + " := dsdlruntime.SetF32(buffer, offsetBits, float32(" +
                     valueExpr + "))");
      } else {
        emitLine(out, indent,
                 err + " := dsdlruntime.SetF64(buffer, offsetBits, " + valueExpr +
                     ")");
      }
      emitLine(out, indent, "if " + err + " < 0 {");
      emitLine(out, indent + 1, "return " + err + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
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

  void emitDeserializeScalar(std::ostringstream &out, const SemanticFieldType &type,
                             const std::string &expr, int indent,
                             const LoweredFieldFacts *fieldFacts) {
    switch (type.scalarCategory) {
    case SemanticScalarCategory::Bool:
      emitLine(out, indent,
               expr + " = dsdlruntime.GetBit(buffer, offsetBits)");
      emitLine(out, indent, "offsetBits += 1");
      break;
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto raw = nextName("raw");
      emitLine(out, indent,
               raw + " := uint64(dsdlruntime.GetU64(buffer, offsetBits, " +
                   std::to_string(type.bitLength) + "))");
      emitLine(out, indent,
               expr + " = " + unsignedStorageType(type.bitLength) + "(" + helper +
                   "(" + raw + "))");
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
      break;
    }
    case SemanticScalarCategory::SignedInt: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto raw = nextName("raw");
      emitLine(out, indent,
               raw + " := int64(dsdlruntime.GetU64(buffer, offsetBits, " +
                   std::to_string(type.bitLength) + "))");
      emitLine(out, indent,
               expr + " = " + signedStorageType(type.bitLength) + "(" + helper +
                   "(" + raw + "))");
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
      break;
    }
    case SemanticScalarCategory::Float: {
      const auto helperSymbol = resolveScalarHelperSymbol(
          type, fieldFacts, HelperBindingDirection::Deserialize);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      if (type.bitLength == 16U) {
        emitLine(out, indent,
                 expr + " = float32(" + helper +
                     "(float64(dsdlruntime.GetF16(buffer, offsetBits))))");
      } else if (type.bitLength == 32U) {
        emitLine(out, indent,
                 expr + " = float32(" + helper +
                     "(float64(dsdlruntime.GetF32(buffer, offsetBits))))");
      } else {
        emitLine(out, indent,
                 expr + " = " + helper +
                     "(dsdlruntime.GetF64(buffer, offsetBits))");
      }
      emitLine(out, indent, "offsetBits += " + std::to_string(type.bitLength));
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

  void emitSerializeArray(
      std::ostringstream &out, const SemanticFieldType &type,
      const std::string &expr, int indent,
      std::optional<std::uint32_t> arrayLengthPrefixBitsOverride,
      const LoweredFieldFacts *fieldFacts) {
    const auto arrayPlan =
        buildArrayWirePlan(type, fieldFacts, arrayLengthPrefixBitsOverride,
                           HelperBindingDirection::Serialize);
    if (arrayPlan.variable) {
      std::string validateHelper;
      if (arrayPlan.descriptor && !arrayPlan.descriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayPlan.descriptor->validateSymbol);
      }
      if (validateHelper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto validateRc = nextName("lenRc");
      emitLine(out, indent,
               validateRc + " := " + validateHelper +
                   "(int64(len(" + expr + ")))");
      emitLine(out, indent, "if " + validateRc + " < 0 {");
      emitLine(out, indent + 1, "return " + validateRc + ", 0");
      emitLine(out, indent, "}");

      std::string prefixExpr = "uint64(len(" + expr + "))";
      std::string serPrefixHelper;
      if (arrayPlan.descriptor && !arrayPlan.descriptor->prefixSymbol.empty()) {
        serPrefixHelper = helperBindingName(arrayPlan.descriptor->prefixSymbol);
      }
      if (serPrefixHelper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      prefixExpr = serPrefixHelper + "(" + prefixExpr + ")";
      const auto err = nextName("err");
      emitLine(out, indent,
               err + " := dsdlruntime.SetUxx(buffer, offsetBits, " + prefixExpr +
                   ", " + std::to_string(arrayPlan.prefixBits) + ")");
      emitLine(out, indent, "if " + err + " < 0 {");
      emitLine(out, indent + 1, "return " + err + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent,
               "offsetBits += " + std::to_string(arrayPlan.prefixBits));
    }

    const auto index = nextName("index");
    const auto count = arrayPlan.variable
                           ? "len(" + expr + ")"
                           : std::to_string(type.arrayCapacity);
    emitLine(out, indent,
             "for " + index + " := 0; " + index + " < " + count + "; " + index +
                 "++ {");
    emitSerializeScalar(out, arrayElementType(type), expr + "[" + index + "]",
                        indent + 1, fieldFacts);
    emitLine(out, indent, "}");
  }

  void emitDeserializeArray(
      std::ostringstream &out, const SemanticFieldType &type,
      const std::string &expr, int indent,
      std::optional<std::uint32_t> arrayLengthPrefixBitsOverride,
      const LoweredFieldFacts *fieldFacts) {
    const auto arrayPlan =
        buildArrayWirePlan(type, fieldFacts, arrayLengthPrefixBitsOverride,
                           HelperBindingDirection::Deserialize);
    const auto count = nextName("count");
    if (arrayPlan.variable) {
      const auto rawCount = nextName("countRaw");
      emitLine(out, indent,
               rawCount + " := dsdlruntime.GetU64(buffer, offsetBits, " +
                   std::to_string(arrayPlan.prefixBits) + ")");
      emitLine(out, indent,
               "offsetBits += " + std::to_string(arrayPlan.prefixBits));
      std::string countExpr = "int(" + rawCount + ")";
      std::string deserPrefixHelper;
      if (arrayPlan.descriptor && !arrayPlan.descriptor->prefixSymbol.empty()) {
        deserPrefixHelper = helperBindingName(arrayPlan.descriptor->prefixSymbol);
      }
      if (deserPrefixHelper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      countExpr = "int(" + deserPrefixHelper + "(" + rawCount + "))";
      emitLine(out, indent, count + " := " + countExpr);
      std::string validateHelper;
      if (arrayPlan.descriptor && !arrayPlan.descriptor->validateSymbol.empty()) {
        validateHelper = helperBindingName(arrayPlan.descriptor->validateSymbol);
      }
      if (validateHelper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto validateRc = nextName("lenRc");
      emitLine(out, indent,
               validateRc + " := " + validateHelper + "(int64(" + count + "))");
      emitLine(out, indent, "if " + validateRc + " < 0 {");
      emitLine(out, indent + 1, "return " + validateRc + ", 0");
      emitLine(out, indent, "}");
      const auto itemType =
          goBaseFieldType(arrayElementType(type), ctx_, currentPackagePath_,
                          importAliases_);
      emitLine(out, indent, expr + " = make([]" + itemType + ", " + count + ")");
    } else {
      emitLine(out, indent,
               count + " := " + std::to_string(type.arrayCapacity));
    }

    const auto index = nextName("index");
    emitLine(out, indent,
             "for " + index + " := 0; " + index + " < " + count + "; " + index +
                 "++ {");
    emitDeserializeScalar(out, arrayElementType(type), expr + "[" + index + "]",
                          indent + 1, fieldFacts);
    emitLine(out, indent, "}");
  }

  void emitSerializeComposite(std::ostringstream &out, const SemanticFieldType &type,
                              const std::string &expr, int indent,
                              const LoweredFieldFacts *fieldFacts) {
    const auto sizeVar = nextName("sizeBytes");
    const auto maxBytes = (type.bitLengthSet.max() + 7) / 8;
    if (!type.compositeSealed) {
      emitLine(out, indent, "offsetBits += 32");
    }
    emitLine(out, indent, sizeVar + " := " + std::to_string(maxBytes));
    if (!type.compositeSealed) {
      const auto remainingVar = nextName("remaining");
      emitLine(out, indent,
               remainingVar +
                   " := len(buffer) - dsdlruntime.ChooseMin(offsetBits/8, "
                   "len(buffer))");
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto validateRc = nextName("rc");
      emitLine(out, indent,
               validateRc + " := " + helper + "(int64(" + sizeVar + "), int64(" +
                   remainingVar + "))");
      emitLine(out, indent, "if " + validateRc + " < 0 {");
      emitLine(out, indent + 1, "return " + validateRc + ", 0");
      emitLine(out, indent, "}");
    }
    const auto startVar = nextName("start");
    const auto endVar = nextName("end");
    emitLine(out, indent,
             startVar + " := dsdlruntime.ChooseMin(offsetBits/8, len(buffer))");
    emitLine(out, indent,
             endVar + " := dsdlruntime.ChooseMin(" + startVar + "+" + sizeVar +
                 ", len(buffer))");
    const auto rcVar = nextName("rc");
    const auto consumedVar = nextName("consumed");
    emitLine(out, indent,
             rcVar + ", " + consumedVar + " := " + expr + ".Serialize(buffer[" +
                 startVar + ":" + endVar + "])");
    emitLine(out, indent, "if " + rcVar + " < 0 {");
    emitLine(out, indent + 1, "return " + rcVar + ", 0");
    emitLine(out, indent, "}");
    emitLine(out, indent, sizeVar + " = " + consumedVar);
    if (!type.compositeSealed) {
      const auto hdrErr = nextName("err");
      emitLine(out, indent,
               hdrErr + " := dsdlruntime.SetUxx(buffer, offsetBits-32, uint64(" +
                   sizeVar + "), 32)");
      emitLine(out, indent, "if " + hdrErr + " < 0 {");
      emitLine(out, indent + 1, "return " + hdrErr + ", 0");
      emitLine(out, indent, "}");
    }
    emitLine(out, indent, "offsetBits += " + sizeVar + " * 8");
  }

  void emitDeserializeComposite(std::ostringstream &out,
                                const SemanticFieldType &type,
                                const std::string &expr, int indent,
                                const LoweredFieldFacts *fieldFacts) {
    if (!type.compositeSealed) {
      const auto sizeVar = nextName("sizeBytes");
      emitLine(out, indent,
               sizeVar + " := int(dsdlruntime.GetU32(buffer, offsetBits, 32))");
      emitLine(out, indent, "offsetBits += 32");
      const auto remainingVar = nextName("remaining");
      emitLine(out, indent,
               remainingVar +
                   " := capacityBytes - dsdlruntime.ChooseMin(offsetBits/8, "
                   "capacityBytes)");
      const auto helperSymbol =
          resolveDelimiterValidateHelperSymbol(type, fieldFacts);
      const auto helper = helperSymbol.empty() ? std::string{}
                                               : helperBindingName(helperSymbol);
      if (helper.empty()) {
        emitLine(out, indent,
                 "return -dsdlruntime.DSDL_RUNTIME_ERROR_INVALID_ARGUMENT, 0");
        return;
      }
      const auto validateRc = nextName("rc");
      emitLine(out, indent,
               validateRc + " := " + helper + "(int64(" + sizeVar + "), int64(" +
                   remainingVar + "))");
      emitLine(out, indent, "if " + validateRc + " < 0 {");
      emitLine(out, indent + 1, "return " + validateRc + ", 0");
      emitLine(out, indent, "}");
      const auto startVar = nextName("start");
      const auto endVar = nextName("end");
      emitLine(out, indent,
               startVar + " := dsdlruntime.ChooseMin(offsetBits/8, len(buffer))");
      emitLine(out, indent,
               endVar + " := dsdlruntime.ChooseMin(" + startVar + "+" + sizeVar +
                   ", len(buffer))");
      const auto rcVar = nextName("rc");
      const auto consumedVar = nextName("consumed");
      emitLine(out, indent,
               rcVar + ", " + consumedVar + " := " + expr +
                   ".Deserialize(buffer[" + startVar + ":" + endVar + "])");
      emitLine(out, indent, "_ = " + consumedVar);
      emitLine(out, indent, "if " + rcVar + " < 0 {");
      emitLine(out, indent + 1, "return " + rcVar + ", 0");
      emitLine(out, indent, "}");
      emitLine(out, indent, "offsetBits += " + sizeVar + " * 8");
      return;
    }

    const auto startVar = nextName("start");
    emitLine(out, indent,
             startVar + " := dsdlruntime.ChooseMin(offsetBits/8, len(buffer))");
    const auto rcVar = nextName("rc");
    const auto consumedVar = nextName("consumed");
    emitLine(out, indent,
             rcVar + ", " + consumedVar + " := " + expr +
                 ".Deserialize(buffer[" + startVar + ":len(buffer)])");
    emitLine(out, indent, "if " + rcVar + " < 0 {");
    emitLine(out, indent + 1, "return " + rcVar + ", 0");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offsetBits += " + consumedVar + " * 8");
  }
};

void emitSectionType(std::ostringstream &out, const EmitterContext &ctx,
                     const std::string &typeName, const std::string &fullName,
                     std::uint32_t majorVersion, std::uint32_t minorVersion,
                     const SemanticSection &section,
                     const std::string &currentPackagePath,
                     const std::map<std::string, std::string> &importAliases,
                     const LoweredSectionFacts *sectionFacts) {
  const auto typeConstPrefix = toUpperSnake(typeName);
  emitLine(out, 0, "const " + typeConstPrefix + "_FULL_NAME = \"" + fullName + "\"");
  emitLine(out, 0,
           "const " + typeConstPrefix + "_FULL_NAME_AND_VERSION = \"" + fullName +
               "." + std::to_string(majorVersion) + "." + std::to_string(minorVersion) +
               "\"");
  emitLine(out, 0,
           "const " + typeConstPrefix + "_EXTENT_BYTES = " +
               std::to_string(section.extentBits.value_or(0) / 8));
  emitLine(out, 0,
           "const " + typeConstPrefix + "_SERIALIZATION_BUFFER_SIZE_BYTES = " +
               std::to_string((section.serializationBufferSizeBits + 7) / 8));

  if (section.isUnion) {
    std::size_t optionCount = 0;
    for (const auto &f : section.fields) {
      if (!f.isPadding) {
        ++optionCount;
      }
    }
    emitLine(out, 0,
             "const " + typeConstPrefix + "_UNION_OPTION_COUNT = " +
                 std::to_string(optionCount));
  }

  for (const auto &c : section.constants) {
    emitLine(out, 0,
             "const " + typeConstPrefix + "_" + toUpperSnake(c.name) + " = " +
                 goConstValue(c.value));
  }
  out << "\n";

  emitLine(out, 0, "type " + typeName + " struct {");
  for (const auto &field : section.fields) {
    if (field.isPadding) {
      continue;
    }
    emitLine(out, 1,
             toExportedIdent(field.name) + " " +
                 goFieldType(field.resolvedType, ctx, currentPackagePath, importAliases));
  }
  if (section.isUnion) {
    emitLine(out, 1, "Tag uint8");
  }
  if (section.fields.empty()) {
    emitLine(out, 1, "_ uint8");
  }
  emitLine(out, 0, "}");
  out << "\n";

  FunctionBodyEmitter body(ctx, currentPackagePath, importAliases);
  body.emitSerializeFunction(out, typeName, section, sectionFacts);
  out << "\n";
  body.emitDeserializeFunction(out, typeName, section, sectionFacts);
}

std::string renderDefinitionFile(const SemanticDefinition &def, const EmitterContext &ctx,
                                 const std::string &moduleName,
                                 const LoweredFactsMap &loweredFacts) {
  const auto currentPackagePath = ctx.packagePath(def.info);
  const auto packageName = packageNameFromPath(currentPackagePath);
  const auto imports = computeImportAliases(def, ctx);

  std::ostringstream out;
  emitLine(out, 0, "package " + packageName);
  out << "\n";

  emitLine(out, 0, "import (");
  emitLine(out, 1, "dsdlruntime \"" + moduleName + "/dsdlruntime\"");
  for (const auto &[path, alias] : imports) {
    emitLine(out, 1, alias + " \"" + moduleName + "/" + path + "\"");
  }
  emitLine(out, 0, ")");
  out << "\n";

  const auto baseType = ctx.goTypeName(def.info);
  if (!def.isService) {
    emitSectionType(out, ctx, baseType, def.info.fullName, def.info.majorVersion,
                    def.info.minorVersion, def.request, currentPackagePath, imports,
                    findLoweredSectionFacts(loweredFacts, def, ""));
    return out.str();
  }

  const auto reqType = baseType + "_Request";
  const auto respType = baseType + "_Response";
  emitSectionType(out, ctx, reqType, def.info.fullName + ".Request",
                  def.info.majorVersion, def.info.minorVersion, def.request,
                  currentPackagePath, imports,
                  findLoweredSectionFacts(loweredFacts, def, "request"));
  out << "\n";
  if (def.response) {
    emitSectionType(out, ctx, respType, def.info.fullName + ".Response",
                    def.info.majorVersion, def.info.minorVersion, *def.response,
                    currentPackagePath, imports,
                    findLoweredSectionFacts(loweredFacts, def, "response"));
    out << "\n";
  }
  emitLine(out, 0, "type " + baseType + " = " + reqType);
  return out.str();
}

llvm::Expected<std::string> loadGoRuntime() {
  const std::filesystem::path runtimePath =
      std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "go" /
      "dsdl_runtime.go";
  std::ifstream in(runtimePath.string());
  if (!in) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to read Go runtime");
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

std::string renderGoMod(const GoEmitOptions &options) {
  std::ostringstream out;
  out << "module " << options.moduleName << "\n\n";
  out << "go 1.22\n";
  return out.str();
}

} // namespace

llvm::Error emitGo(const SemanticModule &semantic, mlir::ModuleOp module,
                   const GoEmitOptions &options,
                   DiagnosticEngine &diagnostics) {
  if (options.outDir.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output directory is required");
  }
  LoweredFactsMap loweredFacts;
  if (!collectLoweredFactsFromMlir(semantic, module, diagnostics, "Go",
                                   &loweredFacts,
                                   options.optimizeLoweredSerDes)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "MLIR schema coverage validation failed for Go emission");
  }

  std::filesystem::path outRoot(options.outDir);
  std::filesystem::create_directories(outRoot);

  if (options.emitGoMod) {
    if (auto err = writeFile(outRoot / "go.mod", renderGoMod(options))) {
      return err;
    }
  }

  auto runtime = loadGoRuntime();
  if (!runtime) {
    return runtime.takeError();
  }
  std::filesystem::create_directories(outRoot / "dsdlruntime");
  if (auto err = writeFile(outRoot / "dsdlruntime" / "dsdl_runtime.go", *runtime)) {
    return err;
  }

  EmitterContext ctx(semantic);

  for (const auto &def : semantic.definitions) {
    const auto dirRel = ctx.packagePath(def.info);
    std::filesystem::path dir = outRoot;
    if (!dirRel.empty()) {
      dir /= dirRel;
    }
    std::filesystem::create_directories(dir);
    if (auto err = writeFile(dir / ctx.goFileName(def.info),
                             renderDefinitionFile(def, ctx, options.moduleName,
                                                  loweredFacts))) {
      return err;
    }
  }

  return llvm::Error::success();
}

} // namespace llvmdsdl
