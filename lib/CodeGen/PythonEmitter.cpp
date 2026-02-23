//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements Python backend code emission from lowered DSDL modules.
///
/// This file emits Python dataclass models, serializer/deserializer methods,
/// and runtime wiring from lowering contracts.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/PythonEmitter.h"

#include <llvm/ADT/StringRef.h>
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
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <utility>
#include <variant>

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/CodegenDiagnosticText.h"
#include "llvmdsdl/CodeGen/HelperBindingRender.h"
#include "llvmdsdl/CodeGen/RuntimeHelperBindings.h"
#include "llvmdsdl/CodeGen/ScriptedBodyPlan.h"
#include "llvmdsdl/CodeGen/SectionHelperBindingPlan.h"
#include "llvmdsdl/CodeGen/RuntimeLoweredPlan.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Rational.h"
#include "mlir/IR/BuiltinOps.h"

namespace llvmdsdl
{
class DiagnosticEngine;

namespace
{

bool isPyKeyword(const std::string& name)
{
    static const std::set<std::string> kKeywords = {"False",  "None",     "True",  "and",    "as",       "assert",
                                                    "async",  "await",    "break", "class",  "continue", "def",
                                                    "del",    "elif",     "else",  "except", "finally",  "for",
                                                    "from",   "global",   "if",    "import", "in",       "is",
                                                    "lambda", "nonlocal", "not",   "or",     "pass",     "raise",
                                                    "return", "try",      "while", "with",   "yield",    "match",
                                                    "case"};
    return kKeywords.contains(name);
}

std::string sanitizePyIdent(std::string name)
{
    if (name.empty())
    {
        return "_";
    }
    for (char& c : name)
    {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
        {
            c = '_';
        }
    }
    if (std::isdigit(static_cast<unsigned char>(name.front())))
    {
        name.insert(name.begin(), '_');
    }
    if (isPyKeyword(name))
    {
        name += "_";
    }
    return name;
}

std::string toSnakeCase(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);

    bool prevUnderscore = false;
    for (std::size_t i = 0; i < in.size(); ++i)
    {
        const char c    = in[i];
        const char prev = (i > 0) ? in[i - 1] : '\0';
        const char next = (i + 1 < in.size()) ? in[i + 1] : '\0';
        if (!std::isalnum(static_cast<unsigned char>(c)))
        {
            if (!out.empty() && !prevUnderscore)
            {
                out.push_back('_');
                prevUnderscore = true;
            }
            continue;
        }

        if (std::isupper(static_cast<unsigned char>(c)))
        {
            const bool boundary =
                std::islower(static_cast<unsigned char>(prev)) ||
                (std::isupper(static_cast<unsigned char>(prev)) && std::islower(static_cast<unsigned char>(next)));
            if (!out.empty() && !prevUnderscore && boundary)
            {
                out.push_back('_');
            }
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            prevUnderscore = false;
        }
        else
        {
            out.push_back(c);
            prevUnderscore = (c == '_');
        }
    }

    if (out.empty())
    {
        out = "_";
    }
    if (std::isdigit(static_cast<unsigned char>(out.front())))
    {
        out.insert(out.begin(), '_');
    }
    return sanitizePyIdent(out);
}

std::string toPascalCase(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);

    bool upperNext = true;
    for (char c : in)
    {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
        {
            upperNext = true;
            continue;
        }
        if (c == '_')
        {
            upperNext = true;
            continue;
        }
        if (upperNext)
        {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
            upperNext = false;
        }
        else
        {
            out.push_back(c);
        }
    }

    if (out.empty())
    {
        out = "X";
    }
    return sanitizePyIdent(out);
}

std::string toUpperSnake(const std::string& in)
{
    auto out = toSnakeCase(in);
    for (char& c : out)
    {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string pyConstValue(const Value& value)
{
    if (const auto* b = std::get_if<bool>(&value.data))
    {
        return *b ? "True" : "False";
    }
    if (const auto* r = std::get_if<Rational>(&value.data))
    {
        if (r->isInteger())
        {
            return std::to_string(r->asInteger().value());
        }
        std::ostringstream out;
        out << "(" << r->numerator() << " / " << r->denominator() << ")";
        return out.str();
    }
    if (const auto* s = std::get_if<std::string>(&value.data))
    {
        std::string escaped;
        escaped.reserve(s->size() + 2);
        escaped.push_back('"');
        for (char c : *s)
        {
            if (c == '\\' || c == '"')
            {
                escaped.push_back('\\');
            }
            escaped.push_back(c);
        }
        escaped.push_back('"');
        return escaped;
    }
    return value.str();
}

llvm::Error writeFile(const std::filesystem::path& p, llvm::StringRef content)
{
    std::error_code      ec;
    llvm::raw_fd_ostream os(p.string(), ec, llvm::sys::fs::OF_Text);
    if (ec)
    {
        return llvm::createStringError(ec, "failed to open %s", p.string().c_str());
    }
    os << content;
    os.close();
    return llvm::Error::success();
}

void emitLine(std::ostringstream& out, const int indent, const std::string& line)
{
    out << std::string(static_cast<std::size_t>(indent) * 4U, ' ') << line << '\n';
}

std::string runtimeSerializeFn(const std::string& typeName)
{
    return "_serialize_" + sanitizePyIdent(typeName);
}

std::string runtimeDeserializeFn(const std::string& typeName)
{
    return "_deserialize_" + sanitizePyIdent(typeName);
}

std::string joinDotted(const std::vector<std::string>& parts)
{
    std::string out;
    for (const auto& p : parts)
    {
        if (!out.empty())
        {
            out += ".";
        }
        out += p;
    }
    return out;
}

std::vector<std::string> splitPackageName(const std::string& packageName)
{
    std::vector<std::string> out;
    std::string              current;
    for (char c : packageName)
    {
        if (c == '.')
        {
            if (!current.empty())
            {
                out.push_back(sanitizePyIdent(toSnakeCase(current)));
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty())
    {
        out.push_back(sanitizePyIdent(toSnakeCase(current)));
    }
    if (out.empty())
    {
        out.push_back("dsdl_gen");
    }
    return out;
}

class EmitterContext final
{
public:
    EmitterContext(const SemanticModule& semantic, std::vector<std::string> packageComponents)
        : packageComponents_(std::move(packageComponents))
    {
        for (const auto& def : semantic.definitions)
        {
            byKey_.emplace(loweredTypeKey(def.info.fullName, def.info.majorVersion, def.info.minorVersion), &def);
        }
    }

    const SemanticDefinition* find(const SemanticTypeRef& ref) const
    {
        const auto it = byKey_.find(loweredTypeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
        if (it == byKey_.end())
        {
            return nullptr;
        }
        return it->second;
    }

    std::string namespacePath(const DiscoveredDefinition& info) const
    {
        std::string out;
        for (const auto& component : info.namespaceComponents)
        {
            if (!out.empty())
            {
                out += "/";
            }
            out += toSnakeCase(component);
        }
        return out;
    }

    std::string typeName(const DiscoveredDefinition& info) const
    {
        return toPascalCase(info.shortName) + "_" + std::to_string(info.majorVersion) + "_" +
               std::to_string(info.minorVersion);
    }

    std::string typeName(const SemanticTypeRef& ref) const
    {
        if (const auto* def = find(ref))
        {
            return typeName(def->info);
        }

        DiscoveredDefinition tmp;
        tmp.shortName    = ref.shortName;
        tmp.majorVersion = ref.majorVersion;
        tmp.minorVersion = ref.minorVersion;
        return typeName(tmp);
    }

    std::string fileStem(const DiscoveredDefinition& info) const
    {
        return toSnakeCase(info.shortName) + "_" + std::to_string(info.majorVersion) + "_" +
               std::to_string(info.minorVersion);
    }

    std::filesystem::path relativeFilePath(const DiscoveredDefinition& info) const
    {
        std::filesystem::path rel;
        const auto            nsPath = namespacePath(info);
        if (!nsPath.empty())
        {
            rel /= nsPath;
        }
        rel /= fileStem(info) + ".py";
        return rel;
    }

    std::filesystem::path relativeFilePath(const SemanticTypeRef& ref) const
    {
        if (const auto* def = find(ref))
        {
            return relativeFilePath(def->info);
        }

        std::filesystem::path rel;
        for (const auto& component : ref.namespaceComponents)
        {
            rel /= toSnakeCase(component);
        }
        rel /= toSnakeCase(ref.shortName) + "_" + std::to_string(ref.majorVersion) + "_" +
               std::to_string(ref.minorVersion) + ".py";
        return rel;
    }

    std::string packageName() const
    {
        return joinDotted(packageComponents_);
    }

    std::filesystem::path packageRootPath() const
    {
        std::filesystem::path out;
        for (const auto& c : packageComponents_)
        {
            out /= c;
        }
        return out;
    }

    std::string modulePath(const DiscoveredDefinition& info) const
    {
        const auto rel = relativeFilePath(info);
        return modulePathFromRelPath(rel);
    }

    std::string modulePath(const SemanticTypeRef& ref) const
    {
        if (const auto* def = find(ref))
        {
            return modulePath(def->info);
        }
        return modulePathFromRelPath(relativeFilePath(ref));
    }

private:
    std::string modulePathFromRelPath(const std::filesystem::path& relPath) const
    {
        std::vector<std::string> parts = packageComponents_;
        for (const auto& comp : relPath)
        {
            auto part = comp.string();
            if (part.empty() || part == ".")
            {
                continue;
            }
            if (part.ends_with(".py"))
            {
                part.resize(part.size() - 3);
            }
            parts.push_back(part);
        }
        return joinDotted(parts);
    }

    std::vector<std::string>                                   packageComponents_;
    std::unordered_map<std::string, const SemanticDefinition*> byKey_;
};

std::string pyFieldBaseType(const SemanticFieldType& type, const EmitterContext& ctx)
{
    switch (type.scalarCategory)
    {
    case SemanticScalarCategory::Bool:
        return "bool";
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt:
    case SemanticScalarCategory::SignedInt:
        return "int";
    case SemanticScalarCategory::Float:
        return "float";
    case SemanticScalarCategory::Void:
        return "int";
    case SemanticScalarCategory::Composite:
        if (type.compositeType)
        {
            return ctx.typeName(*type.compositeType);
        }
        return "object";
    }
    return "object";
}

std::string pyFieldType(const SemanticFieldType& type, const EmitterContext& ctx)
{
    const auto base = pyFieldBaseType(type, ctx);
    if (type.arrayKind == ArrayKind::None)
    {
        return base;
    }
    return "list[" + base + "]";
}

std::string pyDefaultExpr(const SemanticFieldType& type, const EmitterContext& ctx)
{
    if (type.arrayKind != ArrayKind::None)
    {
        return "field(default_factory=list)";
    }

    switch (type.scalarCategory)
    {
    case SemanticScalarCategory::Bool:
        return "False";
    case SemanticScalarCategory::Float:
        return "0.0";
    case SemanticScalarCategory::Composite:
        if (type.compositeType)
        {
            return "field(default_factory=" + ctx.typeName(*type.compositeType) + ")";
        }
        return "None";
    case SemanticScalarCategory::Void:
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt:
    case SemanticScalarCategory::SignedInt:
        return "0";
    }
    return "None";
}

struct ImportSpec final
{
    std::string modulePath;
    std::string typeName;
};

std::vector<ImportSpec> collectCompositeImports(const SemanticSection&      section,
                                                const DiscoveredDefinition& owner,
                                                const EmitterContext&       ctx)
{
    std::map<std::string, std::set<std::string>> importsByModule;
    const auto                                   ownerPath = ctx.relativeFilePath(owner);

    for (const auto& field : section.fields)
    {
        if (field.isPadding || !field.resolvedType.compositeType)
        {
            continue;
        }
        const auto& ref        = *field.resolvedType.compositeType;
        const auto  targetPath = ctx.relativeFilePath(ref);
        if (targetPath == ownerPath)
        {
            continue;
        }
        importsByModule[ctx.modulePath(ref)].insert(ctx.typeName(ref));
    }

    std::vector<ImportSpec> out;
    for (const auto& [modulePath, names] : importsByModule)
    {
        for (const auto& name : names)
        {
            out.push_back({modulePath, name});
        }
    }
    return out;
}

void emitSectionConstants(std::ostringstream& out, const std::string& prefix, const SemanticSection& section)
{
    for (const auto& constant : section.constants)
    {
        const auto constName = toUpperSnake(prefix) + "_" + toUpperSnake(constant.name);
        emitLine(out, 0, constName + " = " + pyConstValue(constant.value));
    }
}

void emitClassMethods(std::ostringstream& out, const std::string& typeName)
{
    emitLine(out, 1, "def serialize(self) -> bytes:");
    emitLine(out, 2, "return " + runtimeSerializeFn(typeName) + "(self)");
    emitLine(out, 0, "");

    emitLine(out, 1, "@classmethod");
    emitLine(out, 1, "def deserialize(cls, data: bytes | bytearray | memoryview) -> \"" + typeName + "\":");
    emitLine(out, 2, "value, _consumed = " + runtimeDeserializeFn(typeName) + "(bytes(data))");
    emitLine(out, 2, "return value");
    emitLine(out, 0, "");

    emitLine(out, 1, "def _serialize_to(self, writer: object) -> None:");
    emitLine(out, 2, "writer.write(self.serialize())");
    emitLine(out, 0, "");

    emitLine(out, 1, "@classmethod");
    emitLine(out, 1, "def _deserialize_from(cls, reader: object) -> \"" + typeName + "\":");
    emitLine(out, 2, "return cls.deserialize(reader.read())");
}

void emitStructSectionType(std::ostringstream&    out,
                           const std::string&     typeName,
                           const SemanticSection& section,
                           const EmitterContext&  ctx)
{
    emitLine(out, 0, "@dataclass(slots=True)");
    emitLine(out, 0, "class " + typeName + ":");

    bool emittedField = false;
    for (const auto& field : section.fields)
    {
        if (field.isPadding)
        {
            continue;
        }
        emittedField         = true;
        const auto fieldName = sanitizePyIdent(toSnakeCase(field.name));
        emitLine(out,
                 1,
                 fieldName + ": " + pyFieldType(field.resolvedType, ctx) + " = " +
                     pyDefaultExpr(field.resolvedType, ctx));
    }
    if (!emittedField)
    {
        emitLine(out, 1, "pass");
    }
    else
    {
        emitLine(out, 0, "");
    }
    emitClassMethods(out, typeName);
}

void emitUnionSectionType(std::ostringstream&    out,
                          const std::string&     typeName,
                          const SemanticSection& section,
                          const EmitterContext&  ctx)
{
    emitLine(out, 0, "@dataclass(slots=True)");
    emitLine(out, 0, "class " + typeName + ":");
    emitLine(out, 1, "_tag: int = 0");

    for (const auto& field : section.fields)
    {
        if (field.isPadding)
        {
            continue;
        }
        const auto fieldName = sanitizePyIdent(toSnakeCase(field.name));
        emitLine(out, 1, fieldName + ": " + pyFieldType(field.resolvedType, ctx) + " | None = None");
    }

    emitLine(out, 0, "");
    emitClassMethods(out, typeName);
}

void emitSectionType(std::ostringstream&    out,
                     const std::string&     typeName,
                     const SemanticSection& section,
                     const EmitterContext&  ctx)
{
    if (section.isUnion)
    {
        emitUnionSectionType(out, typeName, section, ctx);
    }
    else
    {
        emitStructSectionType(out, typeName, section, ctx);
    }
}

std::string compositeTypeName(const RuntimeFieldPlan& field, const EmitterContext& ctx)
{
    return field.compositeType ? ctx.typeName(*field.compositeType) : std::string{"object"};
}

std::string helperBindingNamePy(const std::string& helperSymbol)
{
    return "mlir_" + sanitizePyIdent(helperSymbol);
}

void emitPyRuntimeSerializeCompositeValue(std::ostringstream&     out,
                                          int                     indent,
                                          const RuntimeFieldPlan& field,
                                          const std::string&      valueExpr,
                                          const EmitterContext&   ctx)
{
    const auto nestedVar = field.fieldName + "_bytes";
    emitLine(out, indent, nestedVar + " = " + valueExpr + ".serialize()");
    if (field.compositeSealed)
    {
        emitLine(out,
                 indent,
                 "dsdl_runtime.copy_bits(out, offset_bits, " + nestedVar + ", 0, " + std::to_string(field.bitLength) +
                     ")");
        emitLine(out, indent, "offset_bits += " + std::to_string(field.bitLength));
        return;
    }

    const auto sizeVar         = field.fieldName + "_size_bytes";
    const auto remainingVar    = field.fieldName + "_remaining_bytes";
    const auto maxPayloadBytes = std::to_string((field.compositePayloadMaxBits + 7) / 8);
    emitLine(out, indent, sizeVar + " = len(" + nestedVar + ")");
    emitLine(out, indent, "if " + sizeVar + " > " + maxPayloadBytes + ":");
    emitLine(out,
             indent + 1,
             "raise ValueError(\"" +
                 codegen_diagnostic_text::encodedCompositePayloadExceedsMaxPayloadBytes(field.fieldName,
                                                                                        maxPayloadBytes) +
                 "\")");
    emitLine(out, indent, remainingVar + " = len(out) - min(offset_bits // 8, len(out))");
    emitLine(out, indent, "if " + sizeVar + " > " + remainingVar + ":");
    emitLine(out,
             indent + 1,
             "raise ValueError(\"" +
                 codegen_diagnostic_text::encodedCompositePayloadExceedsRemainingBufferSpace(field.fieldName) + "\")");
    emitLine(out, indent, "dsdl_runtime.write_unsigned(out, offset_bits, 32, " + sizeVar + ", False)");
    emitLine(out, indent, "offset_bits += 32");
    emitLine(out, indent, "dsdl_runtime.copy_bits(out, offset_bits, " + nestedVar + ", 0, " + sizeVar + " * 8)");
    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8");
}

void emitPyRuntimeDeserializeCompositeValue(std::ostringstream&     out,
                                            int                     indent,
                                            const RuntimeFieldPlan& field,
                                            const std::string&      targetExpr,
                                            const EmitterContext&   ctx,
                                            const std::string&      delimiterValidateHelper = {})
{
    const auto typeName = compositeTypeName(field, ctx);
    if (field.compositeSealed)
    {
        const auto nestedVar = field.fieldName + "_bytes";
        emitLine(out,
                 indent,
                 nestedVar + " = dsdl_runtime.extract_bits(data, offset_bits, " + std::to_string(field.bitLength) +
                     ")");
        emitLine(out, indent, targetExpr + " = " + typeName + ".deserialize(" + nestedVar + ")");
        emitLine(out, indent, "offset_bits += " + std::to_string(field.bitLength));
        return;
    }

    const auto sizeVar      = field.fieldName + "_size_bytes";
    const auto remainingVar = field.fieldName + "_remaining_bytes";
    const auto startVar     = field.fieldName + "_start_byte";
    const auto endVar       = field.fieldName + "_end_byte";
    const auto nestedVar    = field.fieldName + "_bytes";
    emitLine(out, indent, sizeVar + " = int(dsdl_runtime.read_unsigned(data, offset_bits, 32))");
    emitLine(out, indent, "offset_bits += 32");
    emitLine(out, indent, remainingVar + " = len(data) - min(offset_bits // 8, len(data))");
    if (!delimiterValidateHelper.empty())
    {
        emitLine(out, indent, "if not " + delimiterValidateHelper + "(" + sizeVar + ", " + remainingVar + "):");
    }
    else
    {
        emitLine(out, indent, "if " + sizeVar + " < 0 or " + sizeVar + " > " + remainingVar + ":");
    }
    emitLine(out,
             indent + 1,
             "raise ValueError(\"" +
                 codegen_diagnostic_text::decodedCompositePayloadExceedsRemainingBufferSpace(field.fieldName) + "\")");
    emitLine(out, indent, startVar + " = min(offset_bits // 8, len(data))");
    emitLine(out, indent, endVar + " = min(" + startVar + " + " + sizeVar + ", len(data))");
    emitLine(out, indent, nestedVar + " = data[" + startVar + ":" + endVar + "]");
    emitLine(out, indent, targetExpr + " = " + typeName + ".deserialize(" + nestedVar + ")");
    emitLine(out, indent, "offset_bits += " + sizeVar + " * 8");
}

void emitPyRuntimeAlignSerialize(std::ostringstream& out,
                                 int                 indent,
                                 std::int64_t        alignmentBits,
                                 const std::string&  prefix)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    const auto alignedVar = prefix + "_aligned_offset_bits";
    const auto bitVar     = prefix + "_align_bit";
    emitLine(out,
             indent,
             alignedVar + " = ((offset_bits + " + std::to_string(alignmentBits - 1) + ") // " +
                 std::to_string(alignmentBits) + ") * " + std::to_string(alignmentBits));
    emitLine(out, indent, "for " + bitVar + " in range(offset_bits, " + alignedVar + "):");
    emitLine(out, indent + 1, "dsdl_runtime.set_bit(out, " + bitVar + ", False)");
    emitLine(out, indent, "offset_bits = " + alignedVar);
}

void emitPyRuntimeAlignDeserialize(std::ostringstream& out, int indent, std::int64_t alignmentBits)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    emitLine(out,
             indent,
             "offset_bits = ((offset_bits + " + std::to_string(alignmentBits - 1) + ") // " +
                 std::to_string(alignmentBits) + ") * " + std::to_string(alignmentBits));
}

void emitPyRuntimeSerializePadding(std::ostringstream& out, int indent, const RuntimeFieldPlan& field)
{
    if (field.bitLength <= 0)
    {
        return;
    }
    const auto bitVar = field.fieldName + "_padding_bit";
    emitLine(out, indent, "for " + bitVar + " in range(" + std::to_string(field.bitLength) + "):");
    emitLine(out, indent + 1, "dsdl_runtime.set_bit(out, offset_bits + " + bitVar + ", False)");
    emitLine(out, indent, "offset_bits += " + std::to_string(field.bitLength));
}

void emitPyRuntimeDeserializePadding(std::ostringstream& out, int indent, const RuntimeFieldPlan& field)
{
    if (field.bitLength <= 0)
    {
        return;
    }
    emitLine(out, indent, "offset_bits += " + std::to_string(field.bitLength));
}

void emitPyRuntimeSerializeFieldValue(std::ostringstream&            out,
                                      int                            indent,
                                      const RuntimeFieldPlan&        field,
                                      const std::string&             valueExpr,
                                      const EmitterContext&          ctx,
                                      const RuntimeFieldHelperNames& helpers)
{
    const auto bits       = std::to_string(field.bitLength);
    const auto saturating = field.castMode == CastMode::Saturated ? "True" : "False";

    emitPyRuntimeAlignSerialize(out, indent, field.alignmentBits, field.fieldName);
    if (field.arrayKind == RuntimeArrayKind::None)
    {
        if (field.kind == RuntimeFieldKind::Padding)
        {
            emitPyRuntimeSerializePadding(out, indent, field);
        }
        else if (field.kind == RuntimeFieldKind::Bool)
        {
            emitLine(out, indent, "dsdl_runtime.set_bit(out, offset_bits, bool(" + valueExpr + "))");
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Composite)
        {
            emitPyRuntimeSerializeCompositeValue(out, indent, field, valueExpr, ctx);
        }
        else if (field.kind == RuntimeFieldKind::Unsigned)
        {
            std::string scalarExpr = "int(" + valueExpr + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out,
                     indent,
                     "dsdl_runtime.write_unsigned(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating +
                         ")");
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Float)
        {
            std::string scalarExpr = "float(" + valueExpr + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "float(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out, indent, "dsdl_runtime.write_float(out, offset_bits, " + bits + ", " + scalarExpr + ")");
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else
        {
            std::string scalarExpr = "int(" + valueExpr + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out,
                     indent,
                     "dsdl_runtime.write_signed(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating +
                         ")");
            emitLine(out, indent, "offset_bits += " + bits);
        }
        return;
    }

    if (field.arrayKind == RuntimeArrayKind::Fixed)
    {
        const auto arrVar  = field.fieldName + "_arr";
        const auto itemVar = field.fieldName + "_item";
        const auto cap     = std::to_string(field.arrayCapacity);
        emitLine(out, indent, arrVar + " = " + valueExpr);
        emitLine(out, indent, "if len(" + arrVar + ") != " + cap + ":");
        emitLine(out,
                 indent + 1,
                 "raise ValueError(\"" +
                     codegen_diagnostic_text::fieldExpectsExactlyElements(field.fieldName, cap, false) + "\")");
        emitLine(out, indent, "for " + itemVar + " in " + arrVar + ":");
        if (field.kind == RuntimeFieldKind::Bool)
        {
            emitLine(out, indent + 1, "dsdl_runtime.set_bit(out, offset_bits, bool(" + itemVar + "))");
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Composite)
        {
            emitPyRuntimeSerializeCompositeValue(out, indent + 1, field, itemVar, ctx);
        }
        else if (field.kind == RuntimeFieldKind::Unsigned)
        {
            std::string scalarExpr = "int(" + itemVar + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out,
                     indent + 1,
                     "dsdl_runtime.write_unsigned(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating +
                         ")");
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Float)
        {
            std::string scalarExpr = "float(" + itemVar + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "float(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out, indent + 1, "dsdl_runtime.write_float(out, offset_bits, " + bits + ", " + scalarExpr + ")");
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else
        {
            std::string scalarExpr = "int(" + itemVar + ")";
            if (!helpers.serScalar.empty())
            {
                scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
            }
            emitLine(out,
                     indent + 1,
                     "dsdl_runtime.write_signed(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating +
                         ")");
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        return;
    }

    const auto arrVar    = field.fieldName + "_arr";
    const auto itemVar   = field.fieldName + "_item";
    const auto cap       = std::to_string(field.arrayCapacity);
    const auto prefixBit = std::to_string(field.arrayLengthPrefixBits);
    emitLine(out, indent, arrVar + " = " + valueExpr);
    if (!helpers.arrayValidate.empty())
    {
        emitLine(out, indent, "if not " + helpers.arrayValidate + "(len(" + arrVar + ")):");
    }
    else
    {
        emitLine(out, indent, "if len(" + arrVar + ") > " + cap + ":");
    }
    emitLine(out,
             indent + 1,
             "raise ValueError(\"" + codegen_diagnostic_text::fieldExceedsMaxLength(field.fieldName, cap, false) +
                 "\")");
    std::string prefixExpr = "len(" + arrVar + ")";
    if (!helpers.serArrayPrefix.empty())
    {
        prefixExpr = "int(" + helpers.serArrayPrefix + "(" + prefixExpr + "))";
    }
    emitLine(out,
             indent,
             "dsdl_runtime.write_unsigned(out, offset_bits, " + prefixBit + ", " + prefixExpr + ", False)");
    emitLine(out, indent, "offset_bits += " + prefixBit);
    emitLine(out, indent, "for " + itemVar + " in " + arrVar + ":");
    if (field.kind == RuntimeFieldKind::Bool)
    {
        emitLine(out, indent + 1, "dsdl_runtime.set_bit(out, offset_bits, bool(" + itemVar + "))");
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else if (field.kind == RuntimeFieldKind::Composite)
    {
        emitPyRuntimeSerializeCompositeValue(out, indent + 1, field, itemVar, ctx);
    }
    else if (field.kind == RuntimeFieldKind::Unsigned)
    {
        std::string scalarExpr = "int(" + itemVar + ")";
        if (!helpers.serScalar.empty())
        {
            scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
        }
        emitLine(out,
                 indent + 1,
                 "dsdl_runtime.write_unsigned(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating + ")");
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else if (field.kind == RuntimeFieldKind::Float)
    {
        std::string scalarExpr = "float(" + itemVar + ")";
        if (!helpers.serScalar.empty())
        {
            scalarExpr = "float(" + helpers.serScalar + "(" + scalarExpr + "))";
        }
        emitLine(out, indent + 1, "dsdl_runtime.write_float(out, offset_bits, " + bits + ", " + scalarExpr + ")");
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else
    {
        std::string scalarExpr = "int(" + itemVar + ")";
        if (!helpers.serScalar.empty())
        {
            scalarExpr = "int(" + helpers.serScalar + "(" + scalarExpr + "))";
        }
        emitLine(out,
                 indent + 1,
                 "dsdl_runtime.write_signed(out, offset_bits, " + bits + ", " + scalarExpr + ", " + saturating + ")");
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
}

void emitPyRuntimeDeserializeFieldValue(std::ostringstream&            out,
                                        int                            indent,
                                        const RuntimeFieldPlan&        field,
                                        const std::string&             targetExpr,
                                        const EmitterContext&          ctx,
                                        const RuntimeFieldHelperNames& helpers)
{
    const auto bits = std::to_string(field.bitLength);

    emitPyRuntimeAlignDeserialize(out, indent, field.alignmentBits);
    if (field.arrayKind == RuntimeArrayKind::None)
    {
        if (field.kind == RuntimeFieldKind::Padding)
        {
            emitPyRuntimeDeserializePadding(out, indent, field);
        }
        else if (field.kind == RuntimeFieldKind::Bool)
        {
            emitLine(out, indent, targetExpr + " = dsdl_runtime.get_bit(data, offset_bits)");
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Composite)
        {
            emitPyRuntimeDeserializeCompositeValue(out, indent, field, targetExpr, ctx, helpers.delimiterValidate);
        }
        else if (field.kind == RuntimeFieldKind::Unsigned)
        {
            const auto rawVar = field.fieldName + "_raw";
            emitLine(out, indent, rawVar + " = dsdl_runtime.read_unsigned(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent, targetExpr + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent, targetExpr + " = " + rawVar);
            }
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Float)
        {
            const auto rawVar = field.fieldName + "_raw";
            emitLine(out, indent, rawVar + " = dsdl_runtime.read_float(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent, targetExpr + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent, targetExpr + " = " + rawVar);
            }
            emitLine(out, indent, "offset_bits += " + bits);
        }
        else
        {
            const auto rawVar = field.fieldName + "_raw";
            emitLine(out, indent, rawVar + " = dsdl_runtime.read_signed(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent, targetExpr + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent, targetExpr + " = " + rawVar);
            }
            emitLine(out, indent, "offset_bits += " + bits);
        }
        return;
    }

    if (field.arrayKind == RuntimeArrayKind::Fixed)
    {
        const auto arrVar  = field.fieldName + "_arr";
        const auto itemVar = field.fieldName + "_item";
        const auto cap     = std::to_string(field.arrayCapacity);
        emitLine(out, indent, arrVar + " = []");
        emitLine(out, indent, "for _ in range(" + cap + "):");
        if (field.kind == RuntimeFieldKind::Bool)
        {
            emitLine(out, indent + 1, itemVar + " = dsdl_runtime.get_bit(data, offset_bits)");
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Composite)
        {
            emitLine(out, indent + 1, itemVar + " = None");
            emitPyRuntimeDeserializeCompositeValue(out, indent + 1, field, itemVar, ctx, helpers.delimiterValidate);
        }
        else if (field.kind == RuntimeFieldKind::Unsigned)
        {
            const auto rawVar = field.fieldName + "_item_raw";
            emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_unsigned(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent + 1, itemVar + " = " + rawVar);
            }
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else if (field.kind == RuntimeFieldKind::Float)
        {
            const auto rawVar = field.fieldName + "_item_raw";
            emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_float(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent + 1, itemVar + " = " + rawVar);
            }
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        else
        {
            const auto rawVar = field.fieldName + "_item_raw";
            emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_signed(data, offset_bits, " + bits + ")");
            if (!helpers.deserScalar.empty())
            {
                emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
            }
            else
            {
                emitLine(out, indent + 1, itemVar + " = " + rawVar);
            }
            emitLine(out, indent + 1, "offset_bits += " + bits);
        }
        emitLine(out, indent + 1, arrVar + ".append(" + itemVar + ")");
        emitLine(out, indent, targetExpr + " = " + arrVar);
        return;
    }

    const auto arrVar     = field.fieldName + "_arr";
    const auto lenVar     = field.fieldName + "_len";
    const auto itemVar    = field.fieldName + "_item";
    const auto cap        = std::to_string(field.arrayCapacity);
    const auto prefixBits = std::to_string(field.arrayLengthPrefixBits);
    const auto rawLenVar  = field.fieldName + "_len_raw";
    emitLine(out, indent, rawLenVar + " = int(dsdl_runtime.read_unsigned(data, offset_bits, " + prefixBits + "))");
    emitLine(out, indent, "offset_bits += " + prefixBits);
    if (!helpers.deserArrayPrefix.empty())
    {
        emitLine(out, indent, lenVar + " = int(" + helpers.deserArrayPrefix + "(" + rawLenVar + "))");
    }
    else
    {
        emitLine(out, indent, lenVar + " = " + rawLenVar);
    }
    if (!helpers.arrayValidate.empty())
    {
        emitLine(out, indent, "if not " + helpers.arrayValidate + "(" + lenVar + "):");
    }
    else
    {
        emitLine(out, indent, "if " + lenVar + " < 0 or " + lenVar + " > " + cap + ":");
    }
    emitLine(out,
             indent + 1,
             "raise ValueError(\"" +
                 codegen_diagnostic_text::decodedLengthExceedsMaxLength(field.fieldName, cap, false) + "\")");
    emitLine(out, indent, arrVar + " = []");
    emitLine(out, indent, "for _ in range(" + lenVar + "):");
    if (field.kind == RuntimeFieldKind::Bool)
    {
        emitLine(out, indent + 1, itemVar + " = dsdl_runtime.get_bit(data, offset_bits)");
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else if (field.kind == RuntimeFieldKind::Composite)
    {
        emitLine(out, indent + 1, itemVar + " = None");
        emitPyRuntimeDeserializeCompositeValue(out, indent + 1, field, itemVar, ctx, helpers.delimiterValidate);
    }
    else if (field.kind == RuntimeFieldKind::Unsigned)
    {
        const auto rawVar = field.fieldName + "_item_raw";
        emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_unsigned(data, offset_bits, " + bits + ")");
        if (!helpers.deserScalar.empty())
        {
            emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
        }
        else
        {
            emitLine(out, indent + 1, itemVar + " = " + rawVar);
        }
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else if (field.kind == RuntimeFieldKind::Float)
    {
        const auto rawVar = field.fieldName + "_item_raw";
        emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_float(data, offset_bits, " + bits + ")");
        if (!helpers.deserScalar.empty())
        {
            emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
        }
        else
        {
            emitLine(out, indent + 1, itemVar + " = " + rawVar);
        }
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    else
    {
        const auto rawVar = field.fieldName + "_item_raw";
        emitLine(out, indent + 1, rawVar + " = dsdl_runtime.read_signed(data, offset_bits, " + bits + ")");
        if (!helpers.deserScalar.empty())
        {
            emitLine(out, indent + 1, itemVar + " = " + helpers.deserScalar + "(" + rawVar + ")");
        }
        else
        {
            emitLine(out, indent + 1, itemVar + " = " + rawVar);
        }
        emitLine(out, indent + 1, "offset_bits += " + bits);
    }
    emitLine(out, indent + 1, arrVar + ".append(" + itemVar + ")");
    emitLine(out, indent, targetExpr + " = " + arrVar);
}

void emitPyRuntimeFunctions(std::ostringstream&        out,
                            const std::string&         typeName,
                            const RuntimeSectionPlan&  plan,
                            const EmitterContext&      ctx,
                            const SemanticSection&     section,
                            const LoweredSectionFacts* sectionFacts)
{
    const auto serializeFn   = runtimeSerializeFn(typeName);
    const auto deserializeFn = runtimeDeserializeFn(typeName);
    const auto maxByteLength = (plan.maxBits + 7) / 8;
    const auto serializeHelpers =
        buildSectionHelperBindingPlan(section, sectionFacts, HelperBindingDirection::Serialize);
    const auto deserializeHelpers =
        buildSectionHelperBindingPlan(section, sectionFacts, HelperBindingDirection::Deserialize);

    const RuntimeHelperNameResolver helperNameResolver = [](const std::string& symbol) {
        return helperBindingNamePy(symbol);
    };
    const auto  bodyPlan           = buildScriptedSectionBodyPlan(section, plan, sectionFacts, helperNameResolver);
    const auto& sectionHelperNames = bodyPlan.sectionHelpers;

    const auto emitSerializeHelperBindings = [&]() {
        const auto lines = renderSectionHelperBindings(serializeHelpers,
                                                       HelperBindingRenderLanguage::Python,
                                                       ScalarBindingRenderDirection::Serialize,
                                                       helperNameResolver,
                                                       /*emitCapacityCheck=*/true);
        for (const auto& line : lines)
        {
            emitLine(out, 1, line);
        }
        if (!lines.empty())
        {
            emitLine(out, 1, "");
        }
    };

    const auto emitDeserializeHelperBindings = [&]() {
        const auto lines = renderSectionHelperBindings(deserializeHelpers,
                                                       HelperBindingRenderLanguage::Python,
                                                       ScalarBindingRenderDirection::Deserialize,
                                                       helperNameResolver,
                                                       /*emitCapacityCheck=*/false);
        for (const auto& line : lines)
        {
            emitLine(out, 1, line);
        }
        if (!lines.empty())
        {
            emitLine(out, 1, "");
        }
    };

    if (plan.isUnion)
    {
        emitLine(out, 0, "def " + serializeFn + "(value: " + typeName + ") -> bytes:");
        emitLine(out, 1, "out = bytearray(" + std::to_string(maxByteLength) + ")");
        emitLine(out, 1, "offset_bits = 0");
        emitSerializeHelperBindings();
        if (!sectionHelperNames.capacityCheck.empty())
        {
            emitLine(out, 1, "if not " + sectionHelperNames.capacityCheck + "(len(out) * 8):");
            emitLine(out, 2, "raise ValueError(\"" + codegen_diagnostic_text::serializationBufferTooSmall() + "\")");
        }
        emitLine(out, 1, "tag = int(value._tag)");
        if (!sectionHelperNames.unionTagValidate.empty())
        {
            emitLine(out, 1, "if not " + sectionHelperNames.unionTagValidate + "(tag):");
            emitLine(out, 2, "raise ValueError(f\"" + codegen_diagnostic_text::invalidUnionTagPrefix() + "{tag}\")");
        }
        if (!sectionHelperNames.serUnionTagMask.empty())
        {
            emitLine(out, 1, "tag = int(" + sectionHelperNames.serUnionTagMask + "(tag))");
        }
        emitLine(out,
                 1,
                 "dsdl_runtime.write_unsigned(out, offset_bits, " + std::to_string(plan.unionTagBits) +
                     ", tag, False)");
        emitLine(out, 1, "offset_bits += " + std::to_string(plan.unionTagBits));

        bool first = true;
        for (const auto& scriptedField : bodyPlan.fields)
        {
            const auto& field     = scriptedField.field;
            const auto& helpers   = scriptedField.helpers;
            const auto  optionTag = std::to_string(field.unionOptionIndex);
            emitLine(out, 1, std::string(first ? "if " : "elif ") + "tag == " + optionTag + ":");
            const auto optionValueExpr = "value." + field.fieldName;
            emitLine(out, 2, "if " + optionValueExpr + " is None:");
            emitLine(out,
                     3,
                     "raise ValueError(\"" +
                         codegen_diagnostic_text::unionFieldMissingForTag(field.fieldName, optionTag) + "\")");
            emitPyRuntimeSerializeFieldValue(out, 2, field, optionValueExpr, ctx, helpers);
            first = false;
        }
        emitLine(out, 1, "else:");
        emitLine(out, 2, "raise ValueError(f\"" + codegen_diagnostic_text::invalidUnionTagPrefix() + "{tag}\")");
        emitLine(out, 1, "aligned_offset_bits = dsdl_runtime.byte_length_for_bits(offset_bits) * 8");
        emitLine(out, 1, "for bit in range(offset_bits, aligned_offset_bits):");
        emitLine(out, 2, "dsdl_runtime.set_bit(out, bit, False)");
        emitLine(out, 1, "offset_bits = aligned_offset_bits");
        emitLine(out, 1, "used_bytes = dsdl_runtime.byte_length_for_bits(offset_bits)");
        emitLine(out, 1, "return bytes(out[:used_bytes])");
        emitLine(out, 0, "");

        emitLine(out,
                 0,
                 "def " + deserializeFn + "(data: bytes | bytearray | memoryview) -> tuple[" + typeName + ", int]:");
        emitLine(out, 1, "data = bytes(data)");
        emitDeserializeHelperBindings();
        emitLine(out, 1, "offset_bits = 0");
        emitLine(out,
                 1,
                 "tag = int(dsdl_runtime.read_unsigned(data, offset_bits, " + std::to_string(plan.unionTagBits) + "))");
        emitLine(out, 1, "offset_bits += " + std::to_string(plan.unionTagBits));
        if (!sectionHelperNames.deserUnionTagMask.empty())
        {
            emitLine(out, 1, "tag = int(" + sectionHelperNames.deserUnionTagMask + "(tag))");
        }
        if (!sectionHelperNames.unionTagValidate.empty())
        {
            emitLine(out, 1, "if not " + sectionHelperNames.unionTagValidate + "(tag):");
            emitLine(out,
                     2,
                     "raise ValueError(f\"" + codegen_diagnostic_text::decodedInvalidUnionTagPrefix() + "{tag}\")");
        }
        emitLine(out, 1, "value = " + typeName + "(_tag=tag)");

        first = true;
        for (const auto& scriptedField : bodyPlan.fields)
        {
            const auto& field     = scriptedField.field;
            const auto& helpers   = scriptedField.helpers;
            const auto  optionTag = std::to_string(field.unionOptionIndex);
            emitLine(out, 1, std::string(first ? "if " : "elif ") + "tag == " + optionTag + ":");
            emitPyRuntimeDeserializeFieldValue(out, 2, field, "value." + field.fieldName, ctx, helpers);
            first = false;
        }
        emitLine(out, 1, "else:");
        emitLine(out, 2, "raise ValueError(f\"" + codegen_diagnostic_text::decodedInvalidUnionTagPrefix() + "{tag}\")");
        emitLine(out, 1, "offset_bits = dsdl_runtime.byte_length_for_bits(offset_bits) * 8");
        emitLine(out, 1, "consumed = min(len(data), dsdl_runtime.byte_length_for_bits(offset_bits))");
        emitLine(out, 1, "return value, consumed");
        return;
    }

    emitLine(out, 0, "def " + serializeFn + "(value: " + typeName + ") -> bytes:");
    emitLine(out, 1, "out = bytearray(" + std::to_string(maxByteLength) + ")");
    emitLine(out, 1, "offset_bits = 0");
    emitSerializeHelperBindings();
    if (!sectionHelperNames.capacityCheck.empty())
    {
        emitLine(out, 1, "if not " + sectionHelperNames.capacityCheck + "(len(out) * 8):");
        emitLine(out, 2, "raise ValueError(\"" + codegen_diagnostic_text::serializationBufferTooSmall() + "\")");
    }
    for (const auto& scriptedField : bodyPlan.fields)
    {
        const auto& field   = scriptedField.field;
        const auto& helpers = scriptedField.helpers;
        emitPyRuntimeSerializeFieldValue(out, 1, field, "value." + field.fieldName, ctx, helpers);
    }
    emitLine(out, 1, "used_bytes = dsdl_runtime.byte_length_for_bits(offset_bits)");
    emitLine(out, 1, "return bytes(out[:used_bytes])");
    emitLine(out, 0, "");

    emitLine(out,
             0,
             "def " + deserializeFn + "(data: bytes | bytearray | memoryview) -> tuple[" + typeName + ", int]:");
    emitLine(out, 1, "data = bytes(data)");
    emitDeserializeHelperBindings();
    emitLine(out, 1, "value = " + typeName + "()");
    emitLine(out, 1, "offset_bits = 0");
    for (const auto& scriptedField : bodyPlan.fields)
    {
        const auto& field   = scriptedField.field;
        const auto& helpers = scriptedField.helpers;
        emitPyRuntimeDeserializeFieldValue(out, 1, field, "value." + field.fieldName, ctx, helpers);
    }
    emitLine(out, 1, "consumed = min(len(data), dsdl_runtime.byte_length_for_bits(offset_bits))");
    emitLine(out, 1, "return value, consumed");
}

const LoweredSectionFacts* findLoweredSectionFacts(const LoweredFactsMap&    loweredFacts,
                                                   const SemanticDefinition& def,
                                                   llvm::StringRef           sectionKey)
{
    const auto defIt =
        loweredFacts.find(loweredTypeKey(def.info.fullName, def.info.majorVersion, def.info.minorVersion));
    if (defIt == loweredFacts.end())
    {
        return nullptr;
    }
    const auto sectionIt = defIt->second.find(sectionKey.str());
    if (sectionIt == defIt->second.end())
    {
        return nullptr;
    }
    return &sectionIt->second;
}

llvm::Expected<std::string> renderDefinitionFile(const SemanticDefinition& def,
                                                 const EmitterContext&     ctx,
                                                 const LoweredFactsMap&    loweredFacts)
{
    std::ostringstream out;
    emitLine(out, 0, "# Generated by llvmdsdl (Python backend).");
    emitLine(out, 0, "from __future__ import annotations");
    emitLine(out, 0, "");
    emitLine(out, 0, "from dataclasses import dataclass, field");
    emitLine(out, 0, "");

    const llvm::StringRef      requestSectionKey   = def.isService ? "request" : "";
    const LoweredSectionFacts* requestSectionFacts = findLoweredSectionFacts(loweredFacts, def, requestSectionKey);
    auto                       requestRuntimePlan  = buildRuntimeSectionPlan(def.request, requestSectionFacts);
    if (!requestRuntimePlan)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "failed to build Python request runtime plan for '%s': %s",
                                       def.info.fullName.c_str(),
                                       llvm::toString(requestRuntimePlan.takeError()).c_str());
    }

    std::optional<RuntimeSectionPlan> responseRuntimePlanStorage;
    const RuntimeSectionPlan*         responseRuntimePlan = nullptr;
    if (def.response)
    {
        const auto* const responseSectionFacts = findLoweredSectionFacts(loweredFacts, def, "response");
        auto              responsePlanOrErr    = buildRuntimeSectionPlan(*def.response, responseSectionFacts);
        if (!responsePlanOrErr)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "failed to build Python response runtime plan for '%s': %s",
                                           def.info.fullName.c_str(),
                                           llvm::toString(responsePlanOrErr.takeError()).c_str());
        }
        responseRuntimePlanStorage = std::move(*responsePlanOrErr);
        responseRuntimePlan        = &(*responseRuntimePlanStorage);
    }

    std::map<std::string, std::set<std::string>> importsByModule;
    for (const auto& imp : collectCompositeImports(def.request, def.info, ctx))
    {
        importsByModule[imp.modulePath].insert(imp.typeName);
    }
    if (def.response)
    {
        for (const auto& imp : collectCompositeImports(*def.response, def.info, ctx))
        {
            importsByModule[imp.modulePath].insert(imp.typeName);
        }
    }

    emitLine(out, 0, "from " + ctx.packageName() + "._runtime_loader import runtime as dsdl_runtime");
    for (const auto& [modulePath, names] : importsByModule)
    {
        std::string importNames;
        for (const auto& name : names)
        {
            if (!importNames.empty())
            {
                importNames += ", ";
            }
            importNames += name;
        }
        emitLine(out, 0, "from " + modulePath + " import " + importNames);
    }

    emitLine(out, 0, "");
    const auto baseType = ctx.typeName(def.info);
    emitLine(out, 0, "DSDL_FULL_NAME = \"" + def.info.fullName + "\"");
    emitLine(out, 0, "DSDL_VERSION_MAJOR = " + std::to_string(def.info.majorVersion));
    emitLine(out, 0, "DSDL_VERSION_MINOR = " + std::to_string(def.info.minorVersion));
    emitLine(out, 0, "");

    if (!def.isService)
    {
        emitSectionType(out, baseType, def.request, ctx);
        emitLine(out, 0, "");
        emitSectionConstants(out, baseType, def.request);
        if (!def.request.constants.empty())
        {
            emitLine(out, 0, "");
        }
        emitPyRuntimeFunctions(out, baseType, *requestRuntimePlan, ctx, def.request, requestSectionFacts);
        return out.str();
    }

    const auto reqType  = baseType + "_Request";
    const auto respType = baseType + "_Response";

    emitSectionType(out, reqType, def.request, ctx);
    emitLine(out, 0, "");
    emitSectionConstants(out, reqType, def.request);
    emitLine(out, 0, "");
    emitPyRuntimeFunctions(out, reqType, *requestRuntimePlan, ctx, def.request, requestSectionFacts);
    emitLine(out, 0, "");

    if (def.response)
    {
        emitSectionType(out, respType, *def.response, ctx);
        emitLine(out, 0, "");
        emitSectionConstants(out, respType, *def.response);
        emitLine(out, 0, "");
        emitPyRuntimeFunctions(out,
                               respType,
                               *responseRuntimePlan,
                               ctx,
                               *def.response,
                               findLoweredSectionFacts(loweredFacts, def, "response"));
        emitLine(out, 0, "");
    }

    emitLine(out, 0, baseType + " = " + reqType);
    return out.str();
}

llvm::Expected<std::string> loadRuntimeFile(const std::string& fileName)
{
    const std::filesystem::path runtimePath =
        std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "python" / fileName;
    std::ifstream in(runtimePath.string());
    if (!in)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "failed to read Python runtime file '%s'",
                                       fileName.c_str());
    }
    std::ostringstream content;
    content << in.rdbuf();
    return content.str();
}

/// @brief Returns the runtime scaffold file name for a Python specialization.
/// @param[in] runtimeSpecialization Runtime specialization profile.
/// @return Source runtime scaffold file name under `runtime/python`.
std::string runtimeFileForSpecialization(const PythonRuntimeSpecialization runtimeSpecialization)
{
    if (runtimeSpecialization == PythonRuntimeSpecialization::Fast)
    {
        return "_dsdl_runtime_fast.py";
    }
    return "_dsdl_runtime.py";
}

/// @brief Renders backend metadata emitted with generated Python artifacts.
/// @param[in] options Python emission options.
/// @return Stable JSON metadata payload.
std::string renderPackageMetadata(const PythonEmitOptions& options)
{
    std::ostringstream out;
    out << "{\n";
    out << "  \"llvmdsdl\": {\n";
    out << "    \"pythonRuntimeSpecialization\": \"";
    out << (options.runtimeSpecialization == PythonRuntimeSpecialization::Fast ? "fast" : "portable");
    out << "\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

/// @brief Renders `pyproject.toml` metadata for generated Python packages.
/// @param[in] packageName Generated package name.
/// @param[in] rootPackageName Top-level package component used for discovery.
/// @return Minimal installable `pyproject.toml` payload.
std::string renderPyProjectToml(llvm::StringRef packageName, llvm::StringRef rootPackageName)
{
    std::ostringstream out;
    out << "[build-system]\n";
    out << "requires = [\"setuptools>=61\"]\n";
    out << "build-backend = \"setuptools.build_meta\"\n\n";

    out << "[project]\n";
    out << "name = \"" << packageName.str() << "\"\n";
    out << "version = \"0.1.0\"\n";
    out << "description = \"Generated by llvmdsdl\"\n";
    out << "requires-python = \">=3.10\"\n\n";

    out << "[tool.setuptools]\n";
    out << "include-package-data = true\n\n";

    out << "[tool.setuptools.packages.find]\n";
    out << "where = [\".\"]\n";
    out << "include = [\"" << rootPackageName.str() << "*\"]\n\n";

    out << "[tool.setuptools.package-data]\n";
    out << "\"" << packageName.str() << "\" = [\"py.typed\", \"_dsdl_runtime_accel*.so\", "
        << "\"_dsdl_runtime_accel*.dylib\", \"_dsdl_runtime_accel*.pyd\"]\n";
    return out.str();
}

llvm::Error ensureInitFile(const std::filesystem::path& dir, std::set<std::filesystem::path>& initializedPackages)
{
    if (!initializedPackages.insert(dir).second)
    {
        return llvm::Error::success();
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        return llvm::createStringError(ec, "failed to create package directory %s", dir.string().c_str());
    }
    std::filesystem::path initPath = dir / "__init__.py";
    if (!std::filesystem::exists(initPath, ec))
    {
        if (auto err = writeFile(initPath, "# Generated by llvmdsdl (Python backend).\n"))
        {
            return err;
        }
    }
    return llvm::Error::success();
}

llvm::Error ensurePackageInitChain(const std::filesystem::path&     packageRoot,
                                   const std::filesystem::path&     relDir,
                                   std::set<std::filesystem::path>& initializedPackages)
{
    std::filesystem::path current = packageRoot;
    if (auto err = ensureInitFile(current, initializedPackages))
    {
        return err;
    }

    for (const auto& part : relDir)
    {
        const auto piece = part.string();
        if (piece.empty() || piece == ".")
        {
            continue;
        }
        current /= piece;
        if (auto err = ensureInitFile(current, initializedPackages))
        {
            return err;
        }
    }

    return llvm::Error::success();
}

}  // namespace

llvm::Error emitPython(const SemanticModule&    semantic,
                       mlir::ModuleOp           module,
                       const PythonEmitOptions& options,
                       DiagnosticEngine&        diagnostics)
{
    if (options.outDir.empty())
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(), "output directory is required");
    }

    LoweredFactsMap loweredFacts;
    if (!collectLoweredFactsFromMlir(semantic,
                                     module,
                                     diagnostics,
                                     "Python",
                                     &loweredFacts,
                                     options.optimizeLoweredSerDes))
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "MLIR schema coverage validation failed for Python emission");
    }

    const auto     packageComponents = splitPackageName(options.packageName);
    EmitterContext ctx(semantic, packageComponents);

    std::filesystem::path outRoot(options.outDir);
    std::filesystem::create_directories(outRoot);

    const std::filesystem::path packageRoot = outRoot / ctx.packageRootPath();
    std::filesystem::create_directories(packageRoot);

    std::set<std::filesystem::path> initializedPackages;
    if (auto err = ensurePackageInitChain(packageRoot, std::filesystem::path{}, initializedPackages))
    {
        return err;
    }

    auto runtime = loadRuntimeFile(runtimeFileForSpecialization(options.runtimeSpecialization));
    if (!runtime)
    {
        return runtime.takeError();
    }
    if (auto err = writeFile(packageRoot / "_dsdl_runtime.py", *runtime))
    {
        return err;
    }

    auto runtimeLoader = loadRuntimeFile("_runtime_loader.py");
    if (!runtimeLoader)
    {
        return runtimeLoader.takeError();
    }
    if (auto err = writeFile(packageRoot / "_runtime_loader.py", *runtimeLoader))
    {
        return err;
    }

    if (auto err = writeFile(packageRoot / "llvmdsdl_codegen.json", renderPackageMetadata(options)))
    {
        return err;
    }

    if (auto err =
            writeFile(outRoot / "pyproject.toml", renderPyProjectToml(ctx.packageName(), packageComponents.front())))
    {
        return err;
    }

    if (auto err = writeFile(packageRoot / "py.typed", ""))
    {
        return err;
    }

    std::vector<const SemanticDefinition*> ordered;
    ordered.reserve(semantic.definitions.size());
    for (const auto& def : semantic.definitions)
    {
        ordered.push_back(&def);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto* lhs, const auto* rhs) {
        if (lhs->info.fullName != rhs->info.fullName)
        {
            return lhs->info.fullName < rhs->info.fullName;
        }
        if (lhs->info.majorVersion != rhs->info.majorVersion)
        {
            return lhs->info.majorVersion < rhs->info.majorVersion;
        }
        return lhs->info.minorVersion < rhs->info.minorVersion;
    });

    for (const auto* def : ordered)
    {
        const auto relPath  = ctx.relativeFilePath(def->info);
        const auto fullPath = packageRoot / relPath;

        if (auto err = ensurePackageInitChain(packageRoot, relPath.parent_path(), initializedPackages))
        {
            return err;
        }

        std::filesystem::create_directories(fullPath.parent_path());
        auto rendered = renderDefinitionFile(*def, ctx, loweredFacts);
        if (!rendered)
        {
            return rendered.takeError();
        }

        if (auto err = writeFile(fullPath, *rendered))
        {
            return err;
        }
    }

    return llvm::Error::success();
}

}  // namespace llvmdsdl
