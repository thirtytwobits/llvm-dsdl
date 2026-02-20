//===----------------------------------------------------------------------===//
///
/// @file
/// Implements TypeScript backend code emission from lowered DSDL modules.
///
/// This file emits TypeScript models, codec entry points, and runtime wiring from lowering contracts.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/TsEmitter.h"

#include <llvm/ADT/StringRef.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <utility>
#include <variant>

#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"
#include "llvmdsdl/CodeGen/TsLoweredPlan.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Evaluator.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Rational.h"
#include "mlir/IR/BuiltinOps.h"

namespace llvmdsdl
{
class DiagnosticEngine;

namespace
{

bool isTsKeyword(const std::string& name)
{
    static const std::set<std::string> kKeywords =
        {"break", "case",       "catch",     "class",      "const",   "continue", "debugger",  "default", "delete",
         "do",    "else",       "enum",      "export",     "extends", "false",    "finally",   "for",     "function",
         "if",    "import",     "in",        "instanceof", "new",     "null",     "return",    "super",   "switch",
         "this",  "throw",      "true",      "try",        "typeof",  "var",      "void",      "while",   "with",
         "as",    "implements", "interface", "let",        "package", "private",  "protected", "public",  "static",
         "yield", "any",        "boolean",   "number",     "string",  "symbol",   "type",      "from",    "of"};
    return kKeywords.contains(name);
}

std::string sanitizeTsIdent(std::string name)
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
    if (isTsKeyword(name))
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
    return sanitizeTsIdent(out);
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
    return sanitizeTsIdent(out);
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

std::string tsConstValue(const Value& value)
{
    if (const auto* b = std::get_if<bool>(&value.data))
    {
        return *b ? "true" : "false";
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
    out << std::string(static_cast<std::size_t>(indent) * 2U, ' ') << line << '\n';
}

class EmitterContext final
{
public:
    explicit EmitterContext(const SemanticModule& semantic)
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
        rel /= fileStem(info) + ".ts";
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
               std::to_string(ref.minorVersion) + ".ts";
        return rel;
    }

private:
    std::unordered_map<std::string, const SemanticDefinition*> byKey_;
};

std::string tsFieldBaseType(const SemanticFieldType& type, const EmitterContext& ctx)
{
    switch (type.scalarCategory)
    {
    case SemanticScalarCategory::Bool:
        return "boolean";
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
        return "number";
    case SemanticScalarCategory::UnsignedInt:
    case SemanticScalarCategory::SignedInt:
        if (type.bitLength > 53U)
        {
            return "bigint";
        }
        return "number";
    case SemanticScalarCategory::Float:
    case SemanticScalarCategory::Void:
        return "number";
    case SemanticScalarCategory::Composite:
        if (type.compositeType)
        {
            return ctx.typeName(*type.compositeType);
        }
        return "unknown";
    }
    return "unknown";
}

std::string tsFieldType(const SemanticFieldType& type, const EmitterContext& ctx)
{
    const auto base = tsFieldBaseType(type, ctx);
    if (type.arrayKind == ArrayKind::None)
    {
        return base;
    }
    return "Array<" + base + ">";
}

std::string relativeImportPath(const std::filesystem::path& fromFile, const std::filesystem::path& toFile)
{
    const auto  fromDir    = fromFile.parent_path();
    auto        rel        = toFile.lexically_relative(fromDir);
    std::string importPath = rel.generic_string();
    if (importPath.size() >= 3 && importPath.ends_with(".ts"))
    {
        importPath.resize(importPath.size() - 3);
    }
    if (!importPath.empty() && importPath.front() != '.')
    {
        importPath = "./" + importPath;
    }
    return importPath;
}

std::string moduleAliasFromPath(const std::string& modulePath)
{
    std::string alias;
    alias.reserve(modulePath.size() + 8);
    for (char c : modulePath)
    {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
        {
            alias.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        else
        {
            alias.push_back('_');
        }
    }
    return sanitizeTsIdent(alias.empty() ? "module" : alias);
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
        const auto modulePath = relativeImportPath(ownerPath, targetPath);
        importsByModule[modulePath].insert(ctx.typeName(ref));
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
        emitLine(out, 0, "export const " + constName + " = " + tsConstValue(constant.value) + ";");
    }
}

void emitStructSectionType(std::ostringstream&    out,
                           const std::string&     typeName,
                           const SemanticSection& section,
                           const EmitterContext&  ctx)
{
    emitLine(out, 0, "export interface " + typeName + " {");
    for (const auto& field : section.fields)
    {
        if (field.isPadding)
        {
            continue;
        }
        const auto fieldName = sanitizeTsIdent(toSnakeCase(field.name));
        emitLine(out, 1, fieldName + ": " + tsFieldType(field.resolvedType, ctx) + ";");
    }
    emitLine(out, 0, "}");
}

void emitUnionSectionType(std::ostringstream&    out,
                          const std::string&     typeName,
                          const SemanticSection& section,
                          const EmitterContext&  ctx)
{
    std::vector<const SemanticField*> options;
    for (const auto& field : section.fields)
    {
        if (!field.isPadding)
        {
            options.push_back(&field);
        }
    }

    if (options.empty())
    {
        emitLine(out, 0, "export type " + typeName + " = { _tag: number };");
        return;
    }

    emitLine(out, 0, "export type " + typeName + " =");
    for (std::size_t i = 0; i < options.size(); ++i)
    {
        const auto*        field     = options[i];
        const auto         fieldName = sanitizeTsIdent(toSnakeCase(field->name));
        std::ostringstream variant;
        variant << "{ _tag: " << field->unionOptionIndex << "; " << fieldName << ": "
                << tsFieldType(field->resolvedType, ctx) << "; }";
        const auto prefix = i == 0 ? "  | " : "  | ";
        emitLine(out, 0, prefix + variant.str() + (i + 1 == options.size() ? ";" : ""));
    }
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

std::string tsRuntimeSerializeFn(const std::string& typeName)
{
    return "serialize" + typeName;
}

std::string tsRuntimeDeserializeFn(const std::string& typeName)
{
    return "deserialize" + typeName;
}

std::string compositeTypeName(const TsRuntimeFieldPlan& field, const EmitterContext& ctx)
{
    return field.compositeType ? ctx.typeName(*field.compositeType) : std::string{"unknown_composite"};
}

void emitTsRuntimeSerializeCompositeValue(std::ostringstream&       out,
                                          int                       indent,
                                          const TsRuntimeFieldPlan& field,
                                          const std::string&        valueExpr,
                                          const EmitterContext&     ctx)
{
    const auto nestedVar = field.fieldName + "Bytes";
    const auto typeName  = compositeTypeName(field, ctx);
    emitLine(out, indent, "const " + nestedVar + " = " + tsRuntimeSerializeFn(typeName) + "(" + valueExpr + ");");
    if (field.compositeSealed)
    {
        emitLine(out,
                 indent,
                 "dsdlRuntime.copyBits(out, offsetBits, " + nestedVar + ", 0, " + std::to_string(field.bitLength) +
                     ");");
        emitLine(out, indent, "offsetBits += " + std::to_string(field.bitLength) + ";");
        return;
    }

    const auto sizeVar         = field.fieldName + "SizeBytes";
    const auto remainingVar    = field.fieldName + "RemainingBytes";
    const auto maxPayloadBytes = std::to_string((field.compositePayloadMaxBits + 7) / 8);
    emitLine(out, indent, "const " + sizeVar + " = " + nestedVar + ".length;");
    emitLine(out, indent, "if (" + sizeVar + " > " + maxPayloadBytes + ") {");
    emitLine(out,
             indent + 1,
             "throw new Error(\"encoded payload for composite field '" + field.fieldName +
                 "' exceeds max payload bytes " + maxPayloadBytes + "\");");
    emitLine(out, indent, "}");
    emitLine(out,
             indent,
             "const " + remainingVar + " = out.length - Math.min(Math.trunc(offsetBits / 8), out.length);");
    emitLine(out, indent, "if (" + sizeVar + " > " + remainingVar + ") {");
    emitLine(out,
             indent + 1,
             "throw new Error(\"encoded payload for composite field '" + field.fieldName +
                 "' exceeds remaining buffer space\");");
    emitLine(out, indent, "}");
    emitLine(out, indent, "dsdlRuntime.writeUnsigned(out, offsetBits, 32, " + sizeVar + ", false);");
    emitLine(out, indent, "offsetBits += 32;");
    emitLine(out, indent, "dsdlRuntime.copyBits(out, offsetBits, " + nestedVar + ", 0, " + sizeVar + " * 8);");
    emitLine(out, indent, "offsetBits += " + sizeVar + " * 8;");
}

void emitTsRuntimeDeserializeCompositeValue(std::ostringstream&       out,
                                            int                       indent,
                                            const TsRuntimeFieldPlan& field,
                                            const std::string&        targetExpr,
                                            const EmitterContext&     ctx)
{
    const auto typeName = compositeTypeName(field, ctx);
    if (field.compositeSealed)
    {
        const auto nestedVar = field.fieldName + "Bytes";
        emitLine(out,
                 indent,
                 "const " + nestedVar + " = dsdlRuntime.extractBits(bytes, offsetBits, " +
                     std::to_string(field.bitLength) + ");");
        emitLine(out, indent, targetExpr + " = " + tsRuntimeDeserializeFn(typeName) + "(" + nestedVar + ").value;");
        emitLine(out, indent, "offsetBits += " + std::to_string(field.bitLength) + ";");
        return;
    }

    const auto sizeVar      = field.fieldName + "SizeBytes";
    const auto remainingVar = field.fieldName + "RemainingBytes";
    const auto startVar     = field.fieldName + "StartByte";
    const auto endVar       = field.fieldName + "EndByte";
    const auto nestedVar    = field.fieldName + "Bytes";
    emitLine(out, indent, "const " + sizeVar + " = Math.trunc(dsdlRuntime.readUnsigned(bytes, offsetBits, 32));");
    emitLine(out, indent, "offsetBits += 32;");
    emitLine(out,
             indent,
             "const " + remainingVar + " = bytes.length - Math.min(Math.trunc(offsetBits / 8), bytes.length);");
    emitLine(out, indent, "if (" + sizeVar + " < 0 || " + sizeVar + " > " + remainingVar + ") {");
    emitLine(out,
             indent + 1,
             "throw new Error(\"decoded payload size for composite field '" + field.fieldName +
                 "' exceeds remaining buffer space\");");
    emitLine(out, indent, "}");
    emitLine(out, indent, "const " + startVar + " = Math.min(Math.trunc(offsetBits / 8), bytes.length);");
    emitLine(out, indent, "const " + endVar + " = Math.min(" + startVar + " + " + sizeVar + ", bytes.length);");
    emitLine(out, indent, "const " + nestedVar + " = bytes.subarray(" + startVar + ", " + endVar + ");");
    emitLine(out, indent, targetExpr + " = " + tsRuntimeDeserializeFn(typeName) + "(" + nestedVar + ").value;");
    emitLine(out, indent, "offsetBits += " + sizeVar + " * 8;");
}

void emitTsRuntimeAlignSerialize(std::ostringstream& out,
                                 int                 indent,
                                 std::int64_t        alignmentBits,
                                 const std::string&  prefix)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    const auto alignedVar = prefix + "AlignedOffsetBits";
    const auto bitVar     = prefix + "AlignBit";
    emitLine(out,
             indent,
             "const " + alignedVar + " = Math.trunc((offsetBits + " + std::to_string(alignmentBits - 1) + ") / " +
                 std::to_string(alignmentBits) + ") * " + std::to_string(alignmentBits) + ";");
    emitLine(out,
             indent,
             "for (let " + bitVar + " = offsetBits; " + bitVar + " < " + alignedVar + "; ++" + bitVar + ") {");
    emitLine(out, indent + 1, "dsdlRuntime.setBit(out, " + bitVar + ", false);");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offsetBits = " + alignedVar + ";");
}

void emitTsRuntimeAlignDeserialize(std::ostringstream& out, int indent, std::int64_t alignmentBits)
{
    if (alignmentBits <= 1)
    {
        return;
    }
    emitLine(out,
             indent,
             "offsetBits = Math.trunc((offsetBits + " + std::to_string(alignmentBits - 1) + ") / " +
                 std::to_string(alignmentBits) + ") * " + std::to_string(alignmentBits) + ";");
}

void emitTsRuntimeSerializePadding(std::ostringstream&       out,
                                   int                       indent,
                                   const TsRuntimeFieldPlan& field,
                                   const std::string&        prefix)
{
    if (field.bitLength <= 0)
    {
        return;
    }
    const auto bitVar = prefix + "PaddingBit";
    emitLine(out,
             indent,
             "for (let " + bitVar + " = 0; " + bitVar + " < " + std::to_string(field.bitLength) + "; ++" + bitVar +
                 ") {");
    emitLine(out, indent + 1, "dsdlRuntime.setBit(out, offsetBits + " + bitVar + ", false);");
    emitLine(out, indent, "}");
    emitLine(out, indent, "offsetBits += " + std::to_string(field.bitLength) + ";");
}

void emitTsRuntimeDeserializePadding(std::ostringstream& out, int indent, const TsRuntimeFieldPlan& field)
{
    if (field.bitLength <= 0)
    {
        return;
    }
    emitLine(out, indent, "offsetBits += " + std::to_string(field.bitLength) + ";");
}

void emitTsRuntimeFunctions(std::ostringstream&         out,
                            const std::string&          typeName,
                            const TsRuntimeSectionPlan& plan,
                            const EmitterContext&       ctx)
{
    const auto serializeFn   = tsRuntimeSerializeFn(typeName);
    const auto deserializeFn = tsRuntimeDeserializeFn(typeName);
    const auto maxByteLength = (plan.maxBits + 7) / 8;

    if (plan.isUnion)
    {
        const auto tagBits = std::to_string(plan.unionTagBits);
        emitLine(out, 0, "export function " + serializeFn + "(value: " + typeName + "): Uint8Array {");
        emitLine(out, 1, "const out = new Uint8Array(" + std::to_string(maxByteLength) + ");");
        emitLine(out, 1, "let offsetBits = 0;");
        emitLine(out, 1, "const tag = Math.trunc((value as { _tag: number })._tag);");
        emitLine(out, 1, "dsdlRuntime.writeUnsigned(out, offsetBits, " + tagBits + ", tag, false);");
        emitLine(out, 1, "offsetBits += " + tagBits + ";");
        emitLine(out, 1, "switch (tag) {");
        for (const auto& field : plan.fields)
        {
            const auto bits       = std::to_string(field.bitLength);
            const auto optionTag  = std::to_string(field.unionOptionIndex);
            const auto saturating = field.castMode == CastMode::Saturated ? "true" : "false";
            const auto cap        = std::to_string(field.arrayCapacity);
            const auto prefixBits = std::to_string(field.arrayLengthPrefixBits);
            emitLine(out, 1, "case " + optionTag + ": {");
            emitLine(out, 2, "const optionValue = (value as Record<string, unknown>)." + field.fieldName + ";");
            emitLine(out, 2, "if (optionValue === undefined) {");
            emitLine(out,
                     3,
                     "throw new Error(\"union field '" + field.fieldName + "' missing for tag " + optionTag + "\");");
            emitLine(out, 2, "}");
            emitTsRuntimeAlignSerialize(out, 2, field.alignmentBits, field.fieldName + "Option");
            if (field.arrayKind == TsRuntimeArrayKind::None)
            {
                if (field.kind == TsRuntimeFieldKind::Padding)
                {
                    emitTsRuntimeSerializePadding(out, 2, field, field.fieldName + "Option");
                }
                else if (field.kind == TsRuntimeFieldKind::Bool)
                {
                    emitLine(out, 2, "dsdlRuntime.setBit(out, offsetBits, !!optionValue);");
                }
                else if (field.kind == TsRuntimeFieldKind::Composite)
                {
                    emitTsRuntimeSerializeCompositeValue(out,
                                                         2,
                                                         field,
                                                         "optionValue as " + compositeTypeName(field, ctx),
                                                         ctx);
                }
                else if (field.kind == TsRuntimeFieldKind::Unsigned)
                {
                    emitLine(out,
                             2,
                             "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits +
                                 ", optionValue as number | bigint, " + saturating + ");");
                }
                else if (field.kind == TsRuntimeFieldKind::Float)
                {
                    emitLine(out, 2, "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", optionValue as number);");
                }
                else
                {
                    emitLine(out,
                             2,
                             "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", optionValue as number | bigint, " +
                                 saturating + ");");
                }
                if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
                {
                    emitLine(out, 2, "offsetBits += " + bits + ";");
                }
            }
            else if (field.arrayKind == TsRuntimeArrayKind::Fixed)
            {
                const auto optionArray = field.fieldName + "Array";
                emitLine(out, 2, "const " + optionArray + " = optionValue;");
                emitLine(out,
                         2,
                         "if (!Array.isArray(" + optionArray + ") || " + optionArray + ".length !== " + cap + ") {");
                emitLine(out,
                         3,
                         "throw new Error(\"union field '" + field.fieldName + "' expects exactly " + cap +
                             " elements\");");
                emitLine(out, 2, "}");
                emitLine(out, 2, "for (let i = 0; i < " + cap + "; ++i) {");
                if (field.kind == TsRuntimeFieldKind::Bool)
                {
                    emitLine(out, 3, "dsdlRuntime.setBit(out, offsetBits, " + optionArray + "[i]);");
                }
                else if (field.kind == TsRuntimeFieldKind::Composite)
                {
                    emitTsRuntimeSerializeCompositeValue(out,
                                                         3,
                                                         field,
                                                         optionArray + "[i] as " + compositeTypeName(field, ctx),
                                                         ctx);
                }
                else if (field.kind == TsRuntimeFieldKind::Unsigned)
                {
                    emitLine(out,
                             3,
                             "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits + ", " + optionArray + "[i], " +
                                 saturating + ");");
                }
                else if (field.kind == TsRuntimeFieldKind::Float)
                {
                    emitLine(out, 3, "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", " + optionArray + "[i]);");
                }
                else
                {
                    emitLine(out,
                             3,
                             "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", " + optionArray + "[i], " +
                                 saturating + ");");
                }
                if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
                {
                    emitLine(out, 3, "offsetBits += " + bits + ";");
                }
                emitLine(out, 2, "}");
            }
            else
            {
                const auto optionArray = field.fieldName + "Array";
                emitLine(out, 2, "const " + optionArray + " = optionValue;");
                emitLine(out, 2, "if (!Array.isArray(" + optionArray + ")) {");
                emitLine(out, 3, "throw new Error(\"union field '" + field.fieldName + "' expects an array\");");
                emitLine(out, 2, "}");
                emitLine(out, 2, "if (" + optionArray + ".length > " + cap + ") {");
                emitLine(out,
                         3,
                         "throw new Error(\"union field '" + field.fieldName + "' exceeds max length " + cap + "\");");
                emitLine(out, 2, "}");
                emitLine(out,
                         2,
                         "dsdlRuntime.writeUnsigned(out, offsetBits, " + prefixBits + ", " + optionArray +
                             ".length, false);");
                emitLine(out, 2, "offsetBits += " + prefixBits + ";");
                emitLine(out, 2, "for (let i = 0; i < " + optionArray + ".length; ++i) {");
                if (field.kind == TsRuntimeFieldKind::Bool)
                {
                    emitLine(out, 3, "dsdlRuntime.setBit(out, offsetBits, " + optionArray + "[i]);");
                }
                else if (field.kind == TsRuntimeFieldKind::Composite)
                {
                    emitTsRuntimeSerializeCompositeValue(out,
                                                         3,
                                                         field,
                                                         optionArray + "[i] as " + compositeTypeName(field, ctx),
                                                         ctx);
                }
                else if (field.kind == TsRuntimeFieldKind::Unsigned)
                {
                    emitLine(out,
                             3,
                             "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits + ", " + optionArray + "[i], " +
                                 saturating + ");");
                }
                else if (field.kind == TsRuntimeFieldKind::Float)
                {
                    emitLine(out, 3, "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", " + optionArray + "[i]);");
                }
                else
                {
                    emitLine(out,
                             3,
                             "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", " + optionArray + "[i], " +
                                 saturating + ");");
                }
                if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
                {
                    emitLine(out, 3, "offsetBits += " + bits + ";");
                }
                emitLine(out, 2, "}");
            }
            emitLine(out, 2, "break;");
            emitLine(out, 1, "}");
        }
        emitLine(out, 1, "default:");
        emitLine(out, 2, "throw new Error(\"invalid union tag \" + tag);");
        emitLine(out, 1, "}");
        emitLine(out, 1, "const alignedOffsetBits = dsdlRuntime.byteLengthForBits(offsetBits) * 8;");
        emitLine(out, 1, "for (let bit = offsetBits; bit < alignedOffsetBits; ++bit) {");
        emitLine(out, 2, "dsdlRuntime.setBit(out, bit, false);");
        emitLine(out, 1, "}");
        emitLine(out, 1, "offsetBits = alignedOffsetBits;");
        emitLine(out, 1, "const usedBytes = dsdlRuntime.byteLengthForBits(offsetBits);");
        emitLine(out, 1, "return out.subarray(0, usedBytes);");
        emitLine(out, 0, "}");
        emitLine(out, 0, "");

        emitLine(out,
                 0,
                 "export function " + deserializeFn + "(bytes: Uint8Array): { value: " + typeName +
                     "; consumed: number } {");
        emitLine(out, 1, "let offsetBits = 0;");
        emitLine(out, 1, "const tag = Math.trunc(dsdlRuntime.readUnsigned(bytes, offsetBits, " + tagBits + "));");
        emitLine(out, 1, "offsetBits += " + tagBits + ";");
        emitLine(out, 1, "let value: " + typeName + ";");
        emitLine(out, 1, "switch (tag) {");
        for (const auto& field : plan.fields)
        {
            const auto bits       = std::to_string(field.bitLength);
            const auto optionTag  = std::to_string(field.unionOptionIndex);
            const auto cap        = std::to_string(field.arrayCapacity);
            const auto prefixBits = std::to_string(field.arrayLengthPrefixBits);
            const auto arrayElemType =
                field.kind == TsRuntimeFieldKind::Bool
                    ? "boolean"
                    : (field.kind == TsRuntimeFieldKind::Composite ? compositeTypeName(field, ctx)
                                                                   : (field.useBigInt ? "bigint" : "number"));
            emitLine(out, 1, "case " + optionTag + ": {");
            emitTsRuntimeAlignDeserialize(out, 2, field.alignmentBits);
            if (field.arrayKind == TsRuntimeArrayKind::None)
            {
                if (field.kind == TsRuntimeFieldKind::Padding)
                {
                    emitTsRuntimeDeserializePadding(out, 2, field);
                    emitLine(out, 2, "const optionValue = undefined;");
                }
                else if (field.kind == TsRuntimeFieldKind::Bool)
                {
                    emitLine(out, 2, "const optionValue = dsdlRuntime.getBit(bytes, offsetBits);");
                }
                else if (field.kind == TsRuntimeFieldKind::Composite)
                {
                    emitLine(out, 2, "let optionValue: " + compositeTypeName(field, ctx) + ";");
                    emitTsRuntimeDeserializeCompositeValue(out, 2, field, "optionValue", ctx);
                }
                else if (field.kind == TsRuntimeFieldKind::Unsigned)
                {
                    const std::string fn = field.useBigInt ? "readUnsignedBigInt" : "readUnsigned";
                    emitLine(out, 2, "const optionValue = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
                }
                else if (field.kind == TsRuntimeFieldKind::Float)
                {
                    emitLine(out, 2, "const optionValue = dsdlRuntime.readFloat(bytes, offsetBits, " + bits + ");");
                }
                else
                {
                    const std::string fn = field.useBigInt ? "readSignedBigInt" : "readSigned";
                    emitLine(out, 2, "const optionValue = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
                }
                if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
                {
                    emitLine(out, 2, "offsetBits += " + bits + ";");
                }
            }
            else
            {
                std::string arrayLenExpr = cap;
                if (field.arrayKind == TsRuntimeArrayKind::Variable)
                {
                    emitLine(out,
                             2,
                             "const " + field.fieldName +
                                 "Length = Math.trunc(dsdlRuntime.readUnsigned(bytes, offsetBits, " + prefixBits +
                                 "));");
                    emitLine(out, 2, "offsetBits += " + prefixBits + ";");
                    emitLine(out,
                             2,
                             "if (" + field.fieldName + "Length < 0 || " + field.fieldName + "Length > " + cap + ") {");
                    emitLine(out,
                             3,
                             "throw new Error(\"decoded length for union field '" + field.fieldName +
                                 "' exceeds max length " + cap + "\");");
                    emitLine(out, 2, "}");
                    arrayLenExpr = field.fieldName + "Length";
                }
                const auto optionArray = field.fieldName + "Array";
                emitLine(out,
                         2,
                         "const " + optionArray + ": Array<" + arrayElemType + "> = new Array(" + arrayLenExpr + ");");
                emitLine(out, 2, "for (let i = 0; i < " + arrayLenExpr + "; ++i) {");
                if (field.kind == TsRuntimeFieldKind::Bool)
                {
                    emitLine(out, 3, optionArray + "[i] = dsdlRuntime.getBit(bytes, offsetBits);");
                }
                else if (field.kind == TsRuntimeFieldKind::Composite)
                {
                    emitTsRuntimeDeserializeCompositeValue(out, 3, field, optionArray + "[i]", ctx);
                }
                else if (field.kind == TsRuntimeFieldKind::Unsigned)
                {
                    const std::string fn = field.useBigInt ? "readUnsignedBigInt" : "readUnsigned";
                    emitLine(out, 3, optionArray + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
                }
                else if (field.kind == TsRuntimeFieldKind::Float)
                {
                    emitLine(out, 3, optionArray + "[i] = dsdlRuntime.readFloat(bytes, offsetBits, " + bits + ");");
                }
                else
                {
                    const std::string fn = field.useBigInt ? "readSignedBigInt" : "readSigned";
                    emitLine(out, 3, optionArray + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
                }
                if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
                {
                    emitLine(out, 3, "offsetBits += " + bits + ";");
                }
                emitLine(out, 2, "}");
                emitLine(out, 2, "const optionValue = " + optionArray + ";");
            }
            emitLine(out,
                     2,
                     "value = { _tag: " + optionTag + ", " + field.fieldName + ": optionValue } as " + typeName + ";");
            emitLine(out, 2, "break;");
            emitLine(out, 1, "}");
        }
        emitLine(out, 1, "default:");
        emitLine(out, 2, "throw new Error(\"decoded invalid union tag \" + tag);");
        emitLine(out, 1, "}");
        emitLine(out, 1, "offsetBits = dsdlRuntime.byteLengthForBits(offsetBits) * 8;");
        emitLine(out, 1, "const consumed = Math.min(bytes.length, dsdlRuntime.byteLengthForBits(offsetBits));");
        emitLine(out, 1, "return { value, consumed };");
        emitLine(out, 0, "}");
        return;
    }

    emitLine(out, 0, "export function " + serializeFn + "(value: " + typeName + "): Uint8Array {");
    emitLine(out, 1, "const out = new Uint8Array(" + std::to_string(maxByteLength) + ");");
    emitLine(out, 1, "let offsetBits = 0;");
    for (const auto& field : plan.fields)
    {
        const auto bits       = std::to_string(field.bitLength);
        const auto saturating = field.castMode == CastMode::Saturated ? "true" : "false";
        emitTsRuntimeAlignSerialize(out, 1, field.alignmentBits, field.fieldName);

        if (field.arrayKind == TsRuntimeArrayKind::None)
        {
            if (field.kind == TsRuntimeFieldKind::Padding)
            {
                emitTsRuntimeSerializePadding(out, 1, field, field.fieldName);
            }
            else if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 1, "dsdlRuntime.setBit(out, offsetBits, value." + field.fieldName + ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeSerializeCompositeValue(out, 1, field, "value." + field.fieldName, ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                emitLine(out,
                         1,
                         "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits + ", value." + field.fieldName + ", " +
                             saturating + ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out,
                         1,
                         "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", value." + field.fieldName + ");");
            }
            else
            {
                emitLine(out,
                         1,
                         "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", value." + field.fieldName + ", " +
                             saturating + ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 1, "offsetBits += " + bits + ";");
            }
        }
        else if (field.arrayKind == TsRuntimeArrayKind::Fixed)
        {
            const auto cap      = std::to_string(field.arrayCapacity);
            const auto fieldArr = field.fieldName + "Array";
            emitLine(out, 1, "const " + fieldArr + " = value." + field.fieldName + ";");
            emitLine(out, 1, "if (!Array.isArray(" + fieldArr + ") || " + fieldArr + ".length !== " + cap + ") {");
            emitLine(out,
                     2,
                     "throw new Error(\"field '" + field.fieldName + "' expects exactly " + cap + " elements\");");
            emitLine(out, 1, "}");
            emitLine(out, 1, "for (let i = 0; i < " + cap + "; ++i) {");
            if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 2, "dsdlRuntime.setBit(out, offsetBits, " + fieldArr + "[i]);");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeSerializeCompositeValue(out, 2, field, fieldArr + "[i]", ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                emitLine(out,
                         2,
                         "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits + ", " + fieldArr + "[i], " + saturating +
                             ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out, 2, "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", " + fieldArr + "[i]);");
            }
            else
            {
                emitLine(out,
                         2,
                         "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", " + fieldArr + "[i], " + saturating +
                             ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 2, "offsetBits += " + bits + ";");
            }
            emitLine(out, 1, "}");
        }
        else
        {
            const auto cap        = std::to_string(field.arrayCapacity);
            const auto prefixBits = std::to_string(field.arrayLengthPrefixBits);
            const auto fieldArr   = field.fieldName + "Array";
            emitLine(out, 1, "const " + fieldArr + " = value." + field.fieldName + ";");
            emitLine(out, 1, "if (!Array.isArray(" + fieldArr + ")) {");
            emitLine(out, 2, "throw new Error(\"field '" + field.fieldName + "' expects an array\");");
            emitLine(out, 1, "}");
            emitLine(out, 1, "if (" + fieldArr + ".length > " + cap + ") {");
            emitLine(out, 2, "throw new Error(\"field '" + field.fieldName + "' exceeds max length " + cap + "\");");
            emitLine(out, 1, "}");
            emitLine(out,
                     1,
                     "dsdlRuntime.writeUnsigned(out, offsetBits, " + prefixBits + ", " + fieldArr + ".length, false);");
            emitLine(out, 1, "offsetBits += " + prefixBits + ";");
            emitLine(out, 1, "for (let i = 0; i < " + fieldArr + ".length; ++i) {");
            if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 2, "dsdlRuntime.setBit(out, offsetBits, " + fieldArr + "[i]);");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeSerializeCompositeValue(out, 2, field, fieldArr + "[i]", ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                emitLine(out,
                         2,
                         "dsdlRuntime.writeUnsigned(out, offsetBits, " + bits + ", " + fieldArr + "[i], " + saturating +
                             ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out, 2, "dsdlRuntime.writeFloat(out, offsetBits, " + bits + ", " + fieldArr + "[i]);");
            }
            else
            {
                emitLine(out,
                         2,
                         "dsdlRuntime.writeSigned(out, offsetBits, " + bits + ", " + fieldArr + "[i], " + saturating +
                             ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 2, "offsetBits += " + bits + ";");
            }
            emitLine(out, 1, "}");
        }
    }
    emitLine(out, 1, "const usedBytes = dsdlRuntime.byteLengthForBits(offsetBits);");
    emitLine(out, 1, "return out.subarray(0, usedBytes);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");

    emitLine(out,
             0,
             "export function " + deserializeFn + "(bytes: Uint8Array): { value: " + typeName +
                 "; consumed: number } {");
    emitLine(out, 1, "const value = {} as " + typeName + ";");
    emitLine(out, 1, "let offsetBits = 0;");
    for (const auto& field : plan.fields)
    {
        const auto bits = std::to_string(field.bitLength);
        emitTsRuntimeAlignDeserialize(out, 1, field.alignmentBits);
        if (field.arrayKind == TsRuntimeArrayKind::None)
        {
            if (field.kind == TsRuntimeFieldKind::Padding)
            {
                emitTsRuntimeDeserializePadding(out, 1, field);
            }
            else if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 1, "value." + field.fieldName + " = dsdlRuntime.getBit(bytes, offsetBits);");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeDeserializeCompositeValue(out, 1, field, "value." + field.fieldName, ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                const std::string fn = field.useBigInt ? "readUnsignedBigInt" : "readUnsigned";
                emitLine(out,
                         1,
                         "value." + field.fieldName + " = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out,
                         1,
                         "value." + field.fieldName + " = dsdlRuntime.readFloat(bytes, offsetBits, " + bits + ");");
            }
            else
            {
                const std::string fn = field.useBigInt ? "readSignedBigInt" : "readSigned";
                emitLine(out,
                         1,
                         "value." + field.fieldName + " = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 1, "offsetBits += " + bits + ";");
            }
            continue;
        }

        if (field.arrayKind == TsRuntimeArrayKind::Fixed)
        {
            const auto cap      = std::to_string(field.arrayCapacity);
            const auto fieldArr = field.fieldName + "Array";
            const auto arrayElemType =
                field.kind == TsRuntimeFieldKind::Bool
                    ? "boolean"
                    : (field.kind == TsRuntimeFieldKind::Composite ? compositeTypeName(field, ctx)
                                                                   : (field.useBigInt ? "bigint" : "number"));
            emitLine(out, 1, "const " + fieldArr + ": Array<" + arrayElemType + "> = new Array(" + cap + ");");
            emitLine(out, 1, "for (let i = 0; i < " + cap + "; ++i) {");
            if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.getBit(bytes, offsetBits);");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeDeserializeCompositeValue(out, 2, field, fieldArr + "[i]", ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                const std::string fn = field.useBigInt ? "readUnsignedBigInt" : "readUnsigned";
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.readFloat(bytes, offsetBits, " + bits + ");");
            }
            else
            {
                const std::string fn = field.useBigInt ? "readSignedBigInt" : "readSigned";
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 2, "offsetBits += " + bits + ";");
            }
            emitLine(out, 1, "}");
            emitLine(out, 1, "value." + field.fieldName + " = " + fieldArr + ";");
        }
        else
        {
            const auto cap        = std::to_string(field.arrayCapacity);
            const auto prefixBits = std::to_string(field.arrayLengthPrefixBits);
            const auto fieldArr   = field.fieldName + "Array";
            const auto arrayElemType =
                field.kind == TsRuntimeFieldKind::Bool
                    ? "boolean"
                    : (field.kind == TsRuntimeFieldKind::Composite ? compositeTypeName(field, ctx)
                                                                   : (field.useBigInt ? "bigint" : "number"));
            emitLine(out,
                     1,
                     "const " + field.fieldName + "Length = Math.trunc(dsdlRuntime.readUnsigned(bytes, offsetBits, " +
                         prefixBits + "));");
            emitLine(out, 1, "offsetBits += " + prefixBits + ";");
            emitLine(out, 1, "if (" + field.fieldName + "Length < 0 || " + field.fieldName + "Length > " + cap + ") {");
            emitLine(out,
                     2,
                     "throw new Error(\"decoded length for field '" + field.fieldName + "' exceeds max length " + cap +
                         "\");");
            emitLine(out, 1, "}");
            emitLine(out,
                     1,
                     "const " + fieldArr + ": Array<" + arrayElemType + "> = new Array(" + field.fieldName +
                         "Length);");
            emitLine(out, 1, "for (let i = 0; i < " + field.fieldName + "Length; ++i) {");
            if (field.kind == TsRuntimeFieldKind::Bool)
            {
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.getBit(bytes, offsetBits);");
            }
            else if (field.kind == TsRuntimeFieldKind::Composite)
            {
                emitTsRuntimeDeserializeCompositeValue(out, 2, field, fieldArr + "[i]", ctx);
            }
            else if (field.kind == TsRuntimeFieldKind::Unsigned)
            {
                const std::string fn = field.useBigInt ? "readUnsignedBigInt" : "readUnsigned";
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            else if (field.kind == TsRuntimeFieldKind::Float)
            {
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime.readFloat(bytes, offsetBits, " + bits + ");");
            }
            else
            {
                const std::string fn = field.useBigInt ? "readSignedBigInt" : "readSigned";
                emitLine(out, 2, fieldArr + "[i] = dsdlRuntime." + fn + "(bytes, offsetBits, " + bits + ");");
            }
            if (field.kind != TsRuntimeFieldKind::Composite && field.kind != TsRuntimeFieldKind::Padding)
            {
                emitLine(out, 2, "offsetBits += " + bits + ";");
            }
            emitLine(out, 1, "}");
            emitLine(out, 1, "value." + field.fieldName + " = " + fieldArr + ";");
        }
    }
    emitLine(out, 1, "const consumed = Math.min(bytes.length, dsdlRuntime.byteLengthForBits(offsetBits));");
    emitLine(out, 1, "return { value, consumed };");
    emitLine(out, 0, "}");
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
    emitLine(out, 0, "// Generated by llvmdsdl (TypeScript backend).");

    const llvm::StringRef requestSectionKey = def.isService ? "request" : "";
    auto                  requestRuntimePlan =
        buildTsRuntimeSectionPlan(def.request, findLoweredSectionFacts(loweredFacts, def, requestSectionKey));
    if (!requestRuntimePlan)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "failed to build TypeScript request runtime plan for '%s': %s",
                                       def.info.fullName.c_str(),
                                       llvm::toString(requestRuntimePlan.takeError()).c_str());
    }
    std::optional<TsRuntimeSectionPlan> responseRuntimePlanStorage;
    const TsRuntimeSectionPlan*         responseRuntimePlan = nullptr;
    if (def.response)
    {
        auto responsePlanOrErr =
            buildTsRuntimeSectionPlan(*def.response, findLoweredSectionFacts(loweredFacts, def, "response"));
        if (!responsePlanOrErr)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "failed to build TypeScript response runtime plan for '%s': %s",
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

    const auto                                   ownerPath = ctx.relativeFilePath(def.info);
    std::map<std::string, std::set<std::string>> runtimeImportsByModule;
    const auto addRuntimeImportsForPlan = [&](const TsRuntimeSectionPlan* const plan) {
        if (plan == nullptr)
        {
            return;
        }
        for (const auto& field : plan->fields)
        {
            if (field.kind != TsRuntimeFieldKind::Composite || !field.compositeType)
            {
                continue;
            }
            const auto targetPath = ctx.relativeFilePath(*field.compositeType);
            if (targetPath == ownerPath)
            {
                continue;
            }
            const auto modulePath = relativeImportPath(ownerPath, targetPath);
            const auto typeName   = compositeTypeName(field, ctx);
            runtimeImportsByModule[modulePath].insert(tsRuntimeSerializeFn(typeName));
            runtimeImportsByModule[modulePath].insert(tsRuntimeDeserializeFn(typeName));
        }
    };
    addRuntimeImportsForPlan(&(*requestRuntimePlan));
    addRuntimeImportsForPlan(responseRuntimePlan);

    const auto runtimePath = relativeImportPath(ownerPath, std::filesystem::path("dsdl_runtime.ts"));
    emitLine(out, 0, "import * as dsdlRuntime from \"" + runtimePath + "\";");
    emitLine(out, 0, "");

    for (const auto& [modulePath, names] : runtimeImportsByModule)
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
        emitLine(out, 0, "import { " + importNames + " } from \"" + modulePath + "\";");
    }

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
        emitLine(out, 0, "import type { " + importNames + " } from \"" + modulePath + "\";");
    }
    const auto baseType = ctx.typeName(def.info);
    emitLine(out, 0, "export const DSDL_FULL_NAME = \"" + def.info.fullName + "\";");
    emitLine(out, 0, "export const DSDL_VERSION_MAJOR = " + std::to_string(def.info.majorVersion) + ";");
    emitLine(out, 0, "export const DSDL_VERSION_MINOR = " + std::to_string(def.info.minorVersion) + ";");
    emitLine(out, 0, "");

    if (!def.isService)
    {
        emitSectionType(out, baseType, def.request, ctx);
        emitLine(out, 0, "");
        emitSectionConstants(out, baseType, def.request);
        emitLine(out, 0, "");
        emitTsRuntimeFunctions(out, baseType, *requestRuntimePlan, ctx);
        return out.str();
    }

    const auto reqType  = baseType + "_Request";
    const auto respType = baseType + "_Response";

    emitSectionType(out, reqType, def.request, ctx);
    emitLine(out, 0, "");
    emitSectionConstants(out, reqType, def.request);
    emitLine(out, 0, "");
    emitTsRuntimeFunctions(out, reqType, *requestRuntimePlan, ctx);
    emitLine(out, 0, "");

    if (def.response)
    {
        emitSectionType(out, respType, *def.response, ctx);
        emitLine(out, 0, "");
        emitSectionConstants(out, respType, *def.response);
        emitLine(out, 0, "");
        emitTsRuntimeFunctions(out, respType, *responseRuntimePlan, ctx);
        emitLine(out, 0, "");
    }

    emitLine(out, 0, "export type " + baseType + " = " + reqType + ";");
    return out.str();
}

std::string renderPackageJson(const TsEmitOptions& options)
{
    const auto runtimeSpecialization =
        options.runtimeSpecialization == TsRuntimeSpecialization::Fast ? "fast" : "portable";
    std::ostringstream out;
    out << "{\n";
    out << "  \"name\": \"" << options.moduleName << "\",\n";
    out << "  \"version\": \"0.1.0\",\n";
    out << "  \"type\": \"module\",\n";
    out << "  \"llvmdsdl\": {\n";
    out << "    \"tsRuntimeSpecialization\": \"" << runtimeSpecialization << "\"\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

std::string renderTsRuntimeModule(const TsRuntimeSpecialization runtimeSpecialization)
{
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
    emitLine(out, 0, "export function copyBits(");
    emitLine(out, 1, "dst: Uint8Array,");
    emitLine(out, 1, "dstOffBits: number,");
    emitLine(out, 1, "src: Uint8Array,");
    emitLine(out, 1, "srcOffBits: number,");
    emitLine(out, 1, "lenBits: number");
    emitLine(out, 0, "): void {");
    emitLine(out, 1, "if (lenBits <= 0) {");
    emitLine(out, 2, "return;");
    emitLine(out, 1, "}");
    if (runtimeSpecialization == TsRuntimeSpecialization::Fast)
    {
        emitLine(out, 1, "if (dst !== src && dstOffBits % 8 === 0 && srcOffBits % 8 === 0 && lenBits % 8 === 0) {");
        emitLine(out, 2, "const dstStart = Math.floor(dstOffBits / 8);");
        emitLine(out, 2, "const srcStart = Math.floor(srcOffBits / 8);");
        emitLine(out, 2, "const byteLen = Math.floor(lenBits / 8);");
        emitLine(out, 2, "if (dstStart >= 0 && srcStart >= 0 &&");
        emitLine(out, 3, "dstStart + byteLen <= dst.length && srcStart + byteLen <= src.length) {");
        emitLine(out, 3, "dst.set(src.subarray(srcStart, srcStart + byteLen), dstStart);");
        emitLine(out, 3, "return;");
        emitLine(out, 2, "}");
        emitLine(out, 1, "}");
    }
    emitLine(out, 1, "for (let i = 0; i < lenBits; ++i) {");
    emitLine(out, 2, "setRawBit(dst, dstOffBits + i, getRawBit(src, srcOffBits + i));");
    emitLine(out, 1, "}");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function extractBits(");
    emitLine(out, 1, "src: Uint8Array,");
    emitLine(out, 1, "srcOffBits: number,");
    emitLine(out, 1, "lenBits: number");
    emitLine(out, 0, "): Uint8Array {");
    if (runtimeSpecialization == TsRuntimeSpecialization::Fast)
    {
        emitLine(out, 1, "if (srcOffBits % 8 === 0 && lenBits % 8 === 0) {");
        emitLine(out, 2, "const srcStart = Math.floor(srcOffBits / 8);");
        emitLine(out, 2, "const byteLen = Math.floor(lenBits / 8);");
        emitLine(out, 2, "if (srcStart >= 0 && srcStart + byteLen <= src.length) {");
        emitLine(out, 3, "return src.slice(srcStart, srcStart + byteLen);");
        emitLine(out, 2, "}");
        emitLine(out, 1, "}");
    }
    emitLine(out, 1, "const out = new Uint8Array(byteLengthForBits(lenBits));");
    emitLine(out, 1, "copyBits(out, 0, src, srcOffBits, lenBits);");
    emitLine(out, 1, "return out;");
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
    emitLine(out, 0, "function float32ToBits(value: number): number {");
    emitLine(out, 1, "const bytes = new ArrayBuffer(4);");
    emitLine(out, 1, "const view = new DataView(bytes);");
    emitLine(out, 1, "view.setFloat32(0, value, true);");
    emitLine(out, 1, "return view.getUint32(0, true);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "function bitsToFloat32(bits: number): number {");
    emitLine(out, 1, "const bytes = new ArrayBuffer(4);");
    emitLine(out, 1, "const view = new DataView(bytes);");
    emitLine(out, 1, "view.setUint32(0, bits >>> 0, true);");
    emitLine(out, 1, "return view.getFloat32(0, true);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "function float64ToBits(value: number): bigint {");
    emitLine(out, 1, "const bytes = new ArrayBuffer(8);");
    emitLine(out, 1, "const view = new DataView(bytes);");
    emitLine(out, 1, "view.setFloat64(0, value, true);");
    emitLine(out, 1, "return view.getBigUint64(0, true);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "function bitsToFloat64(bits: bigint): number {");
    emitLine(out, 1, "const bytes = new ArrayBuffer(8);");
    emitLine(out, 1, "const view = new DataView(bytes);");
    emitLine(out, 1, "view.setBigUint64(0, bits, true);");
    emitLine(out, 1, "return view.getFloat64(0, true);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "function float16ToBits(value: number): number {");
    emitLine(out, 1, "if (Number.isNaN(value)) {");
    emitLine(out, 2, "return 0x7e00;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (value === Infinity) {");
    emitLine(out, 2, "return 0x7c00;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (value === -Infinity) {");
    emitLine(out, 2, "return 0xfc00;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const bits = float32ToBits(value);");
    emitLine(out, 1, "const sign = (bits >>> 16) & 0x8000;");
    emitLine(out, 1, "let exp = ((bits >>> 23) & 0xff) - 127 + 15;");
    emitLine(out, 1, "let mant = bits & 0x7fffff;");
    emitLine(out, 1, "if (exp <= 0) {");
    emitLine(out, 2, "if (exp < -10) {");
    emitLine(out, 3, "return sign;");
    emitLine(out, 2, "}");
    emitLine(out, 2, "mant = (mant | 0x800000) >>> (1 - exp);");
    emitLine(out, 2, "if ((mant & 0x1000) !== 0) {");
    emitLine(out, 3, "mant += 0x2000;");
    emitLine(out, 2, "}");
    emitLine(out, 2, "return sign | (mant >>> 13);");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (exp >= 0x1f) {");
    emitLine(out, 2, "return sign | 0x7c00;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if ((mant & 0x1000) !== 0) {");
    emitLine(out, 2, "mant += 0x2000;");
    emitLine(out, 2, "if ((mant & 0x800000) !== 0) {");
    emitLine(out, 3, "mant = 0;");
    emitLine(out, 3, "exp += 1;");
    emitLine(out, 3, "if (exp >= 0x1f) {");
    emitLine(out, 4, "return sign | 0x7c00;");
    emitLine(out, 3, "}");
    emitLine(out, 2, "}");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return sign | (exp << 10) | (mant >>> 13);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "function bitsToFloat16(bits: number): number {");
    emitLine(out, 1, "const sign = (bits & 0x8000) !== 0 ? -1 : 1;");
    emitLine(out, 1, "const exp = (bits >>> 10) & 0x1f;");
    emitLine(out, 1, "const mant = bits & 0x03ff;");
    emitLine(out, 1, "if (exp === 0) {");
    emitLine(out, 2, "if (mant === 0) {");
    emitLine(out, 3, "return sign * 0;");
    emitLine(out, 2, "}");
    emitLine(out, 2, "return sign * Math.pow(2, -14) * (mant / 1024);");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (exp === 0x1f) {");
    emitLine(out, 2, "return mant === 0 ? sign * Infinity : Number.NaN;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return sign * Math.pow(2, exp - 15) * (1 + (mant / 1024));");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function writeFloat(");
    emitLine(out, 1, "buf: Uint8Array,");
    emitLine(out, 1, "offBits: number,");
    emitLine(out, 1, "lenBits: number,");
    emitLine(out, 1, "value: number");
    emitLine(out, 0, "): void {");
    emitLine(out, 1, "if (lenBits === 16) {");
    emitLine(out, 2, "writeUnsignedBits(buf, offBits, lenBits, BigInt(float16ToBits(value)));");
    emitLine(out, 2, "return;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (lenBits === 32) {");
    emitLine(out, 2, "writeUnsignedBits(buf, offBits, lenBits, BigInt(float32ToBits(value)));");
    emitLine(out, 2, "return;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (lenBits === 64) {");
    emitLine(out, 2, "writeUnsignedBits(buf, offBits, lenBits, float64ToBits(value));");
    emitLine(out, 2, "return;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "throw new Error(\"unsupported float bit length \" + lenBits);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function readFloat(buf: Uint8Array, offBits: number, lenBits: number): number {");
    emitLine(out, 1, "if (lenBits === 16) {");
    emitLine(out, 2, "return bitsToFloat16(Number(readUnsignedBits(buf, offBits, lenBits)));");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (lenBits === 32) {");
    emitLine(out, 2, "return bitsToFloat32(Number(readUnsignedBits(buf, offBits, lenBits)));");
    emitLine(out, 1, "}");
    emitLine(out, 1, "if (lenBits === 64) {");
    emitLine(out, 2, "return bitsToFloat64(readUnsignedBits(buf, offBits, lenBits));");
    emitLine(out, 1, "}");
    emitLine(out, 1, "throw new Error(\"unsupported float bit length \" + lenBits);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function readUnsignedBigInt(");
    emitLine(out, 1, "buf: Uint8Array,");
    emitLine(out, 1, "offBits: number,");
    emitLine(out, 1, "lenBits: number");
    emitLine(out, 0, "): bigint {");
    emitLine(out, 1, "if (lenBits <= 0) {");
    emitLine(out, 2, "return 0n;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return readUnsignedBits(buf, offBits, lenBits);");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function readSignedBigInt(");
    emitLine(out, 1, "buf: Uint8Array,");
    emitLine(out, 1, "offBits: number,");
    emitLine(out, 1, "lenBits: number");
    emitLine(out, 0, "): bigint {");
    emitLine(out, 1, "if (lenBits <= 0) {");
    emitLine(out, 2, "return 0n;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "const raw = readUnsignedBits(buf, offBits, lenBits);");
    emitLine(out, 1, "const signBit = 1n << BigInt(lenBits - 1);");
    emitLine(out, 1, "if ((raw & signBit) !== 0n) {");
    emitLine(out, 2, "return raw - (1n << BigInt(lenBits));");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return raw;");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function readUnsigned(buf: Uint8Array, offBits: number, lenBits: number): number {");
    emitLine(out, 1, "if (lenBits <= 0) {");
    emitLine(out, 2, "return 0;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return Number(readUnsignedBigInt(buf, offBits, lenBits));");
    emitLine(out, 0, "}");
    emitLine(out, 0, "");
    emitLine(out, 0, "export function readSigned(buf: Uint8Array, offBits: number, lenBits: number): number {");
    emitLine(out, 1, "if (lenBits <= 0) {");
    emitLine(out, 2, "return 0;");
    emitLine(out, 1, "}");
    emitLine(out, 1, "return Number(readSignedBigInt(buf, offBits, lenBits));");
    emitLine(out, 0, "}");
    return out.str();
}

}  // namespace

llvm::Error emitTs(const SemanticModule& semantic,
                   mlir::ModuleOp        module,
                   const TsEmitOptions&  options,
                   DiagnosticEngine&     diagnostics)
{
    if (options.outDir.empty())
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(), "output directory is required");
    }

    LoweredFactsMap loweredFacts;
    if (!collectLoweredFactsFromMlir(semantic,
                                     module,
                                     diagnostics,
                                     "TypeScript",
                                     &loweredFacts,
                                     options.optimizeLoweredSerDes))
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "MLIR schema coverage validation failed for TypeScript emission");
    }

    std::filesystem::path outRoot(options.outDir);
    std::filesystem::create_directories(outRoot);

    if (options.emitPackageJson)
    {
        if (auto err = writeFile(outRoot / "package.json", renderPackageJson(options)))
        {
            return err;
        }
    }
    if (auto err = writeFile(outRoot / "dsdl_runtime.ts", renderTsRuntimeModule(options.runtimeSpecialization)))
    {
        return err;
    }

    EmitterContext ctx(semantic);

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

    std::vector<std::filesystem::path> generatedRelativePaths;
    generatedRelativePaths.reserve(ordered.size());

    for (const auto* def : ordered)
    {
        const auto relPath = ctx.relativeFilePath(def->info);
        generatedRelativePaths.push_back(relPath);

        const auto fullPath = outRoot / relPath;
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

    std::ostringstream index;
    emitLine(index, 0, "// Generated by llvmdsdl (TypeScript backend).");
    std::map<std::string, unsigned> aliasUseCount;
    for (const auto& relPath : generatedRelativePaths)
    {
        std::string modulePath = relPath.generic_string();
        if (modulePath.size() >= 3 && modulePath.ends_with(".ts"))
        {
            modulePath.resize(modulePath.size() - 3);
        }
        std::string alias    = moduleAliasFromPath(modulePath);
        unsigned&   useCount = aliasUseCount[alias];
        if (useCount > 0)
        {
            alias += "_" + std::to_string(useCount);
        }
        ++useCount;

        emitLine(index, 0, "export * as " + alias + " from \"./" + modulePath + "\";");
    }

    if (auto err = writeFile(outRoot / "index.ts", index.str()))
    {
        return err;
    }

    return llvm::Error::success();
}

}  // namespace llvmdsdl
