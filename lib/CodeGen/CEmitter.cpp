//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements C backend code emission from lowered DSDL modules.
///
/// This file orchestrates pass pipelines, helper synthesis, and translation-unit rendering for generated C artifacts.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/CEmitter.h"
#include "llvmdsdl/CodeGen/MlirLoweredFacts.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/ilist_iterator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <system_error>
#include <variant>

#include "llvmdsdl/CodeGen/TypeStorage.h"
#include "llvmdsdl/CodeGen/ConstantLiteralRender.h"
#include "llvmdsdl/CodeGen/DefinitionIndex.h"
#include "llvmdsdl/CodeGen/NamingPolicy.h"
#include "llvmdsdl/CodeGen/StorageTypeTokens.h"
#include "llvmdsdl/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Evaluator.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "llvmdsdl/Support/Rational.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvmdsdl
{
namespace
{

std::string typeKey(const std::string& name, std::uint32_t major, std::uint32_t minor)
{
    return name + ":" + std::to_string(major) + ":" + std::to_string(minor);
}

std::string sanitizeMacroToken(std::string token)
{
    for (char& c : token)
    {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
        {
            c = '_';
        }
        else
        {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
    }
    if (!token.empty() && std::isdigit(static_cast<unsigned char>(token.front())))
    {
        token.insert(token.begin(), '_');
    }
    return token;
}

std::string headerFileName(const DiscoveredDefinition& info)
{
    return llvm::formatv("{0}_{1}_{2}.h", info.shortName, info.majorVersion, info.minorVersion).str();
}

std::string mangleSymbol(std::string fullName, std::uint32_t major, std::uint32_t minor)
{
    for (char& c : fullName)
    {
        if (c == '.')
        {
            c = '_';
        }
    }
    return fullName + "_" + std::to_string(major) + "_" + std::to_string(minor);
}

std::string sectionSuffix(const std::string& sectionName)
{
    if (sectionName == "request")
    {
        return "__request";
    }
    if (sectionName == "response")
    {
        return "__response";
    }
    return "";
}

std::string sectionIRFunctionStem(const SemanticDefinition& def, const std::string& sectionName)
{
    return mangleSymbol(def.info.fullName, def.info.majorVersion, def.info.minorVersion) + sectionSuffix(sectionName);
}

std::string implFileName(const DiscoveredDefinition& info)
{
    auto name = headerFileName(info);
    if (name.size() >= 2U && name.substr(name.size() - 2U) == ".h")
    {
        name.replace(name.size() - 2U, 2U, ".c");
    }
    else
    {
        name += ".c";
    }
    return name;
}

std::string cTypeNameFromInfo(const DiscoveredDefinition& info)
{
    std::string out;
    for (std::size_t i = 0; i < info.namespaceComponents.size(); ++i)
    {
        if (i > 0)
        {
            out += "__";
        }
        out += codegenSanitizeIdentifier(CodegenNamingLanguage::C, info.namespaceComponents[i]);
    }
    if (!out.empty())
    {
        out += "__";
    }
    out += codegenSanitizeIdentifier(CodegenNamingLanguage::C, info.shortName);
    return out;
}

std::string headerGuard(const DiscoveredDefinition& info)
{
    std::string g = "LLVMDSDL_" + info.fullName + "_" + std::to_string(info.majorVersion) + "_" +
                    std::to_string(info.minorVersion) + "_H";
    for (char& c : g)
    {
        if (!std::isalnum(static_cast<unsigned char>(c)))
        {
            c = '_';
        }
        else
        {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
    }
    return g;
}

std::string valueToCExpr(const Value& value)
{
    return renderConstantLiteral(ConstantLiteralLanguage::C, value);
}

std::string unsignedStorageType(const std::uint32_t bitLength)
{
    return renderUnsignedStorageToken(StorageTokenLanguage::C, bitLength);
}

std::string signedStorageType(const std::uint32_t bitLength)
{
    return renderSignedStorageToken(StorageTokenLanguage::C, bitLength);
}

class EmitterContext final
{
public:
    explicit EmitterContext(const SemanticModule& semantic)
        : index_(semantic)
    {
    }

    const SemanticDefinition* find(const SemanticTypeRef& ref) const
    {
        return index_.find(ref);
    }

    std::string cTypeName(const SemanticDefinition& def) const
    {
        return cTypeNameFromInfo(def.info);
    }

    std::string cTypeName(const SemanticTypeRef& ref) const
    {
        if (const auto* def = find(ref))
        {
            return cTypeName(*def);
        }

        DiscoveredDefinition tmp;
        tmp.fullName            = ref.fullName;
        tmp.shortName           = ref.shortName;
        tmp.namespaceComponents = ref.namespaceComponents;
        tmp.majorVersion        = ref.majorVersion;
        tmp.minorVersion        = ref.minorVersion;
        return cTypeNameFromInfo(tmp);
    }

    std::string relativeHeaderPath(const SemanticDefinition& def) const
    {
        std::filesystem::path p;
        for (const auto& ns : def.info.namespaceComponents)
        {
            p /= ns;
        }
        p /= headerFileName(def.info);
        return p.generic_string();
    }

    std::string relativeHeaderPath(const SemanticTypeRef& ref) const
    {
        if (const auto* def = find(ref))
        {
            return relativeHeaderPath(*def);
        }
        std::filesystem::path p;
        for (const auto& ns : ref.namespaceComponents)
        {
            p /= ns;
        }
        DiscoveredDefinition tmp;
        tmp.shortName    = ref.shortName;
        tmp.majorVersion = ref.majorVersion;
        tmp.minorVersion = ref.minorVersion;
        p /= headerFileName(tmp);
        return p.generic_string();
    }

private:
    DefinitionIndex index_;
};

void emitLine(std::ostringstream& out, const int indent, const std::string& line)
{
    out << std::string(static_cast<std::size_t>(indent) * 2U, ' ') << line << '\n';
}

std::string cTypeFromFieldType(const SemanticFieldType& type, const EmitterContext& ctx)
{
    switch (type.scalarCategory)
    {
    case SemanticScalarCategory::Bool:
        return "bool";
    case SemanticScalarCategory::Byte:
    case SemanticScalarCategory::Utf8:
    case SemanticScalarCategory::UnsignedInt:
        return unsignedStorageType(type.bitLength);
    case SemanticScalarCategory::SignedInt:
        return signedStorageType(type.bitLength);
    case SemanticScalarCategory::Float:
        if (type.bitLength == 64U)
        {
            return "double";
        }
        return "float";
    case SemanticScalarCategory::Void:
        return "uint8_t";
    case SemanticScalarCategory::Composite:
        if (type.compositeType)
        {
            return ctx.cTypeName(*type.compositeType);
        }
        return "uint8_t";
    }
    return "uint8_t";
}

void collectSectionDependencies(const SemanticSection& section, std::set<std::string>& out)
{
    for (const auto& field : section.fields)
    {
        if (field.resolvedType.compositeType)
        {
            const auto& ref = *field.resolvedType.compositeType;
            out.insert(typeKey(ref.fullName, ref.majorVersion, ref.minorVersion));
        }
    }
}

void emitArrayMacros(std::ostringstream& out, const std::string& typeName, const SemanticSection& section)
{
    for (const auto& field : section.fields)
    {
        if (field.isPadding || field.resolvedType.arrayKind == ArrayKind::None)
        {
            continue;
        }
        const auto fieldName = sanitizeMacroToken(field.name);
        emitLine(out,
                 0,
                 "#define " + typeName + "_" + fieldName + "_ARRAY_CAPACITY_ " +
                     std::to_string(field.resolvedType.arrayCapacity) + "U");
        emitLine(out,
                 0,
                 "#define " + typeName + "_" + fieldName + "_ARRAY_IS_VARIABLE_LENGTH_ " +
                     (isVariableArray(field.resolvedType.arrayKind) ? "true" : "false"));
    }
    if (!section.fields.empty())
    {
        out << "\n";
    }
}

void emitSectionTypedef(std::ostringstream&    out,
                        const std::string&     typeName,
                        const SemanticSection& section,
                        const EmitterContext&  ctx)
{
    emitLine(out, 0, "typedef struct " + typeName + " {");

    std::size_t emitted = 0;
    for (const auto& field : section.fields)
    {
        if (field.isPadding)
        {
            continue;
        }

        const auto cMember  = codegenSanitizeIdentifier(CodegenNamingLanguage::C, field.name);
        const auto baseType = cTypeFromFieldType(field.resolvedType, ctx);

        if (field.resolvedType.arrayKind == ArrayKind::None)
        {
            emitLine(out, 1, baseType + " " + cMember + ";");
            ++emitted;
            continue;
        }

        if (field.resolvedType.arrayKind == ArrayKind::Fixed)
        {
            if (field.resolvedType.scalarCategory == SemanticScalarCategory::Bool)
            {
                emitLine(out,
                         1,
                         "uint8_t " + cMember + "[(" + std::to_string(field.resolvedType.arrayCapacity) +
                             "U + 7U) / 8U];");
            }
            else
            {
                emitLine(out,
                         1,
                         baseType + " " + cMember + "[" + std::to_string(field.resolvedType.arrayCapacity) + "U];");
            }
            ++emitted;
            continue;
        }

        emitLine(out, 1, "struct {");
        if (field.resolvedType.scalarCategory == SemanticScalarCategory::Bool)
        {
            emitLine(out,
                     2,
                     "uint8_t bitpacked[(" + std::to_string(field.resolvedType.arrayCapacity) + "U + 7U) / 8U];");
        }
        else
        {
            emitLine(out, 2, baseType + " elements[" + std::to_string(field.resolvedType.arrayCapacity) + "U];");
        }
        emitLine(out, 2, "size_t count;");
        emitLine(out, 1, "} " + cMember + ";");
        ++emitted;
    }

    if (section.isUnion)
    {
        emitLine(out, 1, "uint8_t _tag_;");
        ++emitted;
    }

    if (emitted == 0)
    {
        emitLine(out, 1, "uint8_t _dummy_;");
    }

    emitLine(out, 0, "} " + typeName + ";");
    out << "\n";

    if (section.isUnion)
    {
        std::size_t optionCount = 0;
        for (const auto& f : section.fields)
        {
            if (!f.isPadding)
            {
                ++optionCount;
            }
        }
        emitLine(out, 0, "#define " + typeName + "_UNION_OPTION_COUNT_ " + std::to_string(optionCount) + "U");
        out << "\n";
    }
}

void emitSectionConstants(std::ostringstream& out, const std::string& typeName, const SemanticSection& section)
{
    for (const auto& c : section.constants)
    {
        emitLine(out, 0, "#define " + typeName + "_" + sanitizeMacroToken(c.name) + " (" + valueToCExpr(c.value) + ")");
    }
    if (!section.constants.empty())
    {
        out << "\n";
    }
}

void emitSectionMetadata(std::ostringstream&    out,
                         const std::string&     typeName,
                         const std::string&     fullName,
                         std::uint32_t          majorVersion,
                         std::uint32_t          minorVersion,
                         const SemanticSection& section)
{
    emitLine(out, 0, "#define " + typeName + "_FULL_NAME_ \"" + fullName + "\"");
    emitLine(out,
             0,
             "#define " + typeName + "_FULL_NAME_AND_VERSION_ \"" + fullName + "." + std::to_string(majorVersion) +
                 "." + std::to_string(minorVersion) + "\"");
    emitLine(out,
             0,
             "#define " + typeName + "_EXTENT_BYTES_ " + std::to_string(section.extentBits.value_or(0) / 8) + "UL");
    emitLine(out,
             0,
             "#define " + typeName + "_SERIALIZATION_BUFFER_SIZE_BYTES_ " +
                 std::to_string((section.serializationBufferSizeBits + 7) / 8) + "UL");
    out << "\n";
}

void emitSection(std::ostringstream&       out,
                 const EmitterContext&     ctx,
                 const SemanticDefinition& def,
                 const std::string&        typeName,
                 const std::string&        fullName,
                 const std::string&        sectionName,
                 const SemanticSection&    section)
{
    emitSectionMetadata(out, typeName, fullName, def.info.majorVersion, def.info.minorVersion, section);
    emitSectionConstants(out, typeName, section);
    emitArrayMacros(out, typeName, section);
    emitSectionTypedef(out, typeName, section, ctx);

    const auto irStem = sectionIRFunctionStem(def, sectionName);
    emitLine(out,
             0,
             "int8_t " + irStem + "__serialize_ir_(const " + typeName +
                 "* const obj, uint8_t* buffer, size_t* const "
                 "inout_buffer_size_bytes);");
    emitLine(out,
             0,
             "int8_t " + irStem + "__deserialize_ir_(" + typeName +
                 "* const out_obj, const uint8_t* buffer, size_t* const "
                 "inout_buffer_size_bytes);");
    out << "\n";

    emitLine(out,
             0,
             "static inline int8_t " + typeName + "__serialize_(const " + typeName +
                 "* const obj, uint8_t* const buffer, size_t* const "
                 "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1, "return " + irStem + "__serialize_ir_(obj, buffer, inout_buffer_size_bytes);");
    emitLine(out, 0, "}");
    out << "\n";

    emitLine(out,
             0,
             "static inline int8_t " + typeName + "__deserialize_(" + typeName +
                 "* const out_obj, const uint8_t* buffer, size_t* const "
                 "inout_buffer_size_bytes)");
    emitLine(out, 0, "{");
    emitLine(out, 1, "return " + irStem + "__deserialize_ir_(out_obj, buffer, inout_buffer_size_bytes);");
    emitLine(out, 0, "}");
    out << "\n";
}

llvm::Expected<std::string> loadRuntimeHeader()
{
    const std::filesystem::path absoluteRuntimeHeader =
        std::filesystem::path(LLVMDSDL_SOURCE_DIR) / "runtime" / "dsdl_runtime.h";
    std::ifstream in(absoluteRuntimeHeader.string());
    if (!in)
    {
        // Fallback for environments where compile-time source definitions are
        // unavailable or altered.
        in.open("runtime/dsdl_runtime.h");
    }
    if (!in)
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(), "failed to read runtime header");
    }
    std::ostringstream content;
    content << in.rdbuf();
    return content.str();
}

std::string renderHeader(const SemanticDefinition& def, const EmitterContext& ctx)
{
    std::ostringstream out;
    const auto         guard        = headerGuard(def.info);
    const auto         baseTypeName = ctx.cTypeName(def);

    std::set<std::string> dependencies;
    collectSectionDependencies(def.request, dependencies);
    if (def.response)
    {
        collectSectionDependencies(*def.response, dependencies);
    }

    out << "#ifndef " << guard << "\n";
    out << "#define " << guard << "\n\n";
    out << "#include <stddef.h>\n";
    out << "#include <stdint.h>\n";
    out << "#include <stdbool.h>\n";
    out << "#include \"dsdl_runtime.h\"\n";

    for (const auto& depKey : dependencies)
    {
        auto split0 = depKey.find(':');
        auto split1 = depKey.find(':', split0 + 1);
        if (split0 == std::string::npos || split1 == std::string::npos)
        {
            continue;
        }
        SemanticTypeRef ref;
        ref.fullName     = depKey.substr(0, split0);
        ref.majorVersion = static_cast<std::uint32_t>(std::stoul(depKey.substr(split0 + 1, split1 - split0 - 1)));
        ref.minorVersion = static_cast<std::uint32_t>(std::stoul(depKey.substr(split1 + 1)));
        if (const auto* dep = ctx.find(ref))
        {
            out << "#include \"" << ctx.relativeHeaderPath(*dep) << "\"\n";
        }
    }
    out << "\n";

    if (def.isService)
    {
        const auto requestType  = baseTypeName + "__Request";
        const auto responseType = baseTypeName + "__Response";

        emitLine(out, 0, "#define " + baseTypeName + "_FULL_NAME_ \"" + def.info.fullName + "\"");
        emitLine(out,
                 0,
                 "#define " + baseTypeName + "_FULL_NAME_AND_VERSION_ \"" + def.info.fullName + "." +
                     std::to_string(def.info.majorVersion) + "." + std::to_string(def.info.minorVersion) + "\"");
        out << "\n";

        emitSection(out, ctx, def, requestType, def.info.fullName + ".Request", "request", def.request);
        if (def.response)
        {
            emitSection(out, ctx, def, responseType, def.info.fullName + ".Response", "response", *def.response);
        }

        emitLine(out, 0, "typedef " + requestType + " " + baseTypeName + ";");
        emitLine(out, 0, "#define " + baseTypeName + "_EXTENT_BYTES_ " + requestType + "_EXTENT_BYTES_");
        emitLine(out,
                 0,
                 "#define " + baseTypeName + "_SERIALIZATION_BUFFER_SIZE_BYTES_ " + requestType +
                     "_SERIALIZATION_BUFFER_SIZE_BYTES_");
        out << "\n";

        emitLine(out,
                 0,
                 "static inline int8_t " + baseTypeName + "__serialize_(const " + baseTypeName +
                     "* const obj, uint8_t* const buffer, size_t* const "
                     "inout_buffer_size_bytes)");
        emitLine(out, 0, "{");
        emitLine(out,
                 1,
                 "return " + requestType + "__serialize_((const " + requestType +
                     "*)obj, buffer, "
                     "inout_buffer_size_bytes);");
        emitLine(out, 0, "}");
        out << "\n";

        emitLine(out,
                 0,
                 "static inline int8_t " + baseTypeName + "__deserialize_(" + baseTypeName +
                     "* const out_obj, const uint8_t* buffer, size_t* const "
                     "inout_buffer_size_bytes)");
        emitLine(out, 0, "{");
        emitLine(out,
                 1,
                 "return " + requestType + "__deserialize_((" + requestType +
                     "*)out_obj, buffer, inout_buffer_size_bytes);");
        emitLine(out, 0, "}");
    }
    else
    {
        emitSection(out, ctx, def, baseTypeName, def.info.fullName, "", def.request);
    }

    out << "#endif /* " << guard << " */\n";
    return out.str();
}

}  // namespace

llvm::Error emitC(const SemanticModule& semantic,
                  mlir::ModuleOp        module,
                  const CEmitOptions&   options,
                  DiagnosticEngine&     diagnostics)
{
    if (options.outDir.empty())
    {
        return llvm::createStringError(llvm::inconvertibleErrorCode(), "output directory is required");
    }

    std::filesystem::path outRoot(options.outDir);
    EmitterContext        ctx(semantic);
    const auto            selectedTypeKeys = makeTypeKeySet(options.selectedTypeKeys);

    if (!collectLoweredFactsFromMlir(semantic, module, diagnostics, "C", nullptr, options.optimizeLoweredSerDes))
    {
        diagnostics.error({"<mlir>", 1, 1}, "MLIR schema coverage validation failed for C emission");
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "MLIR schema coverage validation failed for C emission");
    }

    std::unordered_map<std::string, mlir::Operation*> schemaByHeaderPath;
    for (mlir::Operation& op : module.getBodyRegion().front())
    {
        if (op.getName().getStringRef() != "dsdl.schema")
        {
            continue;
        }
        const auto headerPathAttr = op.getAttrOfType<mlir::StringAttr>("header_path");
        if (!headerPathAttr)
        {
            continue;
        }
        schemaByHeaderPath.emplace(headerPathAttr.str(), &op);
    }

    for (const auto& def : semantic.definitions)
    {
        if (!shouldEmitDefinition(def.info, selectedTypeKeys))
        {
            continue;
        }

        auto perDefModuleRef = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(module.getLoc()));
        auto perDefModule    = *perDefModuleRef;
        perDefModule->setAttr("llvmdsdl.headers_available", mlir::UnitAttr::get(perDefModule.getContext()));
        perDefModule->setAttr("llvmdsdl.require_typed_lowering", mlir::UnitAttr::get(perDefModule.getContext()));

        const std::string targetHeaderPath = ctx.relativeHeaderPath(def);
        const auto        targetIt         = schemaByHeaderPath.find(targetHeaderPath);
        if (targetIt == schemaByHeaderPath.end())
        {
            diagnostics.error({"<mlir>", 1, 1},
                              "failed to locate schema op for " + def.info.fullName + " (" + targetHeaderPath + ")");
            return llvm::createStringError(llvm::inconvertibleErrorCode(), "schema selection failed");
        }
        perDefModule.getBodyRegion().front().push_back(targetIt->second->clone());

        mlir::PassManager pm(perDefModule.getContext());
        pm.addPass(createLowerDSDLSerializationPass());
        if (options.optimizeLoweredSerDes)
        {
            addOptimizeLoweredSerDesPipeline(pm);
        }
        pm.addPass(createConvertDSDLToEmitCPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSCFToEmitC());
        pm.addPass(mlir::createConvertArithToEmitC());
        pm.addPass(mlir::createConvertFuncToEmitC());

        if (mlir::failed(pm.run(perDefModule)))
        {
            diagnostics.error({"<mlir>", 1, 1}, "EmitC lowering pipeline failed");
            return llvm::createStringError(llvm::inconvertibleErrorCode(), "EmitC lowering pipeline failed");
        }

        std::string              emitted;
        llvm::raw_string_ostream emittedStream(emitted);
        if (mlir::failed(mlir::emitc::translateToCpp(perDefModule, emittedStream, options.declareVariablesAtTop)))
        {
            diagnostics.error({"<mlir>", 1, 1}, "EmitC translation failed");
            return llvm::createStringError(llvm::inconvertibleErrorCode(), "EmitC translation failed");
        }

        std::filesystem::path implDir = outRoot;
        for (const auto& ns : def.info.namespaceComponents)
        {
            implDir /= ns;
        }
        if (auto err = writeGeneratedFile(implDir / implFileName(def.info), emitted, options.writePolicy))
        {
            return err;
        }
    }

    auto runtimeHeader = loadRuntimeHeader();
    if (!runtimeHeader)
    {
        return runtimeHeader.takeError();
    }
    if (auto err = writeGeneratedFile(outRoot / "dsdl_runtime.h", *runtimeHeader, options.writePolicy))
    {
        return err;
    }

    for (const auto& def : semantic.definitions)
    {
        if (!shouldEmitDefinition(def.info, selectedTypeKeys))
        {
            continue;
        }

        std::filesystem::path dir = outRoot;
        for (const auto& ns : def.info.namespaceComponents)
        {
            dir /= ns;
        }
        if (auto err = writeGeneratedFile(dir / headerFileName(def.info), renderHeader(def, ctx), options.writePolicy))
        {
            return err;
        }
    }

    return llvm::Error::success();
}

}  // namespace llvmdsdl
