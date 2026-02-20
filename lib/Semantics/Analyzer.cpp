//===----------------------------------------------------------------------===//
///
/// @file
/// Implements semantic analysis for parsed DSDL definitions.
///
/// The analyzer resolves symbols, evaluates type rules, and computes layout and extent facts used by lowering.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/Semantics/Analyzer.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvmdsdl/Semantics/Evaluator.h"
#include "llvm/Support/Error.h"
#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Frontend/SourceLocation.h"
#include "llvmdsdl/Semantics/BitLengthSet.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "llvmdsdl/Support/Rational.h"

namespace llvmdsdl
{
namespace
{

struct TypeLayout final
{
    BitLengthSet      bls;
    std::int64_t      alignment{1};
    SemanticFieldType resolved;
};

std::string typeKey(const std::string& name, std::uint32_t major, std::uint32_t minor)
{
    return name + ":" + std::to_string(major) + ":" + std::to_string(minor);
}

std::int64_t ceilLog2(std::int64_t x)
{
    if (x <= 1)
    {
        return 0;
    }
    std::int64_t v   = x - 1;
    std::int64_t out = 0;
    while (v > 0)
    {
        v >>= 1;
        ++out;
    }
    return out;
}

std::int64_t pow2ceil(std::int64_t x)
{
    if (x <= 1)
    {
        return 1;
    }
    std::int64_t out = 1;
    while (out < x && out < (1LL << 62))
    {
        out <<= 1;
    }
    return out;
}

bool exprContainsOffset(const std::shared_ptr<ExprAST>& expr)
{
    if (!expr)
    {
        return false;
    }

    if (auto id = std::get_if<ExprAST::Identifier>(&expr->value))
    {
        return id->name == "_offset_";
    }
    if (auto u = std::get_if<ExprAST::Unary>(&expr->value))
    {
        return exprContainsOffset(u->operand);
    }
    if (auto b = std::get_if<ExprAST::Binary>(&expr->value))
    {
        return exprContainsOffset(b->lhs) || exprContainsOffset(b->rhs);
    }
    if (auto s = std::get_if<ExprAST::SetLiteral>(&expr->value))
    {
        for (const auto& e : s->elements)
        {
            if (exprContainsOffset(e))
            {
                return true;
            }
        }
    }
    return false;
}

Value::Set bitLengthSetToValueSet(const BitLengthSet& bls)
{
    Value::Set out;
    for (const auto v : bls.expand(4096))
    {
        out.insert(Rational(v, 1));
    }
    return out;
}

class AnalyzerImpl final
{
public:
    AnalyzerImpl(const ASTModule& module, DiagnosticEngine& diagnostics)
        : module_(module)
        , diagnostics_(diagnostics)
    {
        state_.resize(module_.definitions.size(), State::Unvisited);
        results_.resize(module_.definitions.size());
        for (std::size_t i = 0; i < module_.definitions.size(); ++i)
        {
            const auto& d = module_.definitions[i];
            indexByKey_.emplace(typeKey(d.info.fullName, d.info.majorVersion, d.info.minorVersion), i);
        }
    }

    llvm::Expected<SemanticModule> run()
    {
        for (std::size_t i = 0; i < module_.definitions.size(); ++i)
        {
            (void) analyzeOne(i);
        }

        checkVersionRules();

        if (diagnostics_.hasErrors())
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(), "semantic analysis failed");
        }

        SemanticModule out;
        out.definitions.reserve(results_.size());
        for (auto& r : results_)
        {
            if (r)
            {
                out.definitions.push_back(*r);
            }
        }
        return out;
    }

private:
    enum class State
    {
        Unvisited,
        Visiting,
        Done,
    };

    const ASTModule&                               module_;
    DiagnosticEngine&                              diagnostics_;
    std::vector<State>                             state_;
    std::vector<std::optional<SemanticDefinition>> results_;
    std::unordered_map<std::string, std::size_t>   indexByKey_;

    std::optional<std::size_t> resolveDefinitionIndex(const std::string& fullName,
                                                      std::uint32_t      major,
                                                      std::uint32_t      minor)
    {
        const auto it = indexByKey_.find(typeKey(fullName, major, minor));
        if (it == indexByKey_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    static std::string joinComponents(const std::vector<std::string>& components)
    {
        std::ostringstream out;
        for (std::size_t i = 0; i < components.size(); ++i)
        {
            if (i > 0)
            {
                out << '.';
            }
            out << components[i];
        }
        return out.str();
    }

    std::optional<std::pair<std::size_t, std::string>> resolveCompositeType(const DiscoveredDefinition& owner,
                                                                            const VersionedTypeExprAST& type,
                                                                            const SourceLocation&       location,
                                                                            bool                        reportError)
    {
        std::vector<std::string> candidates;

        const std::string explicitName = joinComponents(type.nameComponents);
        if (type.nameComponents.size() == 1)
        {
            if (!owner.namespaceComponents.empty())
            {
                candidates.push_back(joinComponents(owner.namespaceComponents) + "." + type.nameComponents.front());
            }
            else
            {
                candidates.push_back(type.nameComponents.front());
            }
        }
        else
        {
            candidates.push_back(explicitName);

            if (!owner.namespaceComponents.empty())
            {
                candidates.push_back(owner.namespaceComponents.front() + "." + explicitName);
                candidates.push_back(joinComponents(owner.namespaceComponents) + "." + explicitName);
            }
        }

        std::set<std::string> seen;
        for (const auto& candidate : candidates)
        {
            if (!seen.insert(candidate).second)
            {
                continue;
            }
            auto idx = resolveDefinitionIndex(candidate, type.major, type.minor);
            if (idx)
            {
                return std::make_pair(*idx, candidate);
            }
        }

        if (reportError)
        {
            diagnostics_.error(location,
                               "unresolved composite type: " + explicitName + "." + std::to_string(type.major) + "." +
                                   std::to_string(type.minor));
        }
        return std::nullopt;
    }

    std::optional<SemanticDefinition>* analyzeOne(std::size_t idx)
    {
        if (state_[idx] == State::Done)
        {
            return &results_[idx];
        }
        if (state_[idx] == State::Visiting)
        {
            diagnostics_.error(module_.definitions[idx].ast.location, "circular dependency detected");
            return &results_[idx];
        }

        state_[idx]        = State::Visiting;
        const auto& parsed = module_.definitions[idx];

        SemanticDefinition sem;
        sem.info      = parsed.info;
        sem.isService = parsed.ast.isService();

        std::vector<StatementAST> req;
        std::vector<StatementAST> resp;
        bool                      inResponse  = false;
        int                       markerCount = 0;

        for (const auto& stmt : parsed.ast.statements)
        {
            if (std::holds_alternative<ServiceResponseMarkerAST>(stmt))
            {
                ++markerCount;
                inResponse = true;
                continue;
            }
            if (inResponse)
            {
                resp.push_back(stmt);
            }
            else
            {
                req.push_back(stmt);
            }
        }

        if (!sem.isService && markerCount > 0)
        {
            diagnostics_.error(parsed.ast.location, "service response marker is only valid in service types");
        }
        if (markerCount > 1)
        {
            diagnostics_.error(parsed.ast.location, "duplicated service response marker");
        }

        sem.request = analyzeSection(parsed.info, req, /*isResponse=*/false);

        if (sem.isService)
        {
            auto response = analyzeSection(parsed.info, resp, /*isResponse=*/true);
            if (response.deprecated)
            {
                diagnostics_.error(parsed.ast.location, "@deprecated is only allowed in service request section");
            }
            response.deprecated = sem.request.deprecated;
            sem.response        = std::move(response);
        }

        results_[idx] = sem;
        state_[idx]   = State::Done;
        return &results_[idx];
    }

    TypeLayout analyzeType(const DiscoveredDefinition&  owner,
                           const TypeExprAST&           type,
                           ValueEnv&                    env,
                           const TypeAttributeResolver* resolver)
    {
        TypeLayout out;
        out.bls                   = BitLengthSet(0);
        out.alignment             = 1;
        out.resolved.bitLengthSet = out.bls;

        auto baseFromScalar = [&]() -> TypeLayout {
            if (auto p = std::get_if<PrimitiveTypeExprAST>(&type.scalar))
            {
                TypeLayout layout;
                layout.bls                    = BitLengthSet(static_cast<std::int64_t>(p->bitLength));
                layout.alignment              = 1;
                layout.resolved.bitLengthSet  = layout.bls;
                layout.resolved.bitLength     = p->bitLength;
                layout.resolved.castMode      = p->castMode;
                layout.resolved.alignmentBits = 1;
                switch (p->kind)
                {
                case PrimitiveKind::Bool:
                    layout.resolved.scalarCategory = SemanticScalarCategory::Bool;
                    break;
                case PrimitiveKind::Byte:
                    layout.resolved.scalarCategory = SemanticScalarCategory::Byte;
                    break;
                case PrimitiveKind::Utf8:
                    layout.resolved.scalarCategory = SemanticScalarCategory::Utf8;
                    break;
                case PrimitiveKind::UnsignedInt:
                    layout.resolved.scalarCategory = SemanticScalarCategory::UnsignedInt;
                    break;
                case PrimitiveKind::SignedInt:
                    layout.resolved.scalarCategory = SemanticScalarCategory::SignedInt;
                    break;
                case PrimitiveKind::Float:
                    layout.resolved.scalarCategory = SemanticScalarCategory::Float;
                    break;
                }
                return layout;
            }
            if (auto p = std::get_if<VoidTypeExprAST>(&type.scalar))
            {
                TypeLayout layout;
                layout.bls                     = BitLengthSet(static_cast<std::int64_t>(p->bitLength));
                layout.alignment               = 1;
                layout.resolved.scalarCategory = SemanticScalarCategory::Void;
                layout.resolved.bitLength      = p->bitLength;
                layout.resolved.alignmentBits  = 1;
                layout.resolved.bitLengthSet   = layout.bls;
                return layout;
            }

            const auto* v = std::get_if<VersionedTypeExprAST>(&type.scalar);
            if (!v)
            {
                return out;
            }

            auto resolved = resolveCompositeType(owner, *v, type.location, true);
            if (!resolved)
            {
                return out;
            }

            auto* resolvedDef = analyzeOne(resolved->first);
            if (!resolvedDef || !*resolvedDef)
            {
                return out;
            }
            const auto& def = **resolvedDef;
            if (def.isService)
            {
                diagnostics_.error(type.location, "service types are not valid field types");
                return out;
            }

            TypeLayout layout;
            layout.alignment               = 8;
            layout.resolved.scalarCategory = SemanticScalarCategory::Composite;
            layout.resolved.alignmentBits  = 8;
            layout.resolved.compositeType  = SemanticTypeRef{def.info.fullName,
                                                            def.info.namespaceComponents,
                                                            def.info.shortName,
                                                            def.info.majorVersion,
                                                            def.info.minorVersion};

            const auto& sec                     = def.request;
            layout.resolved.compositeSealed     = sec.sealed;
            layout.resolved.compositeExtentBits = sec.extentBits.value_or(sec.offsetAtEnd.max());
            if (sec.sealed)
            {
                layout.bls                   = sec.offsetAtEnd;
                layout.resolved.bitLengthSet = layout.bls;
                return layout;
            }

            const std::int64_t extent = sec.extentBits.value_or(0);
            BitLengthSet       bls(32);
            bls                          = bls + BitLengthSet(8).repeatRange(extent / 8);
            layout.bls                   = bls;
            layout.resolved.bitLengthSet = layout.bls;
            return layout;
        };

        auto scalarLayout               = baseFromScalar();
        scalarLayout.resolved.arrayKind = type.arrayKind;

        if (type.arrayKind == ArrayKind::None)
        {
            return scalarLayout;
        }

        if (std::holds_alternative<VoidTypeExprAST>(type.scalar))
        {
            diagnostics_.error(type.location, "void type is not allowed as array element type");
            return out;
        }

        std::int64_t capacity = 0;
        if (!type.arrayCapacity)
        {
            diagnostics_.error(type.location, "array type missing capacity expression");
            return out;
        }

        auto capValue = evaluateExpression(*type.arrayCapacity, env, diagnostics_, type.location, resolver);
        if (!capValue || !std::holds_alternative<Rational>(capValue->data) ||
            !std::get<Rational>(capValue->data).isInteger())
        {
            diagnostics_.error(type.location, "array capacity expression must yield integer rational");
            return out;
        }
        else
        {
            capacity = std::get<Rational>(capValue->data).asInteger().value();
        }

        if (type.arrayKind == ArrayKind::Fixed)
        {
            if (capacity < 1)
            {
                diagnostics_.error(type.location, "fixed-length array capacity must be positive");
                return out;
            }
            TypeLayout layout;
            layout.bls                            = scalarLayout.bls.repeat(capacity);
            layout.alignment                      = scalarLayout.alignment;
            layout.resolved                       = scalarLayout.resolved;
            layout.resolved.arrayKind             = type.arrayKind;
            layout.resolved.arrayCapacity         = capacity;
            layout.resolved.arrayLengthPrefixBits = 0;
            layout.resolved.alignmentBits         = layout.alignment;
            layout.resolved.bitLengthSet          = layout.bls;
            return layout;
        }

        if (type.arrayKind == ArrayKind::VariableExclusive)
        {
            if (capacity <= 1)
            {
                diagnostics_.error(type.location, "exclusive variable-length array bound must be > 1");
                return out;
            }
            capacity -= 1;
        }
        else if (type.arrayKind == ArrayKind::VariableInclusive)
        {
            if (capacity < 1)
            {
                diagnostics_.error(type.location, "inclusive variable-length array bound must be >= 1");
                return out;
            }
        }

        const auto b          = ceilLog2(capacity + 1);
        const auto prefixBits = pow2ceil(std::max<std::int64_t>(8, b));

        BitLengthSet bls(prefixBits);
        bls = bls + scalarLayout.bls.repeatRange(capacity);
        TypeLayout layout;
        layout.bls                            = bls;
        layout.alignment                      = scalarLayout.alignment;
        layout.resolved                       = scalarLayout.resolved;
        layout.resolved.arrayKind             = type.arrayKind;
        layout.resolved.arrayCapacity         = capacity;
        layout.resolved.arrayLengthPrefixBits = prefixBits;
        layout.resolved.alignmentBits         = layout.alignment;
        layout.resolved.bitLengthSet          = layout.bls;
        return layout;
    }

    bool checkConstantCompatibility(const ConstantDeclAST& decl, const Value& value)
    {
        auto prim = std::get_if<PrimitiveTypeExprAST>(&decl.type.scalar);
        if (!prim)
        {
            diagnostics_.error(decl.location, "constant attributes must be primitive-typed");
            return false;
        }

        if (prim->kind == PrimitiveKind::Bool)
        {
            if (!std::holds_alternative<bool>(value.data))
            {
                diagnostics_.error(decl.location, "boolean constants must be initialized from bool");
                return false;
            }
            return true;
        }

        if (prim->kind == PrimitiveKind::UnsignedInt || prim->kind == PrimitiveKind::SignedInt)
        {
            if (auto r = std::get_if<Rational>(&value.data))
            {
                if (!r->isInteger())
                {
                    diagnostics_.error(decl.location, "integer constants require integer rational values");
                    return false;
                }
                return true;
            }
            if (auto s = std::get_if<std::string>(&value.data))
            {
                if (prim->kind == PrimitiveKind::UnsignedInt && prim->bitLength == 8 && s->size() == 1 &&
                    static_cast<unsigned char>((*s)[0]) <= 127)
                {
                    return true;
                }
            }
            diagnostics_.error(decl.location, "integer constants require rational or ASCII single-char string");
            return false;
        }

        if (prim->kind == PrimitiveKind::Float)
        {
            if (!std::holds_alternative<Rational>(value.data))
            {
                diagnostics_.error(decl.location, "floating constants require rational values");
                return false;
            }
            return true;
        }

        diagnostics_.error(decl.location, "byte/utf8 are not valid constant attribute types");
        return false;
    }

    SemanticSection analyzeSection(const DiscoveredDefinition&      owner,
                                   const std::vector<StatementAST>& statements,
                                   bool                             isResponse)
    {
        SemanticSection section;

        ValueEnv                env;
        BitLengthSet            structureOffset(0);
        std::vector<TypeLayout> unionFields;
        bool                    attributesStarted    = false;
        bool                    serializationModeSet = false;
        bool                    unionOffsetUsed      = false;
        std::uint32_t           unionOptionCounter   = 0;

        TypeAttributeResolver typeAttrResolver = [&](const TypeExprAST&    typeExpr,
                                                     const std::string&    attributeName,
                                                     const SourceLocation& location) -> std::optional<Value> {
            const auto unsupported = [&](const std::string& message) -> std::optional<Value> {
                diagnostics_.error(location, message);
                return std::nullopt;
            };

            if (typeExpr.arrayKind != ArrayKind::None)
            {
                return unsupported("unsupported metaserializable attribute: " + attributeName);
            }

            const auto* versioned = std::get_if<VersionedTypeExprAST>(&typeExpr.scalar);
            if (!versioned)
            {
                return unsupported("unsupported metaserializable attribute: " + attributeName);
            }

            auto resolved = resolveCompositeType(owner, *versioned, location, true);
            if (!resolved)
            {
                return std::nullopt;
            }

            auto* resolvedDef = analyzeOne(resolved->first);
            if (!resolvedDef || !*resolvedDef)
            {
                diagnostics_.error(location, "failed to analyze dependent type: " + resolved->second);
                return std::nullopt;
            }
            const auto& def = **resolvedDef;
            if (def.isService)
            {
                return unsupported("service types do not expose value attributes: " + resolved->second);
            }

            const auto& sec = def.request;
            if (attributeName == "_extent_")
            {
                return Value{Rational(sec.extentBits.value_or(sec.offsetAtEnd.max()), 1)};
            }

            for (const auto& constant : sec.constants)
            {
                if (constant.name == attributeName)
                {
                    return constant.value;
                }
            }

            return unsupported("unsupported metaserializable attribute: " + attributeName);
        };

        auto computeUnionOffsetFromSeenFields = [&]() -> BitLengthSet {
            if (unionFields.empty())
            {
                return BitLengthSet(0);
            }

            const auto b       = ceilLog2(static_cast<std::int64_t>(unionFields.size()));
            const auto tagBits = pow2ceil(std::max<std::int64_t>(8, b));

            BitLengthSet payloadSet(0);
            bool         first = true;
            for (const auto& field : unionFields)
            {
                if (first)
                {
                    payloadSet = field.bls;
                    first      = false;
                }
                else
                {
                    payloadSet = payloadSet | field.bls;
                }
            }
            return (BitLengthSet(tagBits) + payloadSet).padToAlignment(8);
        };

        auto updateOffsetEnv = [&]() {
            if (section.isUnion)
            {
                env["_offset_"] = Value{bitLengthSetToValueSet(computeUnionOffsetFromSeenFields())};
            }
            else
            {
                env["_offset_"] = Value{bitLengthSetToValueSet(structureOffset)};
            }
        };

        updateOffsetEnv();

        std::set<std::string> names;

        for (const auto& stmt : statements)
        {
            updateOffsetEnv();

            if (std::holds_alternative<DirectiveAST>(stmt))
            {
                const auto& d = std::get<DirectiveAST>(stmt);

                if (d.kind == DirectiveKind::Unknown)
                {
                    diagnostics_.error(d.location, "unknown directive @" + d.rawName);
                    continue;
                }

                if (d.kind == DirectiveKind::Union)
                {
                    if (d.expression)
                    {
                        diagnostics_.error(d.location, "@union does not accept an expression");
                    }
                    if (section.isUnion)
                    {
                        diagnostics_.error(d.location, "duplicated @union directive");
                    }
                    if (attributesStarted)
                    {
                        diagnostics_.error(d.location, "@union must appear before first attribute");
                    }
                    section.isUnion = true;
                    continue;
                }

                if (d.kind == DirectiveKind::Deprecated)
                {
                    if (d.expression)
                    {
                        diagnostics_.error(d.location, "@deprecated does not accept an expression");
                    }
                    if (isResponse)
                    {
                        diagnostics_.error(d.location, "@deprecated is only valid in service request section");
                    }
                    if (section.deprecated)
                    {
                        diagnostics_.error(d.location, "duplicated @deprecated directive");
                    }
                    if (attributesStarted)
                    {
                        diagnostics_.error(d.location, "@deprecated must be placed before first attribute");
                    }
                    section.deprecated = true;
                    continue;
                }

                if (d.kind == DirectiveKind::Extent)
                {
                    if (serializationModeSet)
                    {
                        diagnostics_.error(d.location, "serialization mode already set before @extent");
                        continue;
                    }
                    if (!d.expression)
                    {
                        diagnostics_.error(d.location, "@extent requires an expression");
                        continue;
                    }
                    if (exprContainsOffset(d.expression) && section.isUnion)
                    {
                        unionOffsetUsed = true;
                    }
                    auto v = evaluateExpression(*d.expression, env, diagnostics_, d.location, &typeAttrResolver);
                    if (!v || !std::holds_alternative<Rational>(v->data) || !std::get<Rational>(v->data).isInteger())
                    {
                        diagnostics_.error(d.location, "@extent expression must evaluate to integer rational");
                        continue;
                    }
                    const auto extent = std::get<Rational>(v->data).asInteger().value();
                    if (extent < 0)
                    {
                        diagnostics_.error(d.location, "@extent cannot be negative");
                    }
                    section.extentBits   = extent;
                    section.sealed       = false;
                    serializationModeSet = true;
                    continue;
                }

                if (d.kind == DirectiveKind::Sealed)
                {
                    if (serializationModeSet)
                    {
                        diagnostics_.error(d.location, "serialization mode already set before @sealed");
                        continue;
                    }
                    if (d.expression)
                    {
                        diagnostics_.error(d.location, "@sealed does not accept an expression");
                    }
                    section.sealed       = true;
                    serializationModeSet = true;
                    continue;
                }

                if (d.kind == DirectiveKind::Assert || d.kind == DirectiveKind::Print)
                {
                    if (d.expression && exprContainsOffset(d.expression) && section.isUnion)
                    {
                        unionOffsetUsed = true;
                    }
                    if (!d.expression && d.kind == DirectiveKind::Assert)
                    {
                        diagnostics_.error(d.location, "@assert requires an expression");
                        continue;
                    }
                    if (d.expression)
                    {
                        auto value =
                            evaluateExpression(*d.expression, env, diagnostics_, d.location, &typeAttrResolver);
                        if (d.kind == DirectiveKind::Assert)
                        {
                            if (!value || !std::holds_alternative<bool>(value->data))
                            {
                                diagnostics_.error(d.location, "@assert expression must evaluate to bool");
                            }
                            else if (!std::get<bool>(value->data))
                            {
                                diagnostics_.error(d.location, "assertion failed");
                            }
                        }
                    }
                    continue;
                }
            }

            if (std::holds_alternative<ConstantDeclAST>(stmt))
            {
                const auto& c     = std::get<ConstantDeclAST>(stmt);
                attributesStarted = true;
                if (serializationModeSet)
                {
                    diagnostics_.error(c.location, "attributes cannot follow @extent/@sealed");
                }
                if (names.contains(c.name))
                {
                    diagnostics_.error(c.location, "duplicate attribute name: " + c.name);
                    continue;
                }

                auto value = evaluateExpression(*c.value, env, diagnostics_, c.location, &typeAttrResolver);
                if (!value)
                {
                    continue;
                }
                if (!checkConstantCompatibility(c, *value))
                {
                    continue;
                }

                names.insert(c.name);
                section.constants.push_back(SemanticConstant{c.name, c.type, *value});
                env[c.name] = *value;
                continue;
            }

            if (std::holds_alternative<FieldDeclAST>(stmt))
            {
                const auto& f     = std::get<FieldDeclAST>(stmt);
                attributesStarted = true;
                if (serializationModeSet)
                {
                    diagnostics_.error(f.location, "attributes cannot follow @extent/@sealed");
                }

                if (!f.isPadding && names.contains(f.name))
                {
                    diagnostics_.error(f.location, "duplicate attribute name: " + f.name);
                    continue;
                }
                if (!f.isPadding)
                {
                    names.insert(f.name);
                }

                if (section.isUnion && f.isPadding)
                {
                    diagnostics_.error(f.location, "tagged unions cannot contain padding fields");
                    continue;
                }
                if (section.isUnion && unionOffsetUsed)
                {
                    diagnostics_.error(f.location, "field after _offset_ usage in union is not allowed by spec");
                }

                auto          layout = analyzeType(owner, f.type, env, &typeAttrResolver);
                SemanticField semanticField;
                semanticField.name         = f.name;
                semanticField.type         = f.type;
                semanticField.isPadding    = f.isPadding;
                semanticField.resolvedType = layout.resolved;
                semanticField.sectionName  = isResponse ? "response" : "request";
                if (section.isUnion && !f.isPadding)
                {
                    semanticField.unionOptionIndex = unionOptionCounter++;
                }
                section.fields.push_back(std::move(semanticField));

                if (section.isUnion)
                {
                    unionFields.push_back(layout);
                }
                else
                {
                    structureOffset = structureOffset.padToAlignment(layout.alignment);
                    structureOffset = structureOffset + layout.bls;
                }

                continue;
            }
        }

        if (section.isUnion)
        {
            if (unionFields.size() < 2)
            {
                diagnostics_.error(owner.filePath.empty() ? SourceLocation{"<unknown>", 1, 1}
                                                          : SourceLocation{owner.filePath, 1, 1},
                                   "tagged unions must contain at least two fields");
            }

            section.offsetAtEnd     = computeUnionOffsetFromSeenFields();
            const auto b            = ceilLog2(static_cast<std::int64_t>(unionFields.size()));
            const auto unionTagBits = static_cast<std::uint32_t>(pow2ceil(std::max<std::int64_t>(8, b)));
            for (auto& field : section.fields)
            {
                if (!field.isPadding)
                {
                    field.unionTagBits = unionTagBits;
                }
            }
        }
        else
        {
            section.offsetAtEnd = structureOffset.padToAlignment(8);
        }

        if (!serializationModeSet)
        {
            diagnostics_.error(owner.filePath.empty() ? SourceLocation{"<unknown>", 1, 1}
                                                      : SourceLocation{owner.filePath, 1, 1},
                               "either @sealed or @extent are required");
        }

        if (!section.sealed)
        {
            const auto extent = section.extentBits.value_or(-1);
            if (extent < 0)
            {
                diagnostics_.error(owner.filePath.empty() ? SourceLocation{"<unknown>", 1, 1}
                                                          : SourceLocation{owner.filePath, 1, 1},
                                   "@extent must define non-negative extent bits");
            }
            if (extent % 8 != 0)
            {
                diagnostics_.error(owner.filePath.empty() ? SourceLocation{"<unknown>", 1, 1}
                                                          : SourceLocation{owner.filePath, 1, 1},
                                   "extent must be a multiple of 8 bits");
            }
            if (extent < section.offsetAtEnd.max())
            {
                diagnostics_.error(owner.filePath.empty() ? SourceLocation{"<unknown>", 1, 1}
                                                          : SourceLocation{owner.filePath, 1, 1},
                                   "extent smaller than maximal serialized length");
            }
        }
        else
        {
            section.extentBits = section.offsetAtEnd.max();
        }

        section.minBitLength                = section.offsetAtEnd.min();
        section.maxBitLength                = section.offsetAtEnd.max();
        section.fixedSize                   = section.offsetAtEnd.fixed();
        section.serializationBufferSizeBits = section.maxBitLength;

        return section;
    }

    void checkVersionRules()
    {
        struct GroupKey
        {
            std::string   name;
            std::uint32_t major{0};

            auto operator<=>(const GroupKey&) const = default;
        };

        std::map<GroupKey, std::vector<const SemanticDefinition*>> groups;
        for (const auto& r : results_)
        {
            if (!r)
            {
                continue;
            }
            groups[{r->info.fullName, r->info.majorVersion}].push_back(&*r);
        }

        for (auto& [_, defs] : groups)
        {
            std::sort(defs.begin(), defs.end(), [](const auto* a, const auto* b) {
                return a->info.minorVersion < b->info.minorVersion;
            });

            if (defs.empty())
            {
                continue;
            }
            if (defs.front()->info.majorVersion > 0)
            {
                const bool sealed = defs.front()->request.sealed;
                const auto extent = defs.front()->request.extentBits.value_or(0);

                for (const auto* d : defs)
                {
                    if (d->request.sealed != sealed)
                    {
                        diagnostics_.error({d->info.filePath, 1, 1},
                                           "all minor versions under same major must share sealing status");
                    }
                    if (d->request.extentBits.value_or(0) != extent)
                    {
                        diagnostics_.error({d->info.filePath, 1, 1},
                                           "all minor versions under same major must share extent");
                    }
                }
            }

            std::optional<std::uint32_t> lastFixed;
            for (const auto* d : defs)
            {
                if (!d->info.fixedPortId && lastFixed)
                {
                    diagnostics_.error({d->info.filePath, 1, 1},
                                       "fixed port-ID cannot be removed under same major version");
                }
                if (d->info.fixedPortId)
                {
                    if (lastFixed && *lastFixed != *d->info.fixedPortId)
                    {
                        diagnostics_.error({d->info.filePath, 1, 1}, "fixed port-ID mismatch under same major version");
                    }
                    lastFixed = d->info.fixedPortId;
                }
            }
        }

        // Fixed port-ID collisions across unrelated types of the same kind.
        for (std::size_t i = 0; i < results_.size(); ++i)
        {
            if (!results_[i] || !results_[i]->info.fixedPortId)
            {
                continue;
            }
            for (std::size_t j = i + 1; j < results_.size(); ++j)
            {
                if (!results_[j] || !results_[j]->info.fixedPortId)
                {
                    continue;
                }
                if (results_[i]->isService != results_[j]->isService)
                {
                    continue;
                }
                if (results_[i]->info.fullName == results_[j]->info.fullName &&
                    results_[i]->info.majorVersion == results_[j]->info.majorVersion)
                {
                    continue;
                }
                if (*results_[i]->info.fixedPortId == *results_[j]->info.fixedPortId &&
                    results_[i]->info.majorVersion > 0 && results_[j]->info.majorVersion > 0)
                {
                    diagnostics_.error({results_[j]->info.filePath, 1, 1},
                                       "fixed port-ID collision with " + results_[i]->info.filePath);
                }
            }
        }
    }
};

}  // namespace

llvm::Expected<SemanticModule> analyze(const ASTModule& module, DiagnosticEngine& diagnostics)
{
    AnalyzerImpl impl(module, diagnostics);
    return impl.run();
}

}  // namespace llvmdsdl
