#include "llvmdsdl/Frontend/Lexer.h"
#include "llvmdsdl/Frontend/Parser.h"
#include "llvmdsdl/Semantics/Evaluator.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include <iostream>
#include <string_view>

namespace
{

std::shared_ptr<llvmdsdl::ExprAST> parseAssertExpression(const std::string&          expression,
                                                         llvmdsdl::DiagnosticEngine& diag)
{
    const std::string text = "@assert " + expression + "\n@sealed\n";
    llvmdsdl::Lexer   lexer("evaluator_test.dsdl", text);
    auto              tokens = lexer.lex();
    llvmdsdl::Parser  parser("evaluator_test.dsdl", std::move(tokens), diag);
    auto              def = parser.parseDefinition();
    if (!def || def->statements.empty())
    {
        return nullptr;
    }
    const auto* directive = std::get_if<llvmdsdl::DirectiveAST>(&def->statements[0]);
    if (!directive)
    {
        return nullptr;
    }
    return directive->expression;
}

bool hasErrorContaining(const llvmdsdl::DiagnosticEngine& diag, std::string_view needle)
{
    for (const auto& d : diag.diagnostics())
    {
        if (d.level == llvmdsdl::DiagnosticLevel::Error && d.message.find(needle) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

}  // namespace

bool runEvaluatorTests()
{
    {
        llvmdsdl::DiagnosticEngine diag;
        auto                       expr = parseAssertExpression("Foo.1.0.MAX", diag);
        if (!expr)
        {
            std::cerr << "failed to parse evaluator expression\n";
            return false;
        }

        llvmdsdl::ValueEnv env;
        auto               value = llvmdsdl::evaluateExpression(*expr,
                                                  env,
                                                  diag,
                                                  expr->location,
                                                  nullptr);
        if (value)
        {
            std::cerr << "expected evaluation without resolver to fail\n";
            return false;
        }
        if (!hasErrorContaining(diag, "unsupported metaserializable attribute: MAX"))
        {
            std::cerr << "missing unsupported-attribute diagnostic\n";
            return false;
        }
    }

    {
        llvmdsdl::DiagnosticEngine diag;
        auto                       expr = parseAssertExpression("Foo.1.0.MAX", diag);
        if (!expr)
        {
            std::cerr << "failed to parse resolver evaluator expression\n";
            return false;
        }

        bool                            resolverCalled = false;
        llvmdsdl::TypeAttributeResolver resolver =
            [&](const llvmdsdl::TypeExprAST& type,
                const std::string&           attribute,
                const llvmdsdl::SourceLocation&) -> std::optional<llvmdsdl::Value> {
            resolverCalled        = true;
            const auto* versioned = std::get_if<llvmdsdl::VersionedTypeExprAST>(&type.scalar);
            if (!versioned || attribute != "MAX")
            {
                return std::nullopt;
            }
            return llvmdsdl::Value{llvmdsdl::Rational(42, 1)};
        };

        llvmdsdl::ValueEnv env;
        auto               value = llvmdsdl::evaluateExpression(*expr,
                                                  env,
                                                  diag,
                                                  expr->location,
                                                  &resolver);
        if (!value || !std::holds_alternative<llvmdsdl::Rational>(value->data) ||
            std::get<llvmdsdl::Rational>(value->data) != llvmdsdl::Rational(42, 1))
        {
            std::cerr << "resolver-based evaluation produced unexpected result\n";
            return false;
        }
        if (!resolverCalled)
        {
            std::cerr << "resolver was not invoked for type attribute\n";
            return false;
        }
    }

    {
        llvmdsdl::DiagnosticEngine diag;
        auto                       expr = parseAssertExpression("Foo.1.0._extent_", diag);
        if (!expr)
        {
            std::cerr << "failed to parse _extent_ evaluator expression\n";
            return false;
        }

        llvmdsdl::TypeAttributeResolver resolver =
            [&](const llvmdsdl::TypeExprAST&,
                const std::string& attribute,
                const llvmdsdl::SourceLocation&) -> std::optional<llvmdsdl::Value> {
            if (attribute == "_extent_")
            {
                return llvmdsdl::Value{llvmdsdl::Rational(128, 1)};
            }
            return std::nullopt;
        };

        llvmdsdl::ValueEnv env;
        auto               value = llvmdsdl::evaluateExpression(*expr,
                                                  env,
                                                  diag,
                                                  expr->location,
                                                  &resolver);
        if (!value || !std::holds_alternative<llvmdsdl::Rational>(value->data) ||
            std::get<llvmdsdl::Rational>(value->data) != llvmdsdl::Rational(128, 1))
        {
            std::cerr << "_extent_ resolver evaluation produced unexpected result\n";
            return false;
        }
    }

    return true;
}
