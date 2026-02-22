//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/Error.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvmdsdl/Frontend/Lexer.h"
#include "llvmdsdl/Frontend/Parser.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "llvmdsdl/Frontend/AST.h"

bool runParserTests()
{
    const std::string text = "@union\n"
                             "uint8 a\n"
                             "uint16 b\n"
                             "@extent 64\n"
                             "---\n"
                             "bool ok\n"
                             "@sealed\n";

    llvmdsdl::DiagnosticEngine diag;
    llvmdsdl::Lexer            lexer("test.dsdl", text);
    auto                       tokens = lexer.lex();

    llvmdsdl::Parser parser("test.dsdl", std::move(tokens), diag);
    auto             def = parser.parseDefinition();
    if (!def)
    {
        std::cerr << "parser failed unexpectedly\n";
        return false;
    }

    if (!def->isService())
    {
        std::cerr << "expected service definition\n";
        return false;
    }

    if (def->statements.size() != 7)
    {
        std::cerr << "unexpected statement count: " << def->statements.size() << "\n";
        return false;
    }

    const std::string typeLiteralText = "uint8[<=uavcan.file.Path.2.0.MAX_LENGTH] parameter\n"
                                        "@sealed\n";

    llvmdsdl::DiagnosticEngine typeLiteralDiag;
    llvmdsdl::Lexer            typeLiteralLexer("type_literal_expr.dsdl", typeLiteralText);
    auto                       typeLiteralTokens = typeLiteralLexer.lex();
    llvmdsdl::Parser typeLiteralParser("type_literal_expr.dsdl", std::move(typeLiteralTokens), typeLiteralDiag);
    auto             typeLiteralDef = typeLiteralParser.parseDefinition();
    if (!typeLiteralDef)
    {
        std::cerr << "type-literal parser failed unexpectedly\n";
        return false;
    }
    if (typeLiteralDef->statements.size() != 2)
    {
        std::cerr << "unexpected type-literal statement count: " << typeLiteralDef->statements.size() << "\n";
        return false;
    }

    const auto* field = std::get_if<llvmdsdl::FieldDeclAST>(&typeLiteralDef->statements[0]);
    if (!field)
    {
        std::cerr << "expected first statement to be field declaration\n";
        return false;
    }
    if (field->type.arrayKind != llvmdsdl::ArrayKind::VariableInclusive)
    {
        std::cerr << "expected variable inclusive array kind\n";
        return false;
    }
    if (!field->type.arrayCapacity)
    {
        std::cerr << "expected array capacity expression\n";
        return false;
    }

    const auto* capacityBinary = std::get_if<llvmdsdl::ExprAST::Binary>(&field->type.arrayCapacity->value);
    if (!capacityBinary || capacityBinary->op != llvmdsdl::BinaryOp::Attribute)
    {
        std::cerr << "expected attribute expression in array capacity\n";
        return false;
    }

    const auto* lhsType = std::get_if<llvmdsdl::ExprAST::TypeLiteral>(&capacityBinary->lhs->value);
    if (!lhsType)
    {
        std::cerr << "expected type literal on attribute LHS\n";
        return false;
    }
    const auto* rhsId = std::get_if<llvmdsdl::ExprAST::Identifier>(&capacityBinary->rhs->value);
    if (!rhsId || rhsId->name != "MAX_LENGTH")
    {
        std::cerr << "expected MAX_LENGTH on attribute RHS\n";
        return false;
    }

    const auto* versioned = std::get_if<llvmdsdl::VersionedTypeExprAST>(&lhsType->type.scalar);
    if (!versioned)
    {
        std::cerr << "expected versioned type literal\n";
        return false;
    }
    if (versioned->nameComponents != std::vector<std::string>{"uavcan", "file", "Path"} || versioned->major != 2 ||
        versioned->minor != 0)
    {
        std::cerr << "unexpected type literal target\n";
        return false;
    }

    const std::string commentText = "# file comment\n"
                                    "uint8 a # field comment\n"
                                    "uint16 B = 1 # constant comment\n"
                                    "@sealed # directive comment\n";
    llvmdsdl::DiagnosticEngine commentDiag;
    llvmdsdl::Lexer            commentLexer("comments.dsdl", commentText);
    auto                       commentTokens = commentLexer.lex();
    const std::size_t          commentTokenCount =
        static_cast<std::size_t>(std::count_if(commentTokens.begin(),
                                               commentTokens.end(),
                                               [](const llvmdsdl::Token& token) {
                                                   return token.kind == llvmdsdl::TokenKind::Comment;
                                               }));
    if (commentTokenCount < 3)
    {
        std::cerr << "expected lexer to preserve comment tokens\n";
        return false;
    }

    llvmdsdl::Parser commentParser("comments.dsdl", std::move(commentTokens), commentDiag);
    auto             commentDef = commentParser.parseDefinition();
    if (!commentDef)
    {
        std::cerr << "parser failed unexpectedly on comments fixture\n";
        return false;
    }
    if (commentDef->statements.size() != 3)
    {
        std::cerr << "unexpected statement count in comments fixture: " << commentDef->statements.size() << "\n";
        return false;
    }

    const auto* commentField = std::get_if<llvmdsdl::FieldDeclAST>(&commentDef->statements[0]);
    if (!commentField || commentField->nameLocation.line != 2 || commentField->nameLocation.column != 7)
    {
        std::cerr << "field symbol location was not captured correctly\n";
        return false;
    }
    const auto* commentConstant = std::get_if<llvmdsdl::ConstantDeclAST>(&commentDef->statements[1]);
    if (!commentConstant || commentConstant->nameLocation.line != 3 || commentConstant->nameLocation.column != 8)
    {
        std::cerr << "constant symbol location was not captured correctly\n";
        return false;
    }

    return true;
}
