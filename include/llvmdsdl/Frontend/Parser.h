//===----------------------------------------------------------------------===//
///
/// @file
/// Parser declarations for constructing AST definitions from token streams.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_FRONTEND_PARSER_H
#define LLVMDSDL_FRONTEND_PARSER_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Frontend/Lexer.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvmdsdl
{

class DiagnosticEngine;
struct SourceLocation;

/// @file
/// @brief Recursive-descent parser interfaces.

/// @brief Parses a token stream into a DSDL definition AST.
class Parser final
{
public:
    /// @brief Constructs a parser for one definition file.
    /// @param[in] filePath Source file path for diagnostics.
    /// @param[in] tokens Token stream to parse.
    /// @param[in,out] diagnostics Diagnostic sink for parse errors.
    Parser(std::string filePath, std::vector<Token> tokens, DiagnosticEngine& diagnostics);

    /// @brief Parses the token stream as one definition body.
    /// @return Parsed definition AST or an error on unrecoverable failure.
    llvm::Expected<DefinitionAST> parseDefinition();

private:
    /// @brief Returns the current token.
    const Token& current() const;

    /// @brief Returns the previous token.
    const Token& previous() const;

    /// @brief Returns true when parser reached EOF.
    bool isAtEnd() const;

    /// @brief Returns true when the current token matches `kind`.
    bool check(TokenKind kind) const;

    /// @brief Consumes current token when it matches `kind`.
    bool match(TokenKind kind);

    /// @brief Consumes current token when it matches any candidate kind.
    bool matchAny(std::initializer_list<TokenKind> kinds);

    /// @brief Consumes and returns the current token.
    const Token& advance();

    /// @brief Enforces the next token kind and emits diagnostics on mismatch.
    bool expect(TokenKind kind, const std::string& message);

    /// @brief Error recovery that advances to the next line boundary.
    void syncToNextLine();

    /// @brief Parses one top-level statement.
    std::optional<StatementAST> parseStatement();

    /// @brief Parses one directive statement.
    std::optional<DirectiveAST> parseDirective();

    /// @brief Parses one attribute-like statement.
    std::optional<StatementAST> parseAttribute();

    /// @brief Parses a type expression.
    std::optional<TypeExprAST> parseTypeExpr(bool silent = false);

    /// @brief Parses an expression using precedence climbing.
    std::shared_ptr<ExprAST> parseExpression(int precedence = 0);

    /// @brief Parses a unary expression.
    std::shared_ptr<ExprAST> parseUnary();

    /// @brief Parses a primary expression.
    std::shared_ptr<ExprAST> parsePrimary();

    /// @brief Parses a set literal expression.
    std::shared_ptr<ExprAST> parseSetLiteral(const SourceLocation& location);

    /// @brief Returns precedence for a token kind.
    static int precedenceOf(TokenKind kind);

    /// @brief Returns true for right-associative operators.
    static bool isRightAssociative(TokenKind kind);

    /// @brief Maps token kind to binary operator enum.
    static std::optional<BinaryOp> toBinaryOp(TokenKind kind);

    /// @brief Source file path for diagnostics.
    std::string filePath_;

    /// @brief Token stream being parsed.
    std::vector<Token> tokens_;

    /// @brief Current token index.
    std::size_t cursor_{0};

    /// @brief Diagnostic sink.
    DiagnosticEngine& diagnostics_;
};

/// @brief Discovers, lexes, and parses all definitions from namespace roots.
/// @param[in] rootNamespaceDirs Root namespace directories.
/// @param[in] lookupDirs Additional lookup paths.
/// @param[in,out] diagnostics Diagnostic sink for discovery/parse issues.
/// @return Parsed AST module or an error on unrecoverable failure.
llvm::Expected<ASTModule> parseDefinitions(const std::vector<std::string>& rootNamespaceDirs,
                                           const std::vector<std::string>& lookupDirs,
                                           DiagnosticEngine&               diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_FRONTEND_PARSER_H
