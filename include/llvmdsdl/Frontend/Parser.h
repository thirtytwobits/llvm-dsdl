#ifndef LLVMDSDL_FRONTEND_PARSER_H
#define LLVMDSDL_FRONTEND_PARSER_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Frontend/Lexer.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace llvmdsdl {

class Parser final {
public:
  Parser(std::string filePath, std::vector<Token> tokens,
         DiagnosticEngine &diagnostics);

  llvm::Expected<DefinitionAST> parseDefinition();

private:
  const Token &current() const;
  const Token &previous() const;
  bool isAtEnd() const;
  bool check(TokenKind kind) const;
  bool match(TokenKind kind);
  bool matchAny(std::initializer_list<TokenKind> kinds);
  const Token &advance();
  bool expect(TokenKind kind, const std::string &message);
  void syncToNextLine();

  std::optional<StatementAST> parseStatement();
  std::optional<DirectiveAST> parseDirective();
  std::optional<StatementAST> parseAttribute();
  std::optional<TypeExprAST> parseTypeExpr(bool silent = false);

  std::shared_ptr<ExprAST> parseExpression(int precedence = 0);
  std::shared_ptr<ExprAST> parseUnary();
  std::shared_ptr<ExprAST> parsePrimary();
  std::shared_ptr<ExprAST> parseSetLiteral(const SourceLocation &location);

  static int precedenceOf(TokenKind kind);
  static bool isRightAssociative(TokenKind kind);
  static std::optional<BinaryOp> toBinaryOp(TokenKind kind);

  std::string filePath_;
  std::vector<Token> tokens_;
  std::size_t cursor_{0};
  DiagnosticEngine &diagnostics_;
};

llvm::Expected<ASTModule>
parseDefinitions(const std::vector<std::string> &rootNamespaceDirs,
                 const std::vector<std::string> &lookupDirs,
                 DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_PARSER_H
