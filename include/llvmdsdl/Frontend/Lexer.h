#ifndef LLVMDSDL_FRONTEND_LEXER_H
#define LLVMDSDL_FRONTEND_LEXER_H

#include "llvmdsdl/Frontend/SourceLocation.h"

#include <string>
#include <vector>

namespace llvmdsdl {

enum class TokenKind {
  Eof,
  Newline,
  Identifier,
  Integer,
  Real,
  String,
  True,
  False,
  At,
  ServiceResponseMarker,
  LParen,
  RParen,
  LBracket,
  RBracket,
  LBrace,
  RBrace,
  Comma,
  Dot,
  Equal,
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
  Pipe,
  Caret,
  Amp,
  Bang,
  Less,
  Greater,
  LessEqual,
  GreaterEqual,
  EqualEqual,
  BangEqual,
  PipePipe,
  AmpAmp,
  StarStar,
};

struct Token {
  TokenKind kind{TokenKind::Eof};
  std::string text;
  SourceLocation location;
};

class Lexer final {
public:
  Lexer(std::string file, std::string text);

  [[nodiscard]] std::vector<Token> lex();

private:
  [[nodiscard]] bool isAtEnd() const;
  [[nodiscard]] char peek(std::size_t lookahead = 0) const;
  char advance();
  void emit(TokenKind kind, std::string text, std::uint32_t line,
            std::uint32_t column);
  void lexIdentifierOrKeyword(std::uint32_t line, std::uint32_t column);
  void lexNumber(std::uint32_t line, std::uint32_t column);
  void lexString(std::uint32_t line, std::uint32_t column, char quote);

  std::string file_;
  std::string text_;
  std::size_t index_{0};
  std::uint32_t line_{1};
  std::uint32_t column_{1};
  std::vector<Token> tokens_;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_LEXER_H
