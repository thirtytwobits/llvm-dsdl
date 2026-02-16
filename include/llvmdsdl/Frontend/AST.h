#ifndef LLVMDSDL_FRONTEND_AST_H
#define LLVMDSDL_FRONTEND_AST_H

#include "llvmdsdl/Frontend/SourceLocation.h"
#include "llvmdsdl/Support/Rational.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace llvmdsdl {

enum class CastMode {
  Saturated,
  Truncated,
};

enum class PrimitiveKind {
  Bool,
  Byte,
  Utf8,
  UnsignedInt,
  SignedInt,
  Float,
};

enum class ArrayKind {
  None,
  Fixed,
  VariableInclusive,
  VariableExclusive,
};

struct TypeExprAST;
struct ExprAST;

struct PrimitiveTypeExprAST {
  PrimitiveKind kind{PrimitiveKind::Bool};
  CastMode castMode{CastMode::Saturated};
  std::uint32_t bitLength{1};
};

struct VoidTypeExprAST {
  std::uint32_t bitLength{1};
};

struct VersionedTypeExprAST {
  std::vector<std::string> nameComponents;
  std::uint32_t major{0};
  std::uint32_t minor{0};
};

struct TypeExprAST {
  SourceLocation location;
  std::variant<PrimitiveTypeExprAST, VoidTypeExprAST, VersionedTypeExprAST>
      scalar;
  ArrayKind arrayKind{ArrayKind::None};
  std::shared_ptr<ExprAST> arrayCapacity;

  [[nodiscard]] bool isVoid() const;
  [[nodiscard]] std::string str() const;
};

enum class UnaryOp {
  Plus,
  Minus,
  LogicalNot,
};

enum class BinaryOp {
  Attribute,
  Pow,
  Mul,
  Div,
  Mod,
  Add,
  Sub,
  BitOr,
  BitXor,
  BitAnd,
  Eq,
  Ne,
  Le,
  Ge,
  Lt,
  Gt,
  LogicalOr,
  LogicalAnd,
};

struct ExprAST {
  struct Identifier {
    std::string name;
  };

  struct Unary {
    UnaryOp op;
    std::shared_ptr<ExprAST> operand;
  };

  struct Binary {
    BinaryOp op;
    std::shared_ptr<ExprAST> lhs;
    std::shared_ptr<ExprAST> rhs;
  };

  struct SetLiteral {
    std::vector<std::shared_ptr<ExprAST>> elements;
  };

  struct TypeLiteral {
    TypeExprAST type;
  };

  SourceLocation location;
  std::variant<bool, Rational, std::string, Identifier, Unary, Binary,
               SetLiteral, TypeLiteral>
      value;

  [[nodiscard]] std::string str() const;
};

enum class DirectiveKind {
  Union,
  Extent,
  Sealed,
  Deprecated,
  Assert,
  Print,
  Unknown,
};

struct ConstantDeclAST {
  SourceLocation location;
  TypeExprAST type;
  std::string name;
  std::shared_ptr<ExprAST> value;
};

struct FieldDeclAST {
  SourceLocation location;
  TypeExprAST type;
  std::string name;
  bool isPadding{false};
};

struct DirectiveAST {
  SourceLocation location;
  DirectiveKind kind{DirectiveKind::Unknown};
  std::string rawName;
  std::shared_ptr<ExprAST> expression;
};

struct ServiceResponseMarkerAST {
  SourceLocation location;
};

using StatementAST =
    std::variant<ConstantDeclAST, FieldDeclAST, DirectiveAST,
                 ServiceResponseMarkerAST>;

struct DefinitionAST {
  SourceLocation location;
  std::vector<StatementAST> statements;

  [[nodiscard]] bool isService() const;
};

struct DiscoveredDefinition {
  std::string filePath;
  std::string rootNamespacePath;
  std::string fullName;
  std::string shortName;
  std::vector<std::string> namespaceComponents;
  std::uint32_t majorVersion{0};
  std::uint32_t minorVersion{0};
  std::optional<std::uint32_t> fixedPortId;
  std::string text;
};

struct ParsedDefinition {
  DiscoveredDefinition info;
  DefinitionAST ast;
};

struct ASTModule {
  std::vector<ParsedDefinition> definitions;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_AST_H
