#ifndef LLVMDSDL_SEMANTICS_EVALUATOR_H
#define LLVMDSDL_SEMANTICS_EVALUATOR_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <variant>

namespace llvmdsdl {

struct Value final {
  using Set = std::set<Rational>;
  std::variant<bool, Rational, std::string, Set, TypeExprAST> data;

  [[nodiscard]] std::string typeName() const;
  [[nodiscard]] std::string str() const;
};

using ValueEnv = std::map<std::string, Value, std::less<>>;
using TypeAttributeResolver = std::function<std::optional<Value>(
    const TypeExprAST &, const std::string &, const SourceLocation &)>;

std::optional<Value> evaluateExpression(const ExprAST &expr, const ValueEnv &env,
                                        DiagnosticEngine &diagnostics,
                                        const SourceLocation &location,
                                        bool strictMode,
                                        const TypeAttributeResolver *resolver =
                                            nullptr);

} // namespace llvmdsdl

#endif // LLVMDSDL_SEMANTICS_EVALUATOR_H
