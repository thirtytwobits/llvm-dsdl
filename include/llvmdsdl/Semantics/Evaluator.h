//===----------------------------------------------------------------------===//
///
/// @file
/// Constant-expression evaluator declarations used during semantic analysis.
///
//===----------------------------------------------------------------------===//
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

namespace llvmdsdl
{

/// @file
/// @brief Constant-expression evaluator interfaces.

/// @brief Runtime value representation for semantic constant evaluation.
struct Value final
{
    /// @brief Set literal payload type.
    using Set = std::set<Rational>;

    /// @brief Value payload variant.
    std::variant<bool, Rational, std::string, Set, TypeExprAST> data;

    /// @brief Returns a stable textual type name for the active variant.
    /// @return Value type name string.
    [[nodiscard]] std::string typeName() const;

    /// @brief Returns a human-readable value representation.
    /// @return Value string.
    [[nodiscard]] std::string str() const;
};

/// @brief Symbol table used during expression evaluation.
using ValueEnv = std::map<std::string, Value, std::less<>>;

/// @brief Callback for resolving type-attribute lookups.
using TypeAttributeResolver =
    std::function<std::optional<Value>(const TypeExprAST&, const std::string&, const SourceLocation&)>;

/// @brief Evaluates a constant expression.
/// @param[in] expr Expression to evaluate.
/// @param[in] env Immutable symbol environment.
/// @param[in,out] diagnostics Diagnostic sink for evaluation failures.
/// @param[in] location Fallback source location for diagnostics.
/// @param[in] strictMode Enables strict semantic diagnostics.
/// @param[in] resolver Optional type-attribute resolver.
/// @return Evaluated value, or `std::nullopt` on failure.
std::optional<Value> evaluateExpression(const ExprAST&               expr,
                                        const ValueEnv&              env,
                                        DiagnosticEngine&            diagnostics,
                                        const SourceLocation&        location,
                                        bool                         strictMode,
                                        const TypeAttributeResolver* resolver = nullptr);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_SEMANTICS_EVALUATOR_H
