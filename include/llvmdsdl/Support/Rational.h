//===----------------------------------------------------------------------===//
///
/// @file
/// Declarations for exact rational arithmetic used by compile-time constant evaluation.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_SUPPORT_RATIONAL_H
#define LLVMDSDL_SUPPORT_RATIONAL_H

#include <cstdint>
#include <optional>
#include <string>

namespace llvmdsdl
{

/// @file
/// @brief Exact rational number type used by constant expression evaluation.

/// @brief Immutable normalized rational value.
class Rational final
{
public:
    /// @brief Constructs zero (`0/1`).
    Rational();

    /// @brief Constructs and normalizes a rational value.
    /// @param[in] numerator Numerator value.
    /// @param[in] denominator Denominator value (must not be zero).
    Rational(std::int64_t numerator, std::int64_t denominator = 1);

    /// @brief Returns the normalized numerator.
    [[nodiscard]] std::int64_t numerator() const
    {
        return numerator_;
    }

    /// @brief Returns the normalized denominator.
    [[nodiscard]] std::int64_t denominator() const
    {
        return denominator_;
    }

    /// @brief Returns true when the value is an integer.
    [[nodiscard]] bool isInteger() const
    {
        return denominator_ == 1;
    }

    /// @brief Converts to integer when the value is integral.
    /// @return Integer value or `std::nullopt` for non-integral rationals.
    [[nodiscard]] std::optional<std::int64_t> asInteger() const;

    /// @brief Returns a human-readable textual representation.
    /// @return String representation of the rational.
    [[nodiscard]] std::string str() const;

    /// @brief Adds two rationals.
    friend Rational operator+(const Rational& lhs, const Rational& rhs);

    /// @brief Subtracts two rationals.
    friend Rational operator-(const Rational& lhs, const Rational& rhs);

    /// @brief Multiplies two rationals.
    friend Rational operator*(const Rational& lhs, const Rational& rhs);

    /// @brief Divides two rationals.
    friend Rational operator/(const Rational& lhs, const Rational& rhs);

    /// @brief Equality comparison.
    friend bool operator==(const Rational& lhs, const Rational& rhs);

    /// @brief Inequality comparison.
    friend bool operator!=(const Rational& lhs, const Rational& rhs);

    /// @brief Strict weak ordering.
    friend bool operator<(const Rational& lhs, const Rational& rhs);

    /// @brief Less-or-equal comparison.
    friend bool operator<=(const Rational& lhs, const Rational& rhs);

    /// @brief Greater-than comparison.
    friend bool operator>(const Rational& lhs, const Rational& rhs);

    /// @brief Greater-or-equal comparison.
    friend bool operator>=(const Rational& lhs, const Rational& rhs);

private:
    /// @brief Computes greatest common divisor.
    static std::int64_t gcd(std::int64_t a, std::int64_t b);

    /// @brief Reduces sign and fraction by GCD.
    void normalize();

    /// @brief Normalized numerator.
    std::int64_t numerator_;

    /// @brief Normalized denominator (always positive).
    std::int64_t denominator_;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_SUPPORT_RATIONAL_H
