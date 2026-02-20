//===----------------------------------------------------------------------===//
///
/// @file
/// Implements normalized rational arithmetic utilities.
///
/// Rational helpers provide stable arithmetic primitives used by expression evaluation and semantic calculations.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/Support/Rational.h"

#include <cstdlib>
#include <sstream>

namespace llvmdsdl
{

Rational::Rational()
    : numerator_(0)
    , denominator_(1)
{
}

Rational::Rational(std::int64_t numerator, std::int64_t denominator)
    : numerator_(numerator)
    , denominator_(denominator == 0 ? 1 : denominator)
{
    normalize();
}

std::optional<std::int64_t> Rational::asInteger() const
{
    if (!isInteger())
    {
        return std::nullopt;
    }
    return numerator_;
}

std::string Rational::str() const
{
    std::ostringstream out;
    out << numerator_;
    if (denominator_ != 1)
    {
        out << '/' << denominator_;
    }
    return out.str();
}

Rational operator+(const Rational& lhs, const Rational& rhs)
{
    return Rational(lhs.numerator_ * rhs.denominator_ + rhs.numerator_ * lhs.denominator_,
                    lhs.denominator_ * rhs.denominator_);
}

Rational operator-(const Rational& lhs, const Rational& rhs)
{
    return Rational(lhs.numerator_ * rhs.denominator_ - rhs.numerator_ * lhs.denominator_,
                    lhs.denominator_ * rhs.denominator_);
}

Rational operator*(const Rational& lhs, const Rational& rhs)
{
    return Rational(lhs.numerator_ * rhs.numerator_, lhs.denominator_ * rhs.denominator_);
}

Rational operator/(const Rational& lhs, const Rational& rhs)
{
    if (rhs.numerator_ == 0)
    {
        return Rational(0, 1);
    }
    return Rational(lhs.numerator_ * rhs.denominator_, lhs.denominator_ * rhs.numerator_);
}

bool operator==(const Rational& lhs, const Rational& rhs)
{
    return lhs.numerator_ == rhs.numerator_ && lhs.denominator_ == rhs.denominator_;
}

bool operator!=(const Rational& lhs, const Rational& rhs)
{
    return !(lhs == rhs);
}

bool operator<(const Rational& lhs, const Rational& rhs)
{
    return lhs.numerator_ * rhs.denominator_ < rhs.numerator_ * lhs.denominator_;
}

bool operator<=(const Rational& lhs, const Rational& rhs)
{
    return lhs < rhs || lhs == rhs;
}

bool operator>(const Rational& lhs, const Rational& rhs)
{
    return rhs < lhs;
}

bool operator>=(const Rational& lhs, const Rational& rhs)
{
    return rhs <= lhs;
}

std::int64_t Rational::gcd(std::int64_t a, std::int64_t b)
{
    a = std::llabs(a);
    b = std::llabs(b);
    while (b != 0)
    {
        const auto t = a % b;
        a            = b;
        b            = t;
    }
    return a == 0 ? 1 : a;
}

void Rational::normalize()
{
    if (denominator_ == 0)
    {
        denominator_ = 1;
        numerator_   = 0;
        return;
    }
    if (denominator_ < 0)
    {
        denominator_ = -denominator_;
        numerator_   = -numerator_;
    }
    const auto div = gcd(numerator_, denominator_);
    numerator_ /= div;
    denominator_ /= div;
}

}  // namespace llvmdsdl
