#ifndef LLVMDSDL_SUPPORT_RATIONAL_H
#define LLVMDSDL_SUPPORT_RATIONAL_H

#include <cstdint>
#include <optional>
#include <string>

namespace llvmdsdl {

class Rational final {
public:
  Rational();
  Rational(std::int64_t numerator, std::int64_t denominator = 1);

  [[nodiscard]] std::int64_t numerator() const { return numerator_; }
  [[nodiscard]] std::int64_t denominator() const { return denominator_; }
  [[nodiscard]] bool isInteger() const { return denominator_ == 1; }

  [[nodiscard]] std::optional<std::int64_t> asInteger() const;
  [[nodiscard]] std::string str() const;

  friend Rational operator+(const Rational &lhs, const Rational &rhs);
  friend Rational operator-(const Rational &lhs, const Rational &rhs);
  friend Rational operator*(const Rational &lhs, const Rational &rhs);
  friend Rational operator/(const Rational &lhs, const Rational &rhs);

  friend bool operator==(const Rational &lhs, const Rational &rhs);
  friend bool operator!=(const Rational &lhs, const Rational &rhs);
  friend bool operator<(const Rational &lhs, const Rational &rhs);
  friend bool operator<=(const Rational &lhs, const Rational &rhs);
  friend bool operator>(const Rational &lhs, const Rational &rhs);
  friend bool operator>=(const Rational &lhs, const Rational &rhs);

private:
  static std::int64_t gcd(std::int64_t a, std::int64_t b);
  void normalize();

  std::int64_t numerator_;
  std::int64_t denominator_;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_SUPPORT_RATIONAL_H
