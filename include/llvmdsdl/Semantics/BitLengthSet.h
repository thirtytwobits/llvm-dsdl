#ifndef LLVMDSDL_SEMANTICS_BITLENGTHSET_H
#define LLVMDSDL_SEMANTICS_BITLENGTHSET_H

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace llvmdsdl {

class BitLengthSet final {
public:
  BitLengthSet();
  explicit BitLengthSet(std::int64_t value);
  explicit BitLengthSet(std::set<std::int64_t> values);

  [[nodiscard]] std::int64_t min() const;
  [[nodiscard]] std::int64_t max() const;
  [[nodiscard]] bool fixed() const;

  [[nodiscard]] BitLengthSet padToAlignment(std::int64_t alignment) const;
  [[nodiscard]] BitLengthSet repeat(std::int64_t count) const;
  [[nodiscard]] BitLengthSet repeatRange(std::int64_t countMax) const;

  [[nodiscard]] std::set<std::int64_t> modulo(std::int64_t divisor) const;

  [[nodiscard]] std::set<std::int64_t> expand(std::size_t limit = 16384) const;

  [[nodiscard]] std::string str() const;

  friend BitLengthSet operator+(const BitLengthSet &lhs,
                                const BitLengthSet &rhs);
  friend BitLengthSet operator|(const BitLengthSet &lhs,
                                const BitLengthSet &rhs);

private:
  struct Node;
  explicit BitLengthSet(std::shared_ptr<const Node> root);

  std::shared_ptr<const Node> root_;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_SEMANTICS_BITLENGTHSET_H
