//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Symbolic bit-length set declarations used by semantic analysis and layout reasoning.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_SEMANTICS_BITLENGTHSET_H
#define LLVMDSDL_SEMANTICS_BITLENGTHSET_H

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <cstddef>

namespace llvmdsdl
{

/// @file
/// @brief Symbolic bit-length set algebra used by semantic analysis.

/// @brief Persistent symbolic set of possible serialized bit lengths.
class BitLengthSet final
{
public:
    /// @brief Constructs an empty symbolic set.
    BitLengthSet();

    /// @brief Constructs a singleton set.
    /// @param[in] value Single bit-length value.
    explicit BitLengthSet(std::int64_t value);

    /// @brief Constructs a concrete set from expanded values.
    /// @param[in] values Explicit value set.
    explicit BitLengthSet(std::set<std::int64_t> values);

    /// @brief Returns the minimum representable value.
    /// @return Minimum bit-length value.
    [[nodiscard]] std::int64_t min() const;

    /// @brief Returns the maximum representable value.
    /// @return Maximum bit-length value.
    [[nodiscard]] std::int64_t max() const;

    /// @brief Returns true when the set contains exactly one value.
    /// @return True when fixed-size.
    [[nodiscard]] bool fixed() const;

    /// @brief Aligns each candidate length up to `alignment`.
    /// @param[in] alignment Alignment in bits.
    /// @return Aligned symbolic set.
    [[nodiscard]] BitLengthSet padToAlignment(std::int64_t alignment) const;

    /// @brief Repeats this set exactly `count` times and sums results.
    /// @param[in] count Repeat count.
    /// @return Summed symbolic set.
    [[nodiscard]] BitLengthSet repeat(std::int64_t count) const;

    /// @brief Repeats this set for counts in `[0, countMax]`.
    /// @param[in] countMax Maximum repeat count.
    /// @return Union of repeated symbolic sets.
    [[nodiscard]] BitLengthSet repeatRange(std::int64_t countMax) const;

    /// @brief Computes values modulo `divisor`.
    /// @param[in] divisor Modulo divisor.
    /// @return Expanded modulo results.
    [[nodiscard]] std::set<std::int64_t> modulo(std::int64_t divisor) const;

    /// @brief Expands the symbolic set to concrete values.
    /// @param[in] limit Expansion safety limit.
    /// @return Concrete values up to the requested limit.
    [[nodiscard]] std::set<std::int64_t> expand(std::size_t limit = 16384) const;

    /// @brief Returns a compact textual representation.
    /// @return Human-readable set expression.
    [[nodiscard]] std::string str() const;

    /// @brief Pointwise additive combination of two sets.
    friend BitLengthSet operator+(const BitLengthSet& lhs, const BitLengthSet& rhs);

    /// @brief Set union of two symbolic sets.
    friend BitLengthSet operator|(const BitLengthSet& lhs, const BitLengthSet& rhs);

private:
    /// @brief Internal persistent expression node.
    struct Node;

    /// @brief Constructs from internal node root.
    explicit BitLengthSet(std::shared_ptr<const Node> root);

    /// @brief Root of the persistent symbolic expression tree.
    std::shared_ptr<const Node> root_;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_SEMANTICS_BITLENGTHSET_H
