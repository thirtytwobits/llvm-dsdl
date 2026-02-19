//===----------------------------------------------------------------------===//
///
/// @file
/// Source location primitives shared by parsing, diagnostics, and semantic analysis.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_FRONTEND_SOURCE_LOCATION_H
#define LLVMDSDL_FRONTEND_SOURCE_LOCATION_H

#include <cstdint>
#include <string>

namespace llvmdsdl
{

/// @file
/// @brief Source location primitives shared across frontend and diagnostics.

/// @brief Identifies a concrete position in an input source file.
struct SourceLocation
{
    /// @brief Path to the source file.
    std::string file;

    /// @brief 1-based source line.
    std::uint32_t line{1};

    /// @brief 1-based source column.
    std::uint32_t column{1};

    /// @brief Formats this location as a human-readable string.
    /// @return Formatted location text.
    [[nodiscard]] std::string str() const;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_FRONTEND_SOURCE_LOCATION_H
