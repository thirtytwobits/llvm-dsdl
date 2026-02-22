//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Declarations for diagnostic reporting interfaces used across frontend, semantics, lowering, and code generation.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_SUPPORT_DIAGNOSTICS_H
#define LLVMDSDL_SUPPORT_DIAGNOSTICS_H

#include "llvmdsdl/Frontend/SourceLocation.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvmdsdl
{

/// @file
/// @brief Diagnostic collection and reporting interfaces.

/// @brief Severity level for a diagnostic message.
enum class DiagnosticLevel
{

    /// @brief Informational note.
    Note,

    /// @brief Non-fatal warning.
    Warning,

    /// @brief Fatal error.
    Error,
};

/// @brief Single diagnostic record produced by the pipeline.
struct Diagnostic
{
    /// @brief Severity level.
    DiagnosticLevel level;

    /// @brief Source location associated with the message.
    SourceLocation location;

    /// @brief Human-readable message text.
    std::string message;

    /// @brief Highlight length in source characters.
    std::uint32_t length{1};

    /// @brief Optional numeric suggestion used by fix-it producers.
    std::optional<std::int64_t> suggestedInteger;
};

/// @brief Accumulates diagnostics emitted across all compilation stages.
class DiagnosticEngine final
{
public:
    /// @brief Appends a diagnostic entry.
    /// @param[in] level Severity level.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    /// @param[in] length Highlight length in source characters.
    /// @param[in] suggestedInteger Optional numeric replacement suggestion.
    void report(DiagnosticLevel level, const SourceLocation& location, std::string message, std::uint32_t length = 1);
    void report(DiagnosticLevel             level,
                const SourceLocation&       location,
                std::string                 message,
                std::uint32_t               length,
                std::optional<std::int64_t> suggestedInteger);

    /// @brief Emits a note-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    /// @param[in] length Highlight length in source characters.
    void note(const SourceLocation& location, std::string message, std::uint32_t length = 1);
    void note(const SourceLocation&       location,
              std::string                 message,
              std::uint32_t               length,
              std::optional<std::int64_t> suggestedInteger);

    /// @brief Emits a warning-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    /// @param[in] length Highlight length in source characters.
    void warning(const SourceLocation& location, std::string message, std::uint32_t length = 1);
    void warning(const SourceLocation&       location,
                 std::string                 message,
                 std::uint32_t               length,
                 std::optional<std::int64_t> suggestedInteger);

    /// @brief Emits an error-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    /// @param[in] length Highlight length in source characters.
    void error(const SourceLocation& location, std::string message, std::uint32_t length = 1);
    void error(const SourceLocation&       location,
               std::string                 message,
               std::uint32_t               length,
               std::optional<std::int64_t> suggestedInteger);

    /// @brief Indicates whether any error diagnostics were recorded.
    /// @return True when at least one error exists.
    [[nodiscard]] bool hasErrors() const;

    /// @brief Returns all recorded diagnostics in insertion order.
    /// @return Immutable diagnostic list.
    [[nodiscard]] const std::vector<Diagnostic>& diagnostics() const
    {
        return diagnostics_;
    }

private:
    /// @brief Backing storage for collected diagnostics.
    std::vector<Diagnostic> diagnostics_;
};

}  // namespace llvmdsdl

#endif  // LLVMDSDL_SUPPORT_DIAGNOSTICS_H
