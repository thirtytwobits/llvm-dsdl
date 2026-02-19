//===----------------------------------------------------------------------===//
///
/// @file
/// Declarations for diagnostic reporting interfaces used across frontend, semantics, lowering, and code generation.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_SUPPORT_DIAGNOSTICS_H
#define LLVMDSDL_SUPPORT_DIAGNOSTICS_H

#include "llvmdsdl/Frontend/SourceLocation.h"

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
};

/// @brief Accumulates diagnostics emitted across all compilation stages.
class DiagnosticEngine final
{
public:
    /// @brief Appends a diagnostic entry.
    /// @param[in] level Severity level.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    void report(DiagnosticLevel level, const SourceLocation& location, std::string message);

    /// @brief Emits a note-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    void note(const SourceLocation& location, std::string message);

    /// @brief Emits a warning-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    void warning(const SourceLocation& location, std::string message);

    /// @brief Emits an error-level diagnostic.
    /// @param[in] location Source location associated with the message.
    /// @param[in] message Human-readable message text.
    void error(const SourceLocation& location, std::string message);

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
