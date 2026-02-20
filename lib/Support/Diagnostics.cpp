//===----------------------------------------------------------------------===//
///
/// @file
/// Implements diagnostic collection and formatting helpers.
///
/// The diagnostic engine records source-aware notes, warnings, and errors consumed throughout the frontend pipeline.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/Support/Diagnostics.h"

#include <utility>

namespace llvmdsdl
{

void DiagnosticEngine::report(DiagnosticLevel level, const SourceLocation& location, std::string message)
{
    diagnostics_.push_back(Diagnostic{level, location, std::move(message)});
}

void DiagnosticEngine::note(const SourceLocation& location, std::string message)
{
    report(DiagnosticLevel::Note, location, std::move(message));
}

void DiagnosticEngine::warning(const SourceLocation& location, std::string message)
{
    report(DiagnosticLevel::Warning, location, std::move(message));
}

void DiagnosticEngine::error(const SourceLocation& location, std::string message)
{
    report(DiagnosticLevel::Error, location, std::move(message));
}

bool DiagnosticEngine::hasErrors() const
{
    for (const Diagnostic& d : diagnostics_)
    {
        if (d.level == DiagnosticLevel::Error)
        {
            return true;
        }
    }
    return false;
}

}  // namespace llvmdsdl
