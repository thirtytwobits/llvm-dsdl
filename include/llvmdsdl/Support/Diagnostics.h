#ifndef LLVMDSDL_SUPPORT_DIAGNOSTICS_H
#define LLVMDSDL_SUPPORT_DIAGNOSTICS_H

#include "llvmdsdl/Frontend/SourceLocation.h"

#include <string>
#include <vector>

namespace llvmdsdl {

enum class DiagnosticLevel {
  Note,
  Warning,
  Error,
};

struct Diagnostic {
  DiagnosticLevel level;
  SourceLocation location;
  std::string message;
};

class DiagnosticEngine final {
public:
  void report(DiagnosticLevel level, const SourceLocation &location,
              std::string message);

  void note(const SourceLocation &location, std::string message);
  void warning(const SourceLocation &location, std::string message);
  void error(const SourceLocation &location, std::string message);

  [[nodiscard]] bool hasErrors() const;
  [[nodiscard]] const std::vector<Diagnostic> &diagnostics() const {
    return diagnostics_;
  }

private:
  std::vector<Diagnostic> diagnostics_;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_SUPPORT_DIAGNOSTICS_H
