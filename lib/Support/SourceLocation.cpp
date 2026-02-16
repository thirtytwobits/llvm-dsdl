#include "llvmdsdl/Frontend/SourceLocation.h"

#include <sstream>

namespace llvmdsdl {

std::string SourceLocation::str() const {
  std::ostringstream out;
  out << file << ':' << line << ':' << column;
  return out.str();
}

} // namespace llvmdsdl
