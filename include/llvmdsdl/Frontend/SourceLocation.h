#ifndef LLVMDSDL_FRONTEND_SOURCE_LOCATION_H
#define LLVMDSDL_FRONTEND_SOURCE_LOCATION_H

#include <cstdint>
#include <string>

namespace llvmdsdl {

struct SourceLocation {
  std::string file;
  std::uint32_t line{1};
  std::uint32_t column{1};

  [[nodiscard]] std::string str() const;
};

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_SOURCE_LOCATION_H
