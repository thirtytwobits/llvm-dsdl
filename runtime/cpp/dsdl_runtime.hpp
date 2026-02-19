#ifndef LLVMDSDL_CPP_RUNTIME_HPP
#define LLVMDSDL_CPP_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <memory_resource>

extern "C"
{
#include "dsdl_runtime.h"
}

namespace llvmdsdl::cpp
{

using MemoryResource = std::pmr::memory_resource;

inline MemoryResource* default_memory_resource() noexcept
{
    return std::pmr::get_default_resource();
}

inline constexpr MemoryResource* null_memory_resource() noexcept
{
    return nullptr;
}

}  // namespace llvmdsdl::cpp

#endif  // LLVMDSDL_CPP_RUNTIME_HPP
