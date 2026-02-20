//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// C++ convenience wrappers for the C DSDL runtime.
///
/// This header re-exports the C runtime and provides a small PMR-oriented API
/// used by generated C++ bindings.
///
//===----------------------------------------------------------------------===//

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

/// @brief Alias for the polymorphic memory-resource abstraction.
using MemoryResource = std::pmr::memory_resource;

/// @brief Returns the process-default polymorphic memory resource.
/// @return Pointer to the current default memory resource.
inline MemoryResource* default_memory_resource() noexcept
{
    return std::pmr::get_default_resource();
}

/// @brief Returns a null memory-resource pointer.
/// @return Always `nullptr`.
inline constexpr MemoryResource* null_memory_resource() noexcept
{
    return nullptr;
}

}  // namespace llvmdsdl::cpp

#endif  // LLVMDSDL_CPP_RUNTIME_HPP
