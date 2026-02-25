//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// C++ convenience wrappers for the shared C runtime used by generated DSDL bindings.
///
/// This header re-exports the C runtime and provides common helpers plus an
/// optional PMR-oriented API used by generated std/pmr C++ bindings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVMDSDL_CPP_RUNTIME_HPP
#define LLVMDSDL_CPP_RUNTIME_HPP

#include <cstddef>

#if defined(__cplusplus) && (__cplusplus >= 201703L) && defined(__has_include)
#    if __has_include(<memory_resource>)
#        include <memory_resource>
#        define LLVMDSDL_CPP_HAS_MEMORY_RESOURCE 1
#    else
#        define LLVMDSDL_CPP_HAS_MEMORY_RESOURCE 0
#    endif
#else
#    define LLVMDSDL_CPP_HAS_MEMORY_RESOURCE 0
#endif

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#    if defined(__has_cpp_attribute)
#        if __has_cpp_attribute(nodiscard)
#            define LLVMDSDL_NODISCARD [[nodiscard]]
#        else
#            define LLVMDSDL_NODISCARD
#        endif
#    else
#        define LLVMDSDL_NODISCARD [[nodiscard]]
#    endif
#else
#    define LLVMDSDL_NODISCARD
#endif

extern "C"
{
#include "dsdl_runtime.h"
}

namespace llvmdsdl
{
namespace cpp
{

#if LLVMDSDL_CPP_HAS_MEMORY_RESOURCE
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
#endif

}  // namespace cpp
}  // namespace llvmdsdl

#endif  // LLVMDSDL_CPP_RUNTIME_HPP
