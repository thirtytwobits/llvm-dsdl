//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared C ABI declarations for differential parity integration harnesses.
///
/// The ABI describes per-type serialize/deserialize entry points consumed by
/// cross-language parity drivers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVMDSDL_DIFFERENTIAL_PARITY_ABI_H
#define LLVMDSDL_DIFFERENTIAL_PARITY_ABI_H

#include <stddef.h>
#include <stdint.h>

/// @brief Function pointer signature for serializer entry points.
typedef int8_t (*DifferentialSerializeFn)(const void*, uint8_t*, size_t*);
/// @brief Function pointer signature for deserializer entry points.
typedef int8_t (*DifferentialDeserializeFn)(void*, const uint8_t*, size_t*);

/// @brief Function table and metadata for one differential parity test case.
typedef struct
{
    /// @brief Size in bytes of the object instance accepted by the callbacks.
    size_t object_size;
    /// @brief Maximum serialized payload size in bytes for this type.
    size_t max_serialized_size;
    /// @brief Serializer callback for this case.
    DifferentialSerializeFn serialize;
    /// @brief Deserializer callback for this case.
    DifferentialDeserializeFn deserialize;
} DifferentialCaseInfo;

#endif
