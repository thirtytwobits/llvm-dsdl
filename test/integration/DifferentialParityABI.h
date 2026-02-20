//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef LLVMDSDL_DIFFERENTIAL_PARITY_ABI_H
#define LLVMDSDL_DIFFERENTIAL_PARITY_ABI_H

#include <stddef.h>
#include <stdint.h>

typedef int8_t (*DifferentialSerializeFn)(const void*, uint8_t*, size_t*);
typedef int8_t (*DifferentialDeserializeFn)(void*, const uint8_t*, size_t*);

typedef struct
{
    size_t                    object_size;
    size_t                    max_serialized_size;
    DifferentialSerializeFn   serialize;
    DifferentialDeserializeFn deserialize;
} DifferentialCaseInfo;

#endif
