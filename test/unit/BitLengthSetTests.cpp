//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <set>

#include "llvmdsdl/Semantics/BitLengthSet.h"

bool runBitLengthSetTests()
{
    using llvmdsdl::BitLengthSet;

    BitLengthSet a(8);
    BitLengthSet b = a.repeatRange(3);  // {0,8,16,24}

    if (b.min() != 0 || b.max() != 24)
    {
        std::cerr << "repeatRange bounds mismatch\n";
        return false;
    }

    BitLengthSet c = (BitLengthSet(32) + b).padToAlignment(8);
    if (c.min() != 32 || c.max() != 56)
    {
        std::cerr << "composed set bounds mismatch\n";
        return false;
    }

    auto mod = c.modulo(16);
    if (mod.count(0) == 0 || mod.count(8) == 0)
    {
        std::cerr << "modulo set mismatch\n";
        return false;
    }

    return true;
}
