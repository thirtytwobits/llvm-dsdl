//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "llvmdsdl/Transforms/LoweredSerDesContract.h"

#include <iostream>

bool runLoweredContractVersionTests()
{
    bool ok = true;

    if (llvmdsdl::loweredSerDesContractMajorFromEncoded(1) != 1)
    {
        std::cerr << "major decoding failed for encoded version 1\n";
        ok = false;
    }
    if (!llvmdsdl::isSupportedLoweredSerDesContractVersion(1))
    {
        std::cerr << "supported-version check failed for encoded version 1\n";
        ok = false;
    }

    if (llvmdsdl::loweredSerDesContractMajorFromEncoded(2) != 2)
    {
        std::cerr << "major decoding failed for encoded version 2\n";
        ok = false;
    }
    if (llvmdsdl::isSupportedLoweredSerDesContractVersion(2))
    {
        std::cerr << "unsupported-version check failed for encoded version 2\n";
        ok = false;
    }

    if (llvmdsdl::isSupportedLoweredSerDesContractVersion(0))
    {
        std::cerr << "unsupported-version check failed for encoded version 0\n";
        ok = false;
    }

    if (llvmdsdl::loweredSerDesUnsupportedMajorVersionDiagnosticDetail(2) != "expected 1, got 2 (encoded version 2)")
    {
        std::cerr << "unsupported-major diagnostic detail mismatch for encoded version 2\n";
        ok = false;
    }

    if (llvmdsdl::loweredSerDesUnsupportedMajorVersionDiagnosticDetail(17) != "expected 1, got 17 (encoded version 17)")
    {
        std::cerr << "unsupported-major diagnostic detail mismatch for encoded version 17\n";
        ok = false;
    }

    return ok;
}
