//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "llvmdsdl/LSP/DocumentStore.h"

bool runLspDocumentStoreTests()
{
    llvmdsdl::lsp::DocumentStore store;
    store.open("file:///tmp/demo.dsdl", "uint8 value\n", 1);

    if (store.size() != 1)
    {
        std::cerr << "expected one open document after didOpen\n";
        return false;
    }

    const auto* firstSnapshot = store.lookup("file:///tmp/demo.dsdl");
    if (!firstSnapshot || firstSnapshot->text != "uint8 value\n" || firstSnapshot->version != 1)
    {
        std::cerr << "unexpected snapshot after didOpen\n";
        return false;
    }

    if (!store.applyFullTextChange("file:///tmp/demo.dsdl", "uint16 value\n", 2))
    {
        std::cerr << "expected didChange to update existing document\n";
        return false;
    }

    const auto* updatedSnapshot = store.lookup("file:///tmp/demo.dsdl");
    if (!updatedSnapshot || updatedSnapshot->text != "uint16 value\n" || updatedSnapshot->version != 2)
    {
        std::cerr << "unexpected snapshot after didChange\n";
        return false;
    }

    if (store.applyFullTextChange("file:///tmp/missing.dsdl", "bad\n", 1))
    {
        std::cerr << "didChange on missing document should fail\n";
        return false;
    }

    if (!store.close("file:///tmp/demo.dsdl"))
    {
        std::cerr << "expected didClose to remove existing document\n";
        return false;
    }

    if (store.lookup("file:///tmp/demo.dsdl") != nullptr || store.size() != 0)
    {
        std::cerr << "document should be removed after didClose\n";
        return false;
    }

    return true;
}
