//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements overlay document storage for open editor buffers.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/LSP/DocumentStore.h"

#include <utility>

namespace llvmdsdl::lsp
{

void DocumentStore::open(std::string uri, std::string text, const std::int64_t version)
{
    DocumentSnapshot snapshot{uri, std::move(text), version};
    documents_.insert_or_assign(std::move(uri), std::move(snapshot));
}

bool DocumentStore::applyFullTextChange(const std::string& uri, std::string text, const std::int64_t version)
{
    const auto it = documents_.find(uri);
    if (it == documents_.end())
    {
        return false;
    }
    it->second.text    = std::move(text);
    it->second.version = version;
    return true;
}

bool DocumentStore::close(const std::string& uri)
{
    return documents_.erase(uri) > 0U;
}

const DocumentSnapshot* DocumentStore::lookup(const std::string& uri) const
{
    const auto it = documents_.find(uri);
    return it == documents_.end() ? nullptr : &it->second;
}

std::vector<DocumentSnapshot> DocumentStore::snapshots() const
{
    std::vector<DocumentSnapshot> out;
    out.reserve(documents_.size());
    for (const auto& [_, snapshot] : documents_)
    {
        out.push_back(snapshot);
    }
    return out;
}

}  // namespace llvmdsdl::lsp
