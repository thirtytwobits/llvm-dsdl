//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Overlay document state for open editor buffers.
///
/// The document store keeps in-memory text and version metadata keyed by URI.
/// It represents unsaved editor overlays consumed by language-server analysis.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_LSP_DOCUMENT_STORE_H
#define LLVMDSDL_LSP_DOCUMENT_STORE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvmdsdl::lsp
{

/// @brief Snapshot of one open document.
struct DocumentSnapshot final
{
    /// @brief LSP document URI.
    std::string uri;

    /// @brief Full document text.
    std::string text;

    /// @brief LSP document version.
    std::int64_t version{0};
};

/// @brief Tracks open-document overlays keyed by URI.
class DocumentStore final
{
public:
    /// @brief Registers a document as opened.
    /// @param[in] uri LSP document URI.
    /// @param[in] text Initial full text.
    /// @param[in] version Initial document version.
    void open(std::string uri, std::string text, std::int64_t version);

    /// @brief Applies a full-text replacement for an existing open document.
    /// @param[in] uri LSP document URI.
    /// @param[in] text New full text.
    /// @param[in] version New document version.
    /// @return `true` when the document exists and is updated.
    [[nodiscard]] bool applyFullTextChange(const std::string& uri, std::string text, std::int64_t version);

    /// @brief Closes a document and removes its overlay entry.
    /// @param[in] uri LSP document URI.
    /// @return `true` when an entry existed and was removed.
    [[nodiscard]] bool close(const std::string& uri);

    /// @brief Looks up a document snapshot by URI.
    /// @param[in] uri LSP document URI.
    /// @return Snapshot pointer when present, otherwise `nullptr`.
    [[nodiscard]] const DocumentSnapshot* lookup(const std::string& uri) const;

    /// @brief Returns a copy of all open document snapshots.
    /// @return Open document snapshots.
    [[nodiscard]] std::vector<DocumentSnapshot> snapshots() const;

    /// @brief Returns the number of open documents tracked.
    /// @return Open document count.
    [[nodiscard]] std::size_t size() const
    {
        return documents_.size();
    }

private:
    /// @brief Open-document overlay map.
    std::unordered_map<std::string, DocumentSnapshot> documents_;
};

}  // namespace llvmdsdl::lsp

#endif  // LLVMDSDL_LSP_DOCUMENT_STORE_H
