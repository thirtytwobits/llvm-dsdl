//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Stdio JSON-RPC framing utilities for Language Server Protocol transport.
///
/// Messages are encoded with `Content-Length` framing over stdio and decoded
/// into LLVM JSON values.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_LSP_JSON_RPC_IO_H
#define LLVMDSDL_LSP_JSON_RPC_IO_H

#include "llvm/Support/JSON.h"

#include <iosfwd>
#include <mutex>
#include <string>

namespace llvmdsdl::lsp
{

/// @brief JSON-RPC stdio transport with `Content-Length` framing.
class JsonRpcStdioTransport final
{
public:
    /// @brief Creates a transport over input and output streams.
    /// @param[in] in Input stream.
    /// @param[in] out Output stream.
    JsonRpcStdioTransport(std::istream& in, std::ostream& out);

    /// @brief Reads one framed JSON-RPC message.
    /// @param[out] message Parsed JSON payload.
    /// @param[out] error Parsing/framing error text when read fails.
    /// @return `true` when a full message is read and parsed.
    [[nodiscard]] bool readMessage(llvm::json::Value& message, std::string& error);

    /// @brief Writes one framed JSON-RPC message.
    /// @param[in] message JSON payload to write.
    /// @return `true` when write succeeds.
    [[nodiscard]] bool writeMessage(const llvm::json::Value& message);

private:
    std::istream& input_;
    std::ostream& output_;
    std::mutex    writeMutex_;
};

}  // namespace llvmdsdl::lsp

#endif  // LLVMDSDL_LSP_JSON_RPC_IO_H
