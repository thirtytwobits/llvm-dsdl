//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements `Content-Length` framed JSON-RPC stdio transport.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/LSP/JsonRpcIO.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <istream>
#include <ostream>
#include <sstream>
#include <string>

namespace llvmdsdl::lsp
{
namespace
{

bool parseContentLengthHeader(const std::string& line, std::size_t& contentLength)
{
    static constexpr llvm::StringRef Prefix = "Content-Length:";
    llvm::StringRef                  header = line;
    if (!header.consume_front(Prefix))
    {
        return false;
    }
    header = header.trim();
    std::size_t value = 0;
    for (const char ch : header)
    {
        if (ch < '0' || ch > '9')
        {
            return false;
        }
        value = value * 10U + static_cast<std::size_t>(ch - '0');
    }
    contentLength = value;
    return true;
}

}  // namespace

JsonRpcStdioTransport::JsonRpcStdioTransport(std::istream& in, std::ostream& out)
    : input_(in)
    , output_(out)
{
}

bool JsonRpcStdioTransport::readMessage(llvm::json::Value& message, std::string& error)
{
    std::size_t  contentLength = 0U;
    bool         hasHeaders    = false;
    std::string  line;
    while (std::getline(input_, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            break;
        }

        hasHeaders = true;
        if (std::size_t parsedLength = 0; parseContentLengthHeader(line, parsedLength))
        {
            contentLength = parsedLength;
        }
    }

    if (!hasHeaders)
    {
        return false;
    }

    if (contentLength == 0U)
    {
        error = "missing Content-Length header";
        return false;
    }

    std::string payload(contentLength, '\0');
    input_.read(payload.data(), static_cast<std::streamsize>(contentLength));
    if (input_.gcount() != static_cast<std::streamsize>(contentLength))
    {
        error = "truncated JSON-RPC payload";
        return false;
    }

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(payload);
    if (!parsed)
    {
        std::string parseMessage;
        llvm::raw_string_ostream parseStream(parseMessage);
        parseStream << parsed.takeError();
        parseStream.flush();
        error = "invalid JSON payload: " + parseMessage;
        return false;
    }

    message = std::move(*parsed);
    return true;
}

bool JsonRpcStdioTransport::writeMessage(const llvm::json::Value& message)
{
    std::string             payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << message;
    payloadStream.flush();

    std::lock_guard<std::mutex> lock(writeMutex_);
    output_ << "Content-Length: " << payload.size() << "\r\n\r\n" << payload;
    output_.flush();
    return static_cast<bool>(output_);
}

}  // namespace llvmdsdl::lsp
