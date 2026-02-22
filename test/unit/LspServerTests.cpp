//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llvmdsdl/LSP/Server.h"
#include "llvm/Support/JSON.h"

namespace
{

llvm::json::Value parseJson(const std::string& text)
{
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
    if (!parsed)
    {
        std::cerr << "invalid JSON test fixture\n";
        std::abort();
    }
    return std::move(*parsed);
}

const llvm::json::Object* findResponseByIntegerId(const std::vector<llvm::json::Value>& outgoing, const std::int64_t id)
{
    for (const llvm::json::Value& message : outgoing)
    {
        const auto* object = message.getAsObject();
        if (!object)
        {
            continue;
        }
        const auto responseId = object->getInteger("id");
        if (responseId && *responseId == id)
        {
            return object;
        }
    }
    return nullptr;
}

const llvm::json::Object* findResponseByStringId(const std::vector<llvm::json::Value>& outgoing, llvm::StringRef id)
{
    for (const llvm::json::Value& message : outgoing)
    {
        const auto* object = message.getAsObject();
        if (!object)
        {
            continue;
        }
        const auto responseId = object->getString("id");
        if (responseId && *responseId == id)
        {
            return object;
        }
    }
    return nullptr;
}

const llvm::json::Object* findLatestNotificationByMethod(const std::vector<llvm::json::Value>& outgoing,
                                                         llvm::StringRef                       method)
{
    for (auto it = outgoing.rbegin(); it != outgoing.rend(); ++it)
    {
        const auto* object = it->getAsObject();
        if (!object)
        {
            continue;
        }
        const auto methodName = object->getString("method");
        if (methodName && *methodName == method)
        {
            return object;
        }
    }
    return nullptr;
}

const llvm::json::Object* findLatestDiagnosticsForUri(const std::vector<llvm::json::Value>& outgoing,
                                                      llvm::StringRef                       uri)
{
    for (auto it = outgoing.rbegin(); it != outgoing.rend(); ++it)
    {
        const auto* object = it->getAsObject();
        if (!object)
        {
            continue;
        }
        const auto methodName = object->getString("method");
        if (!methodName || *methodName != "textDocument/publishDiagnostics")
        {
            continue;
        }
        const auto* params          = object->getObject("params");
        const auto  notificationUri = params ? params->getString("uri") : std::nullopt;
        if (notificationUri && *notificationUri == uri)
        {
            return object;
        }
    }
    return nullptr;
}

std::filesystem::path makeUniqueTempDir()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("llvmdsdl-lsp-server-" + std::to_string(now));
}

bool writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.good())
    {
        return false;
    }
    out << text;
    return out.good();
}

}  // namespace

bool runLspServerTests()
{
    std::mutex                                mutex;
    std::condition_variable                   cv;
    std::vector<llvm::json::Value>            outgoing;
    std::vector<llvmdsdl::lsp::RequestMetric> metrics;

    llvmdsdl::lsp::Server server(
        [&mutex, &cv, &outgoing](llvm::json::Value message) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                outgoing.push_back(std::move(message));
            }
            cv.notify_all();
        },
        [&mutex, &metrics](const llvmdsdl::lsp::RequestMetric& metric) {
            std::lock_guard<std::mutex> lock(mutex);
            metrics.push_back(metric);
        });

    server.handleMessage(parseJson(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})"));

    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(2), [&outgoing]() { return !outgoing.empty(); }))
        {
            std::cerr << "timeout waiting for initialize response\n";
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 initializeResponse = findResponseByIntegerId(outgoing, 1);
        if (!initializeResponse || !initializeResponse->getObject("result"))
        {
            std::cerr << "missing initialize result payload\n";
            return false;
        }
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/demo.dsdl","version":5,"text":"uint8 value\n@sealed\n"}}})"));

    const auto* snapshotAfterOpen = server.documentStore().lookup("file:///tmp/demo.dsdl");
    if (!snapshotAfterOpen || snapshotAfterOpen->text != "uint8 value\n@sealed\n" || snapshotAfterOpen->version != 5)
    {
        std::cerr << "didOpen did not populate document overlay\n";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 diagnosticsNotification =
            findLatestNotificationByMethod(outgoing, "textDocument/publishDiagnostics");
        if (!diagnosticsNotification)
        {
            std::cerr << "missing diagnostics notification on didOpen\n";
            return false;
        }
        const auto* params      = diagnosticsNotification->getObject("params");
        const auto* diagnostics = params ? params->getArray("diagnostics") : nullptr;
        if (!diagnostics || !diagnostics->empty())
        {
            std::cerr << "expected empty diagnostics for valid document\n";
            return false;
        }
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///tmp/demo.dsdl","version":6},"contentChanges":[{"text":"uint16 value\n@sealed\n"}]}})"));

    const auto* snapshotAfterChange = server.documentStore().lookup("file:///tmp/demo.dsdl");
    if (!snapshotAfterChange || snapshotAfterChange->text != "uint16 value\n@sealed\n" ||
        snapshotAfterChange->version != 6)
    {
        std::cerr << "didChange did not update document overlay\n";
        return false;
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"workspace/didChangeConfiguration","params":{"settings":{"roots":["/tmp/rootA","/tmp/rootB"],"lookupDirs":["/tmp/lookup"],"lint":{"enabled":false,"disabledRules":["style.no_tabs"],"fileSuppressions":{"file:///tmp/demo.dsdl":["naming.field_snake_case"]},"pluginLibraries":["/tmp/plugin.so"]},"ai":{"enabled":true},"trace":"verbose"}}})"));

    if (server.config().rootNamespaceDirs.size() != 2 || server.config().lookupDirs.size() != 1 ||
        server.config().lintEnabled || !server.config().aiEnabled ||
        server.config().traceLevel != llvmdsdl::lsp::TraceLevel::Verbose)
    {
        std::cerr << "didChangeConfiguration did not apply expected settings\n";
        return false;
    }
    if (!server.config().lintDisabledRules.contains("style.no_tabs") ||
        !server.config().lintFileDisabledRules.contains("file:///tmp/demo.dsdl") ||
        server.config().lintPluginLibraries.size() != 1)
    {
        std::cerr << "didChangeConfiguration did not apply lint schema fields\n";
        return false;
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":"file:///tmp/demo.dsdl"}}})"));

    if (server.documentStore().lookup("file:///tmp/demo.dsdl") != nullptr)
    {
        std::cerr << "didClose did not remove document overlay\n";
        return false;
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/bad.dsdl","version":1,"text":"demo.DoesNotExist.1.0 field\n@sealed\n"}}})"));

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 diagnosticsNotification =
            findLatestNotificationByMethod(outgoing, "textDocument/publishDiagnostics");
        if (!diagnosticsNotification)
        {
            std::cerr << "missing diagnostics notification for invalid document\n";
            return false;
        }
        const auto* params      = diagnosticsNotification->getObject("params");
        const auto* diagnostics = params ? params->getArray("diagnostics") : nullptr;
        if (!diagnostics || diagnostics->empty())
        {
            std::cerr << "expected diagnostics for invalid field type\n";
            return false;
        }
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/extent_bad.dsdl","version":1,"text":"uint16 sample\n@extent 13\n"}}})"));
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto* diagnosticsNotification = findLatestDiagnosticsForUri(outgoing, "file:///tmp/extent_bad.dsdl");
        if (!diagnosticsNotification)
        {
            std::cerr << "missing diagnostics notification for invalid extent document\n";
            return false;
        }
        const auto* params      = diagnosticsNotification->getObject("params");
        const auto* diagnostics = params ? params->getArray("diagnostics") : nullptr;
        if (!diagnostics || diagnostics->empty())
        {
            std::cerr << "expected diagnostics for invalid extent document\n";
            return false;
        }

        bool sawExtentValueRange = false;
        for (const llvm::json::Value& diagnosticValue : *diagnostics)
        {
            const auto* diagnostic = diagnosticValue.getAsObject();
            if (!diagnostic)
            {
                continue;
            }
            const auto message = diagnostic->getString("message");
            if (!message || *message != "extent must be a multiple of 8 bits")
            {
                continue;
            }

            const auto* range          = diagnostic->getObject("range");
            const auto* start          = range ? range->getObject("start") : nullptr;
            const auto* end            = range ? range->getObject("end") : nullptr;
            const auto  startLine      = start ? start->getInteger("line") : std::nullopt;
            const auto  startCharacter = start ? start->getInteger("character") : std::nullopt;
            const auto  endCharacter   = end ? end->getInteger("character") : std::nullopt;
            if (!startLine || !startCharacter || !endCharacter)
            {
                continue;
            }
            if (*startLine == 1 && *startCharacter == 8 && (*endCharacter - *startCharacter) == 2)
            {
                sawExtentValueRange = true;
                break;
            }
        }
        if (!sawExtentValueRange)
        {
            std::cerr << "expected extent diagnostic to underline only the expression value\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 54},
        {"method", "textDocument/codeAction"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", "file:///tmp/extent_bad.dsdl"}}},
             {"range",
              llvm::json::Object{
                  {"start", llvm::json::Object{{"line", 1}, {"character", 8}}},
                  {"end", llvm::json::Object{{"line", 1}, {"character", 10}}},
              }},
             {"context",
              llvm::json::Object{
                  {"diagnostics",
                   llvm::json::Array{
                       llvm::json::Object{{"message", "extent must be a multiple of 8 bits"}},
                   }},
              }},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 actionsResponse = findResponseByIntegerId(outgoing, 54);
        const auto*                 result          = actionsResponse ? actionsResponse->getArray("result") : nullptr;
        if (!result || result->empty())
        {
            std::cerr << "expected codeAction quickfix for invalid extent\n";
            return false;
        }

        bool sawExtentFix = false;
        for (const llvm::json::Value& actionValue : *result)
        {
            const auto* action = actionValue.getAsObject();
            if (!action)
            {
                continue;
            }
            const auto  title   = action->getString("title");
            const auto  kind    = action->getString("kind");
            const auto* edit    = action->getObject("edit");
            const auto* changes = edit ? edit->getObject("changes") : nullptr;
            const auto* edits   = changes ? changes->getArray("file:///tmp/extent_bad.dsdl") : nullptr;
            if (!title || *title != "Set extent to 16 bits" || !kind || *kind != "quickfix" || !edits || edits->empty())
            {
                continue;
            }

            const auto* firstEdit      = (*edits)[0].getAsObject();
            const auto* range          = firstEdit ? firstEdit->getObject("range") : nullptr;
            const auto* start          = range ? range->getObject("start") : nullptr;
            const auto* end            = range ? range->getObject("end") : nullptr;
            const auto  startLine      = start ? start->getInteger("line") : std::nullopt;
            const auto  startCharacter = start ? start->getInteger("character") : std::nullopt;
            const auto  endCharacter   = end ? end->getInteger("character") : std::nullopt;
            const auto  replacement    = firstEdit ? firstEdit->getString("newText") : std::nullopt;
            if (!startLine || !startCharacter || !endCharacter || !replacement)
            {
                continue;
            }
            if (*startLine == 1 && *startCharacter == 8 && *endCharacter == 10 && *replacement == "16")
            {
                sawExtentFix = true;
                break;
            }
        }
        if (!sawExtentFix)
        {
            std::cerr << "missing expected extent quickfix edit payload\n";
            return false;
        }
    }

    server.handleMessage(parseJson(
        R"({"jsonrpc":"2.0","id":3,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///tmp/bad.dsdl"}}})"));
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 semanticTokensResponse = findResponseByIntegerId(outgoing, 3);
        if (!semanticTokensResponse)
        {
            std::cerr << "missing semantic tokens response\n";
            return false;
        }
        const auto* result = semanticTokensResponse->getObject("result");
        const auto* data   = result ? result->getArray("data") : nullptr;
        if (!data || data->empty())
        {
            std::cerr << "expected non-empty semantic tokens payload\n";
            return false;
        }
    }

    const std::filesystem::path tmpRoot = makeUniqueTempDir();
    const std::filesystem::path rootDir = tmpRoot / "demo";
    std::error_code             fsError;
    std::filesystem::create_directories(rootDir, fsError);
    if (fsError)
    {
        std::cerr << "failed to create temporary namespace root: " << fsError.message() << "\n";
        return false;
    }

    const std::filesystem::path typeAPath = rootDir / "TypeA.1.0.dsdl";
    const std::filesystem::path typeBPath = rootDir / "TypeB.1.0.dsdl";
    const std::string           typeAText = "uint8 value\n@sealed\n";
    const std::string           typeBText = "demo.TypeA.1.0 member\nuint8 count\n@sealed\n";
    if (!writeTextFile(typeAPath, typeAText) || !writeTextFile(typeBPath, typeBText))
    {
        std::cerr << "failed to write temporary type definitions\n";
        return false;
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"method", "workspace/didChangeConfiguration"},
        {"params",
         llvm::json::Object{
             {"settings",
              llvm::json::Object{
                  {"roots", llvm::json::Array{rootDir.string()}},
                  {"lookupDirs", llvm::json::Array{}},
                  {"lint", llvm::json::Object{{"enabled", false}}},
                  {"ai", llvm::json::Object{{"enabled", false}}},
              }},
         }},
    });

    const std::string typeBUri = llvmdsdl::lsp::normalizedPathToFileUri(typeBPath.string());
    const std::string typeAUri = llvmdsdl::lsp::normalizedPathToFileUri(typeAPath.string());
    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"method", "textDocument/didOpen"},
        {"params",
         llvm::json::Object{
             {"textDocument",
              llvm::json::Object{
                  {"uri", typeBUri},
                  {"version", 1},
                  {"text", typeBText},
              }},
         }},
    });

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 40},
        {"method", "textDocument/hover"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 hoverResponse = findResponseByIntegerId(outgoing, 40);
        if (!hoverResponse)
        {
            std::cerr << "missing hover response\n";
            return false;
        }
        const auto* result   = hoverResponse->getObject("result");
        const auto* contents = result ? result->getObject("contents") : nullptr;
        const auto  value    = contents ? contents->getString("value") : std::nullopt;
        if (!value || value->find("demo.TypeA.1.0") == std::string::npos)
        {
            std::cerr << "hover payload missing referenced type details\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 41},
        {"method", "textDocument/definition"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 definitionResponse = findResponseByIntegerId(outgoing, 41);
        if (!definitionResponse)
        {
            std::cerr << "missing definition response\n";
            return false;
        }
        const auto* result = definitionResponse->getObject("result");
        const auto  uri    = result ? result->getString("uri") : std::nullopt;
        if (!uri || *uri != typeAUri)
        {
            std::cerr << "definition did not resolve to expected type declaration\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 42},
        {"method", "textDocument/references"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
             {"context", llvm::json::Object{{"includeDeclaration", true}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 referencesResponse = findResponseByIntegerId(outgoing, 42);
        if (!referencesResponse)
        {
            std::cerr << "missing references response\n";
            return false;
        }
        const auto* result = referencesResponse->getArray("result");
        if (!result || result->size() < 2)
        {
            std::cerr << "expected references payload with declaration and usage\n";
            return false;
        }

        bool sawDeclaration = false;
        bool sawUsage       = false;
        for (const llvm::json::Value& locationValue : *result)
        {
            const auto* location = locationValue.getAsObject();
            const auto  uri      = location ? location->getString("uri") : std::nullopt;
            if (!uri)
            {
                continue;
            }
            sawDeclaration = sawDeclaration || *uri == typeAUri;
            sawUsage       = sawUsage || *uri == typeBUri;
        }
        if (!sawDeclaration || !sawUsage)
        {
            std::cerr << "references payload missing declaration or usage location\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 43},
        {"method", "textDocument/documentSymbol"},
        {"params", llvm::json::Object{{"textDocument", llvm::json::Object{{"uri", typeBUri}}}}},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 symbolsResponse = findResponseByIntegerId(outgoing, 43);
        if (!symbolsResponse)
        {
            std::cerr << "missing documentSymbol response\n";
            return false;
        }
        const auto* result = symbolsResponse->getArray("result");
        if (!result || result->empty())
        {
            std::cerr << "expected non-empty document symbols payload\n";
            return false;
        }
        bool sawMember = false;
        for (const llvm::json::Value& symbolValue : *result)
        {
            const auto* symbol = symbolValue.getAsObject();
            const auto  name   = symbol ? symbol->getString("name") : std::nullopt;
            sawMember          = sawMember || (name && *name == "member");
        }
        if (!sawMember)
        {
            std::cerr << "document symbols missing expected field symbol\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 44},
        {"method", "textDocument/completion"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 4}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 completionResponse = findResponseByIntegerId(outgoing, 44);
        if (!completionResponse)
        {
            std::cerr << "missing completion response\n";
            return false;
        }
        const auto* result = completionResponse->getObject("result");
        const auto* items  = result ? result->getArray("items") : nullptr;
        if (!items || items->empty())
        {
            std::cerr << "expected non-empty completion payload\n";
            return false;
        }
        bool sawComposite = false;
        for (const llvm::json::Value& itemValue : *items)
        {
            const auto* item  = itemValue.getAsObject();
            const auto  label = item ? item->getString("label") : std::nullopt;
            sawComposite      = sawComposite || (label && *label == "demo.TypeA.1.0");
        }
        if (!sawComposite)
        {
            std::cerr << "completion payload missing expected composite suggestion\n";
            return false;
        }
    }

    llvm::json::Object resolveItem;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 completionResponse = findResponseByIntegerId(outgoing, 44);
        const auto*                 result = completionResponse ? completionResponse->getObject("result") : nullptr;
        const auto*                 items  = result ? result->getArray("items") : nullptr;
        if (!items || items->empty())
        {
            std::cerr << "missing completion payload for resolve test\n";
            return false;
        }
        const auto* firstItem = (*items)[0].getAsObject();
        if (!firstItem)
        {
            std::cerr << "completion item row is not an object\n";
            return false;
        }
        resolveItem = *firstItem;
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 46},
        {"method", "completionItem/resolve"},
        {"params", llvm::json::Value(std::move(resolveItem))},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 resolveResponse = findResponseByIntegerId(outgoing, 46);
        if (!resolveResponse || !resolveResponse->getObject("result"))
        {
            std::cerr << "missing completionItem/resolve response payload\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 45},
        {"method", "workspace/symbol"},
        {"params", llvm::json::Object{{"query", "TypeA"}}},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 symbolsResponse = findResponseByIntegerId(outgoing, 45);
        if (!symbolsResponse)
        {
            std::cerr << "missing workspace/symbol response\n";
            return false;
        }
        const auto* result = symbolsResponse->getArray("result");
        if (!result || result->empty())
        {
            std::cerr << "expected non-empty workspace symbol payload\n";
            return false;
        }

        bool sawTypeA = false;
        for (const llvm::json::Value& rowValue : *result)
        {
            const auto* row      = rowValue.getAsObject();
            const auto  name     = row ? row->getString("name") : std::nullopt;
            const auto* location = row ? row->getObject("location") : nullptr;
            const auto  uri      = location ? location->getString("uri") : std::nullopt;
            sawTypeA             = sawTypeA || (name && *name == "TypeA" && uri && *uri == typeAUri);
        }
        if (!sawTypeA)
        {
            std::cerr << "workspace symbol payload missing expected TypeA entry\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 49},
        {"method", "textDocument/prepareRename"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 prepareResponse = findResponseByIntegerId(outgoing, 49);
        const auto*                 result          = prepareResponse ? prepareResponse->getObject("result") : nullptr;
        const auto                  placeholder     = result ? result->getString("placeholder") : std::nullopt;
        if (!placeholder || *placeholder != "TypeA")
        {
            std::cerr << "prepareRename did not return expected placeholder\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 50},
        {"method", "textDocument/rename"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
             {"newName", "TypeRenamed"},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 renameResponse  = findResponseByIntegerId(outgoing, 50);
        const auto*                 result          = renameResponse ? renameResponse->getObject("result") : nullptr;
        const auto*                 documentChanges = result ? result->getArray("documentChanges") : nullptr;
        if (!documentChanges || documentChanges->empty())
        {
            std::cerr << "rename response missing documentChanges payload\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 51},
        {"method", "dsdld/debug/renamePreview"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"position", llvm::json::Object{{"line", 0}, {"character", 5}}},
             {"newName", "TypeB"},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 previewResponse = findResponseByIntegerId(outgoing, 51);
        const auto*                 result          = previewResponse ? previewResponse->getObject("result") : nullptr;
        const auto                  ok              = result ? result->getBoolean("ok") : std::nullopt;
        const auto*                 conflicts       = result ? result->getArray("conflicts") : nullptr;
        if (!ok.has_value() || *ok || !conflicts || conflicts->empty())
        {
            std::cerr << "rename preview expected conflict when renaming TypeA to existing TypeB\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 52},
        {"method", "textDocument/codeAction"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", "file:///tmp/bad.dsdl"}}},
             {"range",
              llvm::json::Object{
                  {"start", llvm::json::Object{{"line", 0}, {"character", 0}}},
                  {"end", llvm::json::Object{{"line", 0}, {"character", 40}}},
              }},
             {"context",
              llvm::json::Object{
                  {"diagnostics",
                   llvm::json::Array{
                       llvm::json::Object{{"message", "unresolved composite type: demo.DoesNotExist.1.0"}},
                   }},
              }},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 actionsResponse = findResponseByIntegerId(outgoing, 52);
        const auto*                 result          = actionsResponse ? actionsResponse->getArray("result") : nullptr;
        if (!result || result->empty())
        {
            std::cerr << "codeAction expected at least one quickfix for unresolved composite\n";
            return false;
        }
        bool sawQuickFix = false;
        for (const llvm::json::Value& actionValue : *result)
        {
            const auto* action = actionValue.getAsObject();
            const auto  kind   = action ? action->getString("kind") : std::nullopt;
            const auto* edit   = action ? action->getObject("edit") : nullptr;
            sawQuickFix        = sawQuickFix || (kind && *kind == "quickfix" && edit != nullptr);
        }
        if (!sawQuickFix)
        {
            std::cerr << "codeAction payload missing quickfix with edit\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 53},
        {"method", "textDocument/codeAction"},
        {"params",
         llvm::json::Object{
             {"textDocument", llvm::json::Object{{"uri", typeBUri}}},
             {"range",
              llvm::json::Object{
                  {"start", llvm::json::Object{{"line", 0}, {"character", 0}}},
                  {"end", llvm::json::Object{{"line", 1}, {"character", 20}}},
              }},
             {"context", llvm::json::Object{{"diagnostics", llvm::json::Array{}}}},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 actionsResponse = findResponseByIntegerId(outgoing, 53);
        const auto*                 result          = actionsResponse ? actionsResponse->getArray("result") : nullptr;
        if (!result || result->empty())
        {
            std::cerr << "expected refactor code actions for valid file\n";
            return false;
        }
        bool sawRefactorAction = false;
        for (const llvm::json::Value& actionValue : *result)
        {
            const auto* action = actionValue.getAsObject();
            const auto  title  = action ? action->getString("title") : std::nullopt;
            sawRefactorAction =
                sawRefactorAction || (title && *title == "Extract selected field(s) into new type (preview)");
        }
        if (!sawRefactorAction)
        {
            std::cerr << "missing expected refactor code action\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"method", "workspace/didChangeConfiguration"},
        {"params",
         llvm::json::Object{
             {"settings",
              llvm::json::Object{
                  {"roots", llvm::json::Array{rootDir.string()}},
                  {"lookupDirs", llvm::json::Array{}},
                  {"lint",
                   llvm::json::Object{
                       {"enabled", true},
                       {"disabledRules", llvm::json::Array{}},
                       {"fileSuppressions", llvm::json::Object{}},
                   }},
                  {"ai", llvm::json::Object{{"enabled", false}}},
              }},
         }},
    });

    const std::string lintUri = "file:///tmp/lint.dsdl";
    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"method", "textDocument/didOpen"},
        {"params",
         llvm::json::Object{
             {"textDocument",
              llvm::json::Object{
                  {"uri", lintUri},
                  {"version", 1},
                  {"text", "uint8 BadField\\t\\n@sealed\\n"},
              }},
         }},
    });

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 47},
        {"method", "dsdld/debug/scoreExplain"},
        {"params",
         llvm::json::Object{
             {"kind", "workspaceSymbol"},
             {"query", "TypeA"},
             {"limit", 5},
         }},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 explainResponse = findResponseByIntegerId(outgoing, 47);
        if (!explainResponse)
        {
            std::cerr << "missing workspaceSymbol scoreExplain response\n";
            return false;
        }
        const auto* result = explainResponse->getArray("result");
        if (!result || result->empty())
        {
            std::cerr << "expected non-empty workspaceSymbol scoreExplain payload\n";
            return false;
        }
        const auto* first     = (*result)[0].getAsObject();
        const auto* breakdown = first ? first->getObject("breakdown") : nullptr;
        if (!breakdown || !breakdown->getNumber("total_score").has_value())
        {
            std::cerr << "scoreExplain payload missing score breakdown\n";
            return false;
        }
    }

    server.handleMessage(llvm::json::Object{
        {"jsonrpc", "2.0"},
        {"id", 48},
        {"method", "dsdld/debug/indexStats"},
        {"params", llvm::json::Object{}},
    });
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 statsResponse = findResponseByIntegerId(outgoing, 48);
        const auto*                 result        = statsResponse ? statsResponse->getObject("result") : nullptr;
        const auto                  signalEntries = result ? result->getInteger("signal_entries") : std::nullopt;
        if (!signalEntries.has_value() || *signalEntries <= 0)
        {
            std::cerr << "expected adaptive signal entries after completion/symbol requests\n";
            return false;
        }
    }

    std::filesystem::remove_all(tmpRoot, fsError);

    server.handleMessage(
        parseJson(R"({"jsonrpc":"2.0","id":"sleep-1","method":"dsdld/debug/sleep","params":{"duration_ms":1200}})"));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    server.handleMessage(parseJson(R"({"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":"sleep-1"}})"));

    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(3), [&outgoing]() {
                return findResponseByStringId(outgoing, "sleep-1") != nullptr;
            }))
        {
            std::cerr << "timeout waiting for cancelled async request response\n";
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto*                 sleepResponse = findResponseByStringId(outgoing, "sleep-1");
        if (!sleepResponse)
        {
            std::cerr << "missing async request response\n";
            return false;
        }
        const auto* error = sleepResponse->getObject("error");
        if (!error || error->getInteger("code").value_or(0) != -32800)
        {
            std::cerr << "expected cancellation error code for async request\n";
            return false;
        }
    }

    server.handleMessage(parseJson(R"({"jsonrpc":"2.0","id":2,"method":"shutdown","params":null})"));
    server.handleMessage(parseJson(R"({"jsonrpc":"2.0","method":"exit"})"));

    if (!server.shutdownRequested() || !server.shouldExit() || server.exitCode() != 0)
    {
        std::cerr << "expected orderly shutdown + exit state\n";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!findResponseByIntegerId(outgoing, 2))
        {
            std::cerr << "missing shutdown response\n";
            return false;
        }

        const bool sawInitializeMetric =
            std::any_of(metrics.begin(), metrics.end(), [](const llvmdsdl::lsp::RequestMetric& metric) {
                return metric.method == "initialize";
            });
        if (!sawInitializeMetric)
        {
            std::cerr << "missing initialize request telemetry sample\n";
            return false;
        }
    }

    server.shutdown();
    return true;
}
