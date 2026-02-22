//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "llvmdsdl/LSP/Ranking.h"

namespace
{

std::filesystem::path makeUniqueTempDir()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("llvmdsdl-lsp-ranking-" + std::to_string(now));
}

}  // namespace

bool runLspRankingTests()
{
    const std::filesystem::path tmpRoot = makeUniqueTempDir();
    std::error_code             ec;
    std::filesystem::create_directories(tmpRoot, ec);
    if (ec)
    {
        std::cerr << "failed to create temp ranking dir: " << ec.message() << "\n";
        return false;
    }

    const std::filesystem::path signalPath = tmpRoot / "signals.json";

    llvmdsdl::lsp::AdaptiveSignalStore store(signalPath.string(), 4);
    store.noteExposure("completion:demo.TypeA.1.0");
    store.noteSelection("completion:demo.TypeA.1.0");
    store.noteExposure("completion:demo.TypeB.1.0");
    store.noteExposure("completion:demo.TypeC.1.0");
    store.noteExposure("completion:demo.TypeD.1.0");
    store.noteExposure("completion:demo.TypeE.1.0");
    store.noteSelection("completion:demo.TypeA.1.0");
    if (!store.flush())
    {
        std::cerr << "failed to flush signal store\n";
        std::filesystem::remove_all(tmpRoot, ec);
        return false;
    }

    llvmdsdl::lsp::AdaptiveSignalStore reloaded(signalPath.string(), 4);
    if (reloaded.size() > 4)
    {
        std::cerr << "signal store did not respect bounded size\n";
        std::filesystem::remove_all(tmpRoot, ec);
        return false;
    }

    const auto signalA = reloaded.signalFor("completion:demo.TypeA.1.0");
    if (!signalA.has_value() || signalA->selectionCount == 0)
    {
        std::cerr << "signal store did not persist selection count\n";
        std::filesystem::remove_all(tmpRoot, ec);
        return false;
    }

    const auto exact = llvmdsdl::lsp::RankingModel::scoreCompletion(
        "demo.TypeA.1.0",
        llvmdsdl::lsp::CompletionRankingInput{
            "completion:demo.TypeA.1.0",
            "demo.TypeA.1.0",
            "composite",
            7,
            70.0,
        },
        signalA,
        reloaded.currentTick());

    const auto fuzzy = llvmdsdl::lsp::RankingModel::scoreCompletion(
        "tpA",
        llvmdsdl::lsp::CompletionRankingInput{
            "completion:demo.TypeA.1.0",
            "demo.TypeA.1.0",
            "composite",
            7,
            35.0,
        },
        signalA,
        reloaded.currentTick());

    if (exact.totalScore <= fuzzy.totalScore)
    {
        std::cerr << "expected exact completion match to outrank fuzzy match\n";
        std::filesystem::remove_all(tmpRoot, ec);
        return false;
    }

    const auto symbolExact = llvmdsdl::lsp::RankingModel::scoreSymbol(
        "TypeA",
        llvmdsdl::lsp::SymbolRankingInput{
            "symbol:type:demo.TypeA.1.0",
            "TypeA",
            "demo.TypeA.1.0",
            "demo",
            "message",
            23,
            60.0,
        },
        std::nullopt,
        reloaded.currentTick());

    const auto symbolMismatch = llvmdsdl::lsp::RankingModel::scoreSymbol(
        "Altitude",
        llvmdsdl::lsp::SymbolRankingInput{
            "symbol:type:demo.TypeA.1.0",
            "TypeA",
            "demo.TypeA.1.0",
            "demo",
            "message",
            23,
            60.0,
        },
        std::nullopt,
        reloaded.currentTick());

    if (symbolExact.totalScore <= symbolMismatch.totalScore)
    {
        std::cerr << "expected matching workspace symbol query to outrank mismatch\n";
        std::filesystem::remove_all(tmpRoot, ec);
        return false;
    }

    std::filesystem::remove_all(tmpRoot, ec);
    return true;
}
