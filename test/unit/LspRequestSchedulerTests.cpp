//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>

#include "llvmdsdl/LSP/RequestScheduler.h"

bool runLspRequestSchedulerTests()
{
    llvmdsdl::lsp::RequestScheduler scheduler;

    std::mutex                       mutex;
    std::condition_variable          cv;
    bool                             sawCallback = false;
    llvmdsdl::lsp::RequestTaskResult completionResult;
    std::uint64_t                    completionLatency = 0;

    const bool queued = scheduler.enqueue(
        "i:7",
        "test/sleep",
        [](llvmdsdl::lsp::CancellationToken token) -> llvmdsdl::lsp::RequestTaskResult {
            const auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < std::chrono::seconds(1))
            {
                if (token.isCancellationRequested())
                {
                    return {llvmdsdl::lsp::RequestTaskStatus::Cancelled, llvm::json::Value(nullptr), {}};
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            return {llvmdsdl::lsp::RequestTaskStatus::Completed, llvm::json::Object{{"done", true}}, {}};
        },
        [&mutex, &cv, &sawCallback, &completionResult, &completionLatency](llvmdsdl::lsp::RequestTaskResult result,
                                                                           const std::uint64_t latencyMicros) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                sawCallback       = true;
                completionResult  = std::move(result);
                completionLatency = latencyMicros;
            }
            cv.notify_all();
        });

    if (!queued)
    {
        std::cerr << "expected enqueue to succeed\n";
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    if (!scheduler.cancel("i:7"))
    {
        std::cerr << "expected cancel to find enqueued request\n";
        return false;
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(3), [&sawCallback]() { return sawCallback; }))
        {
            std::cerr << "timeout waiting for completion callback\n";
            return false;
        }
    }

    if (completionResult.status != llvmdsdl::lsp::RequestTaskStatus::Cancelled)
    {
        std::cerr << "expected cancelled task status\n";
        return false;
    }

    if (completionLatency == 0)
    {
        std::cerr << "expected non-zero latency metric\n";
        return false;
    }

    scheduler.shutdown();
    return true;
}
