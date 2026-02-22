//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Implements request telemetry recording and sink forwarding.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/LSP/Telemetry.h"

#include <utility>

namespace llvmdsdl::lsp
{

void Telemetry::setSink(RequestMetricSink sink)
{
    std::lock_guard<std::mutex> lock(mutex_);
    sink_ = std::move(sink);
}

void Telemetry::record(std::string method, const std::uint64_t latencyMicros, const bool cancelled)
{
    RequestMetricSink sink;
    RequestMetric     metric{method, latencyMicros, cancelled};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++requestCounts_[method];
        sink = sink_;
    }
    if (sink)
    {
        sink(metric);
    }
}

std::uint64_t Telemetry::requestCount(const std::string_view method) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = requestCounts_.find(std::string(method));
    return it == requestCounts_.end() ? 0U : it->second;
}

}  // namespace llvmdsdl::lsp
