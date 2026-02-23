//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared emission helpers for file-write policy and type-key selection.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/CodeGen/EmitCommon.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

namespace llvmdsdl
{

namespace
{

std::filesystem::perms permsFromMode(const std::uint32_t mode)
{
    using Perm = std::filesystem::perms;
    Perm out   = Perm::none;

    if ((mode & 0400U) != 0U)
    {
        out |= Perm::owner_read;
    }
    if ((mode & 0200U) != 0U)
    {
        out |= Perm::owner_write;
    }
    if ((mode & 0100U) != 0U)
    {
        out |= Perm::owner_exec;
    }
    if ((mode & 0040U) != 0U)
    {
        out |= Perm::group_read;
    }
    if ((mode & 0020U) != 0U)
    {
        out |= Perm::group_write;
    }
    if ((mode & 0010U) != 0U)
    {
        out |= Perm::group_exec;
    }
    if ((mode & 0004U) != 0U)
    {
        out |= Perm::others_read;
    }
    if ((mode & 0002U) != 0U)
    {
        out |= Perm::others_write;
    }
    if ((mode & 0001U) != 0U)
    {
        out |= Perm::others_exec;
    }

    return out;
}

void recordOutputPath(const std::filesystem::path& path, const EmitWritePolicy& policy)
{
    if (policy.recordedOutputs == nullptr)
    {
        return;
    }
    std::error_code ec;
    const auto      absolute = std::filesystem::absolute(path, ec);
    policy.recordedOutputs->push_back(ec ? path.lexically_normal().string() : absolute.lexically_normal().string());
}

}  // namespace

std::string definitionTypeKey(const DiscoveredDefinition& info)
{
    return info.fullName + ":" + std::to_string(info.majorVersion) + ":" + std::to_string(info.minorVersion);
}

std::unordered_set<std::string> makeTypeKeySet(const std::vector<std::string>& typeKeys)
{
    return std::unordered_set<std::string>(typeKeys.begin(), typeKeys.end());
}

bool shouldEmitDefinition(const DiscoveredDefinition& info, const std::unordered_set<std::string>& selectedTypeKeys)
{
    if (selectedTypeKeys.empty())
    {
        return true;
    }
    return selectedTypeKeys.contains(definitionTypeKey(info));
}

llvm::Error writeGeneratedFile(const std::filesystem::path& path, llvm::StringRef content, const EmitWritePolicy& policy)
{
    recordOutputPath(path, policy);

    if (policy.dryRun)
    {
        return llvm::Error::success();
    }

    std::error_code ec;

    const auto parent = path.parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent, ec);
        if (ec)
        {
            return llvm::createStringError(ec, "failed to create output directory %s", parent.string().c_str());
        }
    }

    const bool exists = std::filesystem::exists(path, ec);
    if (ec)
    {
        return llvm::createStringError(ec, "failed to stat output path %s", path.string().c_str());
    }
    if (exists)
    {
        if (policy.noOverwrite)
        {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "refusing to overwrite existing output file: %s",
                                           path.string().c_str());
        }
        const bool removed = std::filesystem::remove(path, ec);
        if (ec || !removed)
        {
            return llvm::createStringError(ec ? ec : llvm::inconvertibleErrorCode(),
                                           "failed to remove existing output file %s",
                                           path.string().c_str());
        }
    }

    llvm::raw_fd_ostream os(path.string(), ec, llvm::sys::fs::OF_Text);
    if (ec)
    {
        return llvm::createStringError(ec, "failed to open %s", path.string().c_str());
    }
    os << content;
    os.close();

    std::filesystem::permissions(path, permsFromMode(policy.fileMode), std::filesystem::perm_options::replace, ec);
    if (ec)
    {
        return llvm::createStringError(ec, "failed to set mode on %s", path.string().c_str());
    }

    return llvm::Error::success();
}

}  // namespace llvmdsdl
