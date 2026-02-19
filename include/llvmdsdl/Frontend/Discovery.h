//===----------------------------------------------------------------------===//
///
/// @file
/// Definition discovery declarations for locating and loading DSDL source files.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_FRONTEND_DISCOVERY_H
#define LLVMDSDL_FRONTEND_DISCOVERY_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include <string>
#include <vector>

namespace llvmdsdl
{

/// @file
/// @brief Discovery routines for locating and loading DSDL definitions.

/// @brief Discovers and loads definitions reachable from namespace roots.
/// @param[in] rootNamespaceDirs Root namespace directories.
/// @param[in] lookupDirs Additional lookup directories.
/// @param[in,out] diagnostics Diagnostic sink for discovery/I/O issues.
/// @return Discovered definitions with metadata and source text.
std::vector<DiscoveredDefinition> discoverDefinitions(const std::vector<std::string>& rootNamespaceDirs,
                                                      const std::vector<std::string>& lookupDirs,
                                                      DiagnosticEngine&               diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_FRONTEND_DISCOVERY_H
