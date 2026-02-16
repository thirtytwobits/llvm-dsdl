#ifndef LLVMDSDL_FRONTEND_DISCOVERY_H
#define LLVMDSDL_FRONTEND_DISCOVERY_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include <string>
#include <vector>

namespace llvmdsdl {

std::vector<DiscoveredDefinition>
discoverDefinitions(const std::vector<std::string> &rootNamespaceDirs,
                    const std::vector<std::string> &lookupDirs,
                    DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_FRONTEND_DISCOVERY_H
