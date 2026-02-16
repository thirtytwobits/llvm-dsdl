#ifndef LLVMDSDL_SEMANTICS_ANALYZER_H
#define LLVMDSDL_SEMANTICS_ANALYZER_H

#include "llvmdsdl/Frontend/AST.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "llvm/Support/Error.h"

namespace llvmdsdl {

llvm::Expected<SemanticModule> analyze(const ASTModule &module,
                                       const SemanticOptions &options,
                                       DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_SEMANTICS_ANALYZER_H
