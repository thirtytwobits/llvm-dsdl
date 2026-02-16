#ifndef LLVMDSDL_CODEGEN_CPPEMITTER_H
#define LLVMDSDL_CODEGEN_CPPEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl {

enum class CppProfile {
  Std,
  Pmr,
  Both,
};

struct CppEmitOptions final {
  std::string outDir;
  CppProfile profile{CppProfile::Both};
};

llvm::Error emitCpp(const SemanticModule &semantic, mlir::ModuleOp module,
                    const CppEmitOptions &options,
                    DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_CPPEMITTER_H
