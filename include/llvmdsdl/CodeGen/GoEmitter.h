#ifndef LLVMDSDL_CODEGEN_GOEMITTER_H
#define LLVMDSDL_CODEGEN_GOEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl {

struct GoEmitOptions final {
  std::string outDir;
  std::string moduleName{"llvmdsdl_generated"};
  bool emitGoMod{true};
};

llvm::Error emitGo(const SemanticModule &semantic, mlir::ModuleOp module,
                   const GoEmitOptions &options,
                   DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_GOEMITTER_H
