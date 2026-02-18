#ifndef LLVMDSDL_CODEGEN_CEMITTER_H
#define LLVMDSDL_CODEGEN_CEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl {

struct CEmitOptions final {
  std::string outDir;
  bool declareVariablesAtTop{true};
  bool optimizeLoweredSerDes{false};
};

llvm::Error emitC(const SemanticModule &semantic, mlir::ModuleOp module,
                  const CEmitOptions &options,
                  DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_CEMITTER_H
