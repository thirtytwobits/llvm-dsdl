#ifndef LLVMDSDL_CODEGEN_TSEMITTER_H
#define LLVMDSDL_CODEGEN_TSEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl {

struct TsEmitOptions final {
  std::string outDir;
  std::string moduleName{"llvmdsdl_generated"};
  bool emitPackageJson{true};
  bool optimizeLoweredSerDes{false};
};

llvm::Error emitTs(const SemanticModule &semantic, mlir::ModuleOp module,
                   const TsEmitOptions &options,
                   DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_TSEMITTER_H
