#ifndef LLVMDSDL_CODEGEN_RUSTEMITTER_H
#define LLVMDSDL_CODEGEN_RUSTEMITTER_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

namespace llvmdsdl {

enum class RustProfile {
  Std,
  NoStdAlloc,
};

struct RustEmitOptions final {
  std::string outDir;
  std::string crateName{"llvmdsdl_generated"};
  bool emitCargoToml{true};
  RustProfile profile{RustProfile::Std};
};

llvm::Error emitRust(const SemanticModule &semantic, mlir::ModuleOp module,
                     const RustEmitOptions &options,
                     DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_CODEGEN_RUSTEMITTER_H
