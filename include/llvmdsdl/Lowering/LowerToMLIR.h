#ifndef LLVMDSDL_LOWERING_LOWERTOMLIR_H
#define LLVMDSDL_LOWERING_LOWERTOMLIR_H

#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;
}

namespace llvmdsdl {

mlir::OwningOpRef<mlir::ModuleOp>
lowerToMLIR(const SemanticModule &module, mlir::MLIRContext &context,
            DiagnosticEngine &diagnostics);

} // namespace llvmdsdl

#endif // LLVMDSDL_LOWERING_LOWERTOMLIR_H
