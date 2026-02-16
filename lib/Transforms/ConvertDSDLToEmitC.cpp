#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <vector>

namespace llvmdsdl {
namespace {

struct ConvertDSDLToEmitCPass
    : public mlir::PassWrapper<ConvertDSDLToEmitCPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final { return "convert-dsdl-to-emitc"; }
  llvm::StringRef getDescription() const final {
    return "Lower DSDL dialect schema ops into Func/Arith ops for EmitC lowering";
  }

  void runOnOperation() override {
    auto module = getOperation();
    std::vector<mlir::Operation *> dsdlOps;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() == "dsdl") {
        dsdlOps.push_back(op);
      }
    });

    for (mlir::Operation *op : dsdlOps) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertDSDLToEmitCPass() {
  return std::make_unique<ConvertDSDLToEmitCPass>();
}

namespace {
struct ConvertPassRegistration {
  ConvertPassRegistration() {
    static mlir::PassRegistration<ConvertDSDLToEmitCPass> reg;
  }
} gReg;
} // namespace

} // namespace llvmdsdl
