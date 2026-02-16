#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace llvmdsdl {

namespace {

struct LowerDSDLSerializationPass
    : public mlir::PassWrapper<LowerDSDLSerializationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final { return "lower-dsdl-serialization"; }
  llvm::StringRef getDescription() const final {
    return "Lower DSDL serialization-plan ops into canonical control-flow form";
  }

  void runOnOperation() override {
    // Reserved for future expansion. The normalized DSDL schema/plan is already
    // explicit in the custom dialect; this pass keeps a stable pipeline hook.
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerDSDLSerializationPass() {
  return std::make_unique<LowerDSDLSerializationPass>();
}

void registerDSDLPasses() {
  static bool once = false;
  if (once) {
    return;
  }
  once = true;
  static mlir::PassRegistration<LowerDSDLSerializationPass> reg;
}

} // namespace llvmdsdl
