#ifndef LLVMDSDL_TRANSFORMS_PASSES_H
#define LLVMDSDL_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace llvmdsdl {

std::unique_ptr<mlir::Pass> createLowerDSDLSerializationPass();
std::unique_ptr<mlir::Pass> createConvertDSDLToEmitCPass();
void registerDSDLConvertPasses();
void registerDSDLPasses();

} // namespace llvmdsdl

#endif // LLVMDSDL_TRANSFORMS_PASSES_H
