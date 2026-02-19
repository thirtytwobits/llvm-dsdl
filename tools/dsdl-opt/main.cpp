#include "llvmdsdl/IR/DSDLDialect.h"
#include "llvmdsdl/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);

    mlir::DialectRegistry registry;
    registry.insert<mlir::dsdl::DSDLDialect,
                    mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::emitc::EmitCDialect>();

    llvmdsdl::registerDSDLPasses();

    return failed(mlir::MlirOptMain(argc, argv, "dsdl-opt", registry));
}
