#include "llvmdsdl/IR/DSDLDialect.h"

#include "llvmdsdl/IR/DSDLOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::dsdl;

#include "llvmdsdl/IR/DSDLDialect.cpp.inc"

void DSDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "llvmdsdl/IR/DSDLOps.cpp.inc"
      >();
}
