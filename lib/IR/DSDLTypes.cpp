#include "llvmdsdl/IR/DSDLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dsdl;

#define GET_TYPEDEF_CLASSES
#include "llvmdsdl/IR/DSDLTypes.cpp.inc"
