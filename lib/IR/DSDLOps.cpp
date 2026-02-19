//===----------------------------------------------------------------------===//
///
/// @file
/// Implements verification and operation glue for DSDL MLIR ops.
///
/// Operation-specific semantic checks are defined here alongside generated operation class inclusions.
///
//===----------------------------------------------------------------------===//

#include "llvmdsdl/IR/DSDLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::dsdl;

LogicalResult SchemaOp::verify()
{
    if (!getSealed() && !getExtentBitsAttr())
    {
        return emitOpError("requires either sealed or extent");
    }
    return success();
}

#define GET_OP_CLASSES
#include "llvmdsdl/IR/DSDLOps.cpp.inc"
