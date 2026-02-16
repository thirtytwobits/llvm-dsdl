#ifndef LLVMDSDL_IR_DSDLOPS_H
#define LLVMDSDL_IR_DSDLOPS_H

#include "llvmdsdl/IR/DSDLDialect.h"
#include "llvmdsdl/IR/DSDLAttrs.h"
#include "llvmdsdl/IR/DSDLTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "llvmdsdl/IR/DSDLOps.h.inc"

#endif // LLVMDSDL_IR_DSDLOPS_H
