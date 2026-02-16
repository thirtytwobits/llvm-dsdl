#ifndef LLVMDSDL_IR_DSDLBASE_TD
#define LLVMDSDL_IR_DSDLBASE_TD

include "mlir/IR/OpBase.td"

def DSDL_Dialect : Dialect {
  let name = "dsdl";
  let cppNamespace = "::mlir::dsdl";
}

#endif // LLVMDSDL_IR_DSDLBASE_TD
