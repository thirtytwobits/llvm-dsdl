//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// TableGen type definitions for the DSDL MLIR dialect.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_IR_DSDLTYPES_TD
#define LLVMDSDL_IR_DSDLTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "llvmdsdl/IR/DSDLBase.td"

def DSDL_TypeRefType : TypeDef<DSDL_Dialect, "TypeRef"> {
  let mnemonic = "type_ref";
  let summary = "A symbolic reference to a DSDL type";
  let parameters = (ins StringRefParameter<"fully-qualified type name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

def DSDL_BitLengthSetType : TypeDef<DSDL_Dialect, "BitLengthSet"> {
  let mnemonic = "bls";
  let summary = "A symbolic bit-length-set type token";
  let parameters = (ins StringRefParameter<"canonical expression">:$expr);
  let assemblyFormat = "`<` $expr `>`";
}

#endif // LLVMDSDL_IR_DSDLTYPES_TD
