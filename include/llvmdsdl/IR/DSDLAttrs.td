//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// TableGen attribute definitions for the DSDL MLIR dialect.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_IR_DSDL_ATTRS_TD
#define LLVMDSDL_IR_DSDL_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "llvmdsdl/IR/DSDLBase.td"

def DSDL_RationalAttr : AttrDef<DSDL_Dialect, "Rational"> {
  let mnemonic = "rational";
  let summary = "Canonical rational value (numerator/denominator)";
  let parameters = (ins StringRefParameter<"canonical form">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def DSDL_VersionAttr : AttrDef<DSDL_Dialect, "Version"> {
  let mnemonic = "version";
  let summary = "Major/minor type version";
  let parameters = (ins "uint32_t":$major, "uint32_t":$minor);
  let assemblyFormat = "`<` $major `,` $minor `>`";
}

#endif // LLVMDSDL_IR_DSDL_ATTRS_TD
