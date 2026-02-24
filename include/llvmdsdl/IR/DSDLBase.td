//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Shared TableGen base definitions used by DSDL dialect ops, attrs, and types.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_IR_DSDL_BASE_TD
#define LLVMDSDL_IR_DSDL_BASE_TD

include "mlir/IR/OpBase.td"

def DSDL_Dialect : Dialect {
  let name = "dsdl";
  let cppNamespace = "::mlir::dsdl";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // LLVMDSDL_IR_DSDL_BASE_TD
