//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// TableGen operation definitions for the DSDL MLIR dialect.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_IR_DSDLOPS_TD
#define LLVMDSDL_IR_DSDLOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "llvmdsdl/IR/DSDLBase.td"

class DSDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<DSDL_Dialect, mnemonic, traits>;

def DSDL_SchemaOp
    : DSDL_Op<"schema", [Symbol, IsolatedFromAbove, NoTerminator]> {
  let summary = "A normalized DSDL schema declaration";
  let description = [{
    Contains field/constant metadata and a serialization plan region.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$full_name,
      I32Attr:$major, I32Attr:$minor, UnitAttr:$sealed,
      OptionalAttr<I64Attr>:$extent_bits, OptionalAttr<I64Attr>:$fixed_port_id,
      UnitAttr:$service, UnitAttr:$deprecated);

  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DSDL_FieldOp : DSDL_Op<"field", [NoMemoryEffect]> {
  let summary = "A field declaration in schema space";
  let arguments = (ins StrAttr:$name, StrAttr:$type_name, UnitAttr:$padding);
  let assemblyFormat = "attr-dict";
}

def DSDL_ConstantOp : DSDL_Op<"constant", [NoMemoryEffect]> {
  let summary = "A constant declaration in schema space";
  let arguments = (ins StrAttr:$name, StrAttr:$type_name, StrAttr:$value_text);
  let assemblyFormat = "attr-dict";
}

def DSDL_SerializationPlanOp
    : DSDL_Op<"serialization_plan", [NoMemoryEffect, NoTerminator]> {
  let summary = "Container for explicit wire-layout operations";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def DSDL_AlignOp : DSDL_Op<"align", [NoMemoryEffect]> {
  let summary = "Alignment event in serialization plan";
  let arguments = (ins I32Attr:$bits);
  let assemblyFormat = "attr-dict";
}

def DSDL_IOOp : DSDL_Op<"io", [NoMemoryEffect]> {
  let summary = "Wire I/O event in serialization plan";
  let arguments = (ins StrAttr:$kind, StrAttr:$name, StrAttr:$type_name);
  let assemblyFormat = "attr-dict";
}

#endif // LLVMDSDL_IR_DSDLOPS_TD
