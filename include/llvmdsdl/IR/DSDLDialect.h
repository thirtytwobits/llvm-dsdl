//===----------------------------------------------------------------------===//
///
/// @file
/// Dialect declarations and generated include wiring for the DSDL MLIR dialect.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_IR_DSDLDIALECT_H
#define LLVMDSDL_IR_DSDLDIALECT_H

#include "mlir/IR/Dialect.h"

/// @file
/// @brief DSDL MLIR dialect declaration and generated registration glue.

namespace mlir
{
namespace dsdl
{

/// @brief Primary DSDL dialect class generated in `DSDLDialect.h.inc`.
class DSDLDialect;

}  // namespace dsdl
}  // namespace mlir

#include "llvmdsdl/IR/DSDLDialect.h.inc"

#endif  // LLVMDSDL_IR_DSDLDIALECT_H
