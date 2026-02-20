//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Declarations for DSDL MLIR pass factories and registration entry points.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_TRANSFORMS_PASSES_H
#define LLVMDSDL_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir
{
class Pass;
class OpPassManager;
}  // namespace mlir

namespace llvmdsdl
{

/// @file
/// @brief Registration and factory APIs for DSDL MLIR transform passes.

/// @brief Creates the pass that lowers serialization plans into the canonical
/// @details Lowered-serdes contract form.
/// @return Newly constructed pass instance.
std::unique_ptr<mlir::Pass> createLowerDSDLSerializationPass();

/// @brief Creates the pass that converts lowered DSDL IR to EmitC-oriented IR.
/// @return Newly constructed pass instance.
std::unique_ptr<mlir::Pass> createConvertDSDLToEmitCPass();

/// @brief Adds optional lowered-serdes optimization passes to a pipeline.
/// @param[in,out] pm Pass manager receiving the optimization pipeline.
void addOptimizeLoweredSerDesPipeline(mlir::OpPassManager& pm);

/// @brief Registers conversion-oriented DSDL passes.
void registerDSDLConvertPasses();

/// @brief Registers all DSDL passes and pipelines.
void registerDSDLPasses();

}  // namespace llvmdsdl

#endif  // LLVMDSDL_TRANSFORMS_PASSES_H
