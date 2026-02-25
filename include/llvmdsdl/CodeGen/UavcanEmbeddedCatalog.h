//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Embedded UAVCAN catalog loader interfaces.
///
//===----------------------------------------------------------------------===//
#ifndef LLVMDSDL_CODEGEN_UAVCAN_EMBEDDED_CATALOG_H
#define LLVMDSDL_CODEGEN_UAVCAN_EMBEDDED_CATALOG_H

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "llvmdsdl/Semantics/Model.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir
{
class Operation;
class MLIRContext;
}  // namespace mlir

namespace llvmdsdl
{

class DiagnosticEngine;

/// @brief Synthetic source-path prefix used for embedded UAVCAN definitions.
inline constexpr const char* kEmbeddedUavcanSyntheticPathPrefix = "<embedded-uavcan>:";

/// @brief Parsed embedded UAVCAN catalog.
struct UavcanEmbeddedCatalog final
{
    /// @brief Reconstructed semantic module for embedded UAVCAN definitions.
    SemanticModule semantic;

    /// @brief Parsed MLIR module containing all embedded `dsdl.schema` ops.
    mlir::OwningOpRef<mlir::ModuleOp> module;

    /// @brief Embedded type keys (`full_name:major:minor`).
    std::unordered_set<std::string> typeKeys;

    /// @brief Embedded schema ops keyed by canonical type key.
    std::unordered_map<std::string, mlir::Operation*> schemaByKey;
};

/// @brief Loads and parses embedded UAVCAN catalog artifacts.
llvm::Expected<UavcanEmbeddedCatalog> loadUavcanEmbeddedCatalog(mlir::MLIRContext& context,
                                                                DiagnosticEngine&  diagnostics);

/// @brief Returns true when `filePath` is synthetic embedded UAVCAN metadata.
bool isEmbeddedUavcanSyntheticPath(const std::string& filePath);

/// @brief Appends embedded schemas for selected keys into destination module.
llvm::Error appendEmbeddedUavcanSchemasForKeys(const UavcanEmbeddedCatalog&           catalog,
                                               mlir::ModuleOp                         destination,
                                               const std::unordered_set<std::string>& selectedTypeKeys,
                                               DiagnosticEngine&                      diagnostics);

}  // namespace llvmdsdl

#endif  // LLVMDSDL_CODEGEN_UAVCAN_EMBEDDED_CATALOG_H
