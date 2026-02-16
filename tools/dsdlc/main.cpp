#include "llvmdsdl/CodeGen/CEmitter.h"
#include "llvmdsdl/Frontend/ASTPrinter.h"
#include "llvmdsdl/Frontend/Parser.h"
#include "llvmdsdl/IR/DSDLDialect.h"
#include "llvmdsdl/Lowering/LowerToMLIR.h"
#include "llvmdsdl/Semantics/Analyzer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace {

void printUsage() {
  llvm::errs() << "Usage: dsdlc <ast|mlir|c> --root-namespace-dir <dir> [options]\n"
               << "Options:\n"
               << "  --root-namespace-dir <dir>   Primary namespace root (repeatable)\n"
               << "  --lookup-dir <dir>           Dependency namespace root (repeatable)\n"
               << "  --out-dir <dir>              Output directory for 'c' mode\n"
               << "  --compat-mode                Enable compatibility mode\n"
               << "  --strict                     Enable strict mode (default)\n"
               << "  --emit-impl-tu               Emit generated_impl.c from MLIR/EmitC\n"
               << "  --emit-runtime-header-only   Emit header-only runtime/codegen (default)\n";
}

void printDiagnostics(const llvmdsdl::DiagnosticEngine &diag) {
  for (const auto &d : diag.diagnostics()) {
    llvm::StringRef level = "note";
    if (d.level == llvmdsdl::DiagnosticLevel::Warning) {
      level = "warning";
    } else if (d.level == llvmdsdl::DiagnosticLevel::Error) {
      level = "error";
    }
    llvm::errs() << d.location.str() << ": " << level << ": " << d.message
                 << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  if (argc < 2) {
    printUsage();
    return 1;
  }

  const std::string command = argv[1];
  std::vector<std::string> roots;
  std::vector<std::string> lookups;
  std::string outDir;
  bool strict = true;
  bool compatMode = false;
  bool emitImplTU = false;
  bool emitHeaderOnly = true;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        llvm::errs() << "Missing value for " << name << "\n";
        printUsage();
        std::exit(1);
      }
      return argv[++i];
    };

    if (arg == "--root-namespace-dir") {
      roots.push_back(requireValue(arg));
    } else if (arg == "--lookup-dir") {
      lookups.push_back(requireValue(arg));
    } else if (arg == "--out-dir") {
      outDir = requireValue(arg);
    } else if (arg == "--compat-mode") {
      compatMode = true;
      strict = false;
    } else if (arg == "--strict") {
      strict = true;
      compatMode = false;
    } else if (arg == "--emit-impl-tu") {
      emitImplTU = true;
    } else if (arg == "--emit-runtime-header-only") {
      emitHeaderOnly = true;
    } else {
      llvm::errs() << "Unknown argument: " << arg << "\n";
      printUsage();
      return 1;
    }
  }

  if (roots.empty()) {
    llvm::errs() << "At least one --root-namespace-dir is required\n";
    return 1;
  }

  llvmdsdl::DiagnosticEngine diagnostics;

  auto ast = llvmdsdl::parseDefinitions(roots, lookups, diagnostics);
  if (!ast) {
    llvm::consumeError(ast.takeError());
    printDiagnostics(diagnostics);
    return 1;
  }

  if (command == "ast") {
    llvm::outs() << llvmdsdl::printAST(*ast);
    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  llvmdsdl::SemanticOptions semOptions;
  semOptions.strict = strict;
  semOptions.compatMode = compatMode;

  auto semantic = llvmdsdl::analyze(*ast, semOptions, diagnostics);
  if (!semantic) {
    llvm::consumeError(semantic.takeError());
    printDiagnostics(diagnostics);
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::dsdl::DSDLDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::emitc::EmitCDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::dsdl::DSDLDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();

  auto mlirModule = llvmdsdl::lowerToMLIR(*semantic, context, diagnostics);
  if (!mlirModule) {
    printDiagnostics(diagnostics);
    return 1;
  }

  if (command == "mlir") {
    mlirModule->print(llvm::outs());
    llvm::outs() << "\n";
    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  if (command == "c") {
    if (outDir.empty()) {
      llvm::errs() << "--out-dir is required for 'c' command\n";
      return 1;
    }
    llvmdsdl::CEmitOptions options;
    options.outDir = outDir;
    options.emitImplTranslationUnit = emitImplTU;
    options.emitHeaderOnly = emitHeaderOnly;

    if (llvm::Error err = llvmdsdl::emitC(*semantic, *mlirModule, options,
                                          diagnostics)) {
      llvm::errs() << llvm::toString(std::move(err)) << "\n";
      printDiagnostics(diagnostics);
      return 1;
    }

    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  llvm::errs() << "Unknown command: " << command << "\n";
  printUsage();
  return 1;
}
