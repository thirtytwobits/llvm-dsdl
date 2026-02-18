#include "llvmdsdl/CodeGen/CEmitter.h"
#include "llvmdsdl/CodeGen/CppEmitter.h"
#include "llvmdsdl/CodeGen/GoEmitter.h"
#include "llvmdsdl/CodeGen/RustEmitter.h"
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
  llvm::errs() << "Usage: dsdlc <ast|mlir|c|cpp|rust|go> --root-namespace-dir <dir> [options]\n"
               << "Options:\n"
               << "  --root-namespace-dir <dir>   Primary namespace root (repeatable)\n"
               << "  --lookup-dir <dir>           Dependency namespace root (repeatable)\n"
               << "  --out-dir <dir>              Output directory for 'c'/'cpp'/'rust'/'go' mode\n"
               << "  --compat-mode                Enable compatibility mode\n"
               << "  --strict                     Enable strict mode (default)\n"
               << "  --cpp-profile <std|pmr|both> C++ backend profile (default: both)\n"
               << "  --rust-crate-name <name>     Rust crate/package name (rust mode)\n"
               << "  --rust-profile <std|no-std-alloc> Rust backend profile (default: std)\n"
               << "  --go-module <name>           Go module name (go mode)\n";
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
  llvmdsdl::CppProfile cppProfile = llvmdsdl::CppProfile::Both;
  std::string rustCrateName = "llvmdsdl_generated";
  llvmdsdl::RustProfile rustProfile = llvmdsdl::RustProfile::Std;
  std::string goModuleName = "llvmdsdl_generated";

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
    } else if (arg == "--cpp-profile") {
      const auto value = requireValue(arg);
      if (value == "std") {
        cppProfile = llvmdsdl::CppProfile::Std;
      } else if (value == "pmr") {
        cppProfile = llvmdsdl::CppProfile::Pmr;
      } else if (value == "both") {
        cppProfile = llvmdsdl::CppProfile::Both;
      } else {
        llvm::errs() << "Invalid --cpp-profile value: " << value << "\n";
        printUsage();
        return 1;
      }
    } else if (arg == "--rust-crate-name") {
      rustCrateName = requireValue(arg);
    } else if (arg == "--rust-profile") {
      const auto value = requireValue(arg);
      if (value == "std") {
        rustProfile = llvmdsdl::RustProfile::Std;
      } else if (value == "no-std-alloc") {
        rustProfile = llvmdsdl::RustProfile::NoStdAlloc;
      } else {
        llvm::errs() << "Invalid --rust-profile value: " << value << "\n";
        printUsage();
        return 1;
      }
    } else if (arg == "--go-module") {
      goModuleName = requireValue(arg);
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

    if (llvm::Error err = llvmdsdl::emitC(*semantic, *mlirModule, options,
                                          diagnostics)) {
      llvm::errs() << llvm::toString(std::move(err)) << "\n";
      printDiagnostics(diagnostics);
      return 1;
    }

    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  if (command == "cpp") {
    if (outDir.empty()) {
      llvm::errs() << "--out-dir is required for 'cpp' command\n";
      return 1;
    }
    llvmdsdl::CppEmitOptions options;
    options.outDir = outDir;
    options.profile = cppProfile;

    if (llvm::Error err = llvmdsdl::emitCpp(*semantic, *mlirModule, options,
                                            diagnostics)) {
      llvm::errs() << llvm::toString(std::move(err)) << "\n";
      printDiagnostics(diagnostics);
      return 1;
    }

    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  if (command == "rust") {
    if (outDir.empty()) {
      llvm::errs() << "--out-dir is required for 'rust' command\n";
      return 1;
    }
    llvmdsdl::RustEmitOptions options;
    options.outDir = outDir;
    options.crateName = rustCrateName;
    options.profile = rustProfile;

    if (llvm::Error err = llvmdsdl::emitRust(*semantic, *mlirModule, options,
                                             diagnostics)) {
      llvm::errs() << llvm::toString(std::move(err)) << "\n";
      printDiagnostics(diagnostics);
      return 1;
    }

    printDiagnostics(diagnostics);
    return diagnostics.hasErrors() ? 1 : 0;
  }

  if (command == "go") {
    if (outDir.empty()) {
      llvm::errs() << "--out-dir is required for 'go' command\n";
      return 1;
    }
    llvmdsdl::GoEmitOptions options;
    options.outDir = outDir;
    options.moduleName = goModuleName;

    if (llvm::Error err =
            llvmdsdl::emitGo(*semantic, *mlirModule, options, diagnostics)) {
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
