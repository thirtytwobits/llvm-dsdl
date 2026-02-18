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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

bool isKnownCommand(llvm::StringRef command) {
  return command == "ast" || command == "mlir" || command == "c" ||
         command == "cpp" || command == "rust" || command == "go";
}

bool isHelpToken(llvm::StringRef arg) { return arg == "--help" || arg == "-h"; }

void printUsage() {
  llvm::errs()
      << "Usage: dsdlc <ast|mlir|c|cpp|rust|go> --root-namespace-dir <dir> [options]\n"
      << "Try: dsdlc --help\n";
}

void printHelp(const std::string &selectedCommand = "") {
  llvm::errs()
      << "NAME\n"
      << "  dsdlc - DSDL frontend, MLIR lowerer, and multi-language code generator\n\n"
      << "SYNOPSIS\n"
      << "  dsdlc <command> --root-namespace-dir <dir> [--root-namespace-dir <dir> ...] [options]\n"
      << "  dsdlc --help\n"
      << "  dsdlc <command> --help\n\n"
      << "DESCRIPTION\n"
      << "  dsdlc discovers and parses .dsdl definitions, performs semantic analysis, lowers into\n"
      << "  a DSDL MLIR dialect, and can emit output for C, C++, Rust, and Go. Root namespaces are\n"
      << "  repeatable for multi-root projects. Lookup directories are optional dependency roots.\n\n"
      << "COMMANDS\n"
      << "  ast   Print parsed AST for all discovered definitions.\n"
      << "  mlir  Print lowered DSDL MLIR module.\n"
      << "  c     Generate C headers + per-definition C implementation units.\n"
      << "  cpp   Generate C++23 headers (std, pmr, or both profiles).\n"
      << "  rust  Generate Rust crate layout and SerDes/runtime integration.\n"
      << "  go    Generate Go module/package layout and SerDes/runtime integration.\n\n"
      << "COMMON OPTIONS\n"
      << "  --root-namespace-dir <dir>\n"
      << "      Primary input root. Repeat to add more top-level namespace roots.\n"
      << "      Required for all commands except --help.\n"
      << "  --lookup-dir <dir>\n"
      << "      Additional dependency lookup root. Repeat as needed.\n"
      << "  --strict\n"
      << "      Enable strict semantic behavior (default).\n"
      << "      Strict means: enforce the Cyphal DSDL specification as implemented by this compiler.\n"
      << "      Recommended for CI/release builds.\n"
      << "  --compat-mode\n"
      << "      Relax selected semantic checks to keep generation running on legacy/non-strict trees.\n"
      << "      Compatibility target: historical OpenCyphal-style definition workflows and existing trees\n"
      << "      that depend on permissive behavior.\n"
      << "  --help, -h\n"
      << "      Print this help text. With a command, prints command-focused guidance.\n\n"
      << "STRICTNESS MODES\n"
      << "  Default behavior:\n"
      << "    - If neither flag is provided, dsdlc runs in strict mode.\n"
      << "  --strict:\n"
      << "    - Prefer when you need deterministic, standards-aligned validation.\n"
      << "    - \"Standards-aligned\" here means Cyphal DSDL spec conformance for language semantics.\n"
      << "    - Definitions that fail strict semantic rules are reported as errors.\n"
      << "  --compat-mode:\n"
      << "    - Accepts some inputs that strict mode would reject, to ease migration.\n"
      << "    - Enables fallback behavior (for example, clamping/defaulting in malformed cases) and emits\n"
      << "      compatibility warnings instead of hard errors in selected situations.\n"
      << "    - Use when porting older trees; then move back to --strict once diagnostics are resolved.\n"
      << "  Flag precedence:\n"
      << "    - If both flags are present, the last one on the command line wins.\n"
      << "      Example: '--compat-mode --strict' ends in strict mode.\n"
      << "      Example: '--strict --compat-mode' ends in compat mode.\n\n"
      << "CODEGEN OPTIONS (c/cpp/rust/go)\n"
      << "  --out-dir <dir>\n"
      << "      Output directory root for generated files.\n\n"
      << "C++ OPTIONS (cpp)\n"
      << "  --cpp-profile <std|pmr|both>\n"
      << "      std  -> std::vector-backed variable arrays\n"
      << "      pmr  -> std::pmr::vector-backed variable arrays\n"
      << "      both -> emit two trees: <out>/std and <out>/pmr (default)\n\n"
      << "RUST OPTIONS (rust)\n"
      << "  --rust-crate-name <name>\n"
      << "      Rust crate name to emit into Cargo.toml (default: llvmdsdl_generated).\n"
      << "  --rust-profile <std|no-std-alloc>\n"
      << "      std is supported and default.\n"
      << "      no-std-alloc is reserved and currently reports not implemented.\n\n"
      << "GO OPTIONS (go)\n"
      << "  --go-module <name>\n"
      << "      Go module name written to go.mod (default: llvmdsdl_generated).\n\n"
      << "RUN SUMMARY\n"
      << "  On successful command execution, dsdlc prints a summary to stderr with:\n"
      << "    - files generated\n"
      << "    - output root\n"
      << "    - elapsed wall time\n\n"
      << "EXAMPLES\n"
      << "  dsdlc ast --root-namespace-dir public_regulated_data_types/uavcan\n"
      << "  dsdlc mlir --root-namespace-dir public_regulated_data_types/uavcan\n"
      << "  dsdlc c --root-namespace-dir public_regulated_data_types/uavcan --strict --out-dir build/uavcan-c\n"
      << "  dsdlc cpp --root-namespace-dir public_regulated_data_types/uavcan --cpp-profile both --out-dir build/uavcan-cpp\n"
      << "  dsdlc rust --root-namespace-dir public_regulated_data_types/uavcan --rust-profile std --rust-crate-name uavcan_dsdl_generated --out-dir build/uavcan-rust\n"
      << "  dsdlc go --root-namespace-dir public_regulated_data_types/uavcan --go-module demo/uavcan/generated --out-dir build/uavcan-go\n\n"
      << "EXIT STATUS\n"
      << "  0 on success, non-zero on parse/semantic/lowering/codegen failure or invalid CLI usage.\n";

  if (!selectedCommand.empty() && isKnownCommand(selectedCommand)) {
    llvm::errs() << "\nCOMMAND FOCUS (" << selectedCommand << ")\n";
    if (selectedCommand == "ast") {
      llvm::errs() << "  Emits AST text to stdout. --out-dir is not used.\n";
    } else if (selectedCommand == "mlir") {
      llvm::errs() << "  Emits DSDL MLIR to stdout. --out-dir is not used.\n";
    } else if (selectedCommand == "c") {
      llvm::errs() << "  Requires --out-dir. Emits C headers, C source files, and runtime header.\n";
    } else if (selectedCommand == "cpp") {
      llvm::errs() << "  Requires --out-dir. Honors --cpp-profile (std|pmr|both).\n";
    } else if (selectedCommand == "rust") {
      llvm::errs()
          << "  Requires --out-dir. Honors --rust-crate-name and --rust-profile.\n";
    } else if (selectedCommand == "go") {
      llvm::errs() << "  Requires --out-dir. Honors --go-module.\n";
    }
  }
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

std::string resolveOutputRoot(const std::string &root) {
  if (root.empty()) {
    return "stdout";
  }
  std::error_code ec;
  const auto abs = std::filesystem::absolute(root, ec);
  if (!ec) {
    return abs.string();
  }
  return root;
}

std::uint64_t countRegularFiles(const std::string &root) {
  if (root.empty()) {
    return 0;
  }
  std::error_code ec;
  const std::filesystem::path outputRoot(root);
  if (!std::filesystem::exists(outputRoot, ec) || ec) {
    return 0;
  }

  std::uint64_t count = 0;
  std::filesystem::recursive_directory_iterator it(
      outputRoot, std::filesystem::directory_options::skip_permission_denied,
      ec);
  std::filesystem::recursive_directory_iterator end;
  if (ec) {
    return 0;
  }
  for (; it != end; it.increment(ec)) {
    if (ec) {
      ec.clear();
      continue;
    }
    if (it->is_regular_file(ec) && !ec) {
      ++count;
    }
    ec.clear();
  }
  return count;
}

void printRunSummary(llvm::StringRef command, llvm::StringRef outputRoot,
                     const std::uint64_t generatedFiles,
                     const std::chrono::steady_clock::duration elapsed) {
  const auto elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  const auto elapsedWholeSec = elapsedMs / 1000;
  const auto elapsedFractionMs = elapsedMs % 1000;
  llvm::errs() << "Run summary:\n"
               << "  command: " << command << "\n"
               << "  output root: " << outputRoot << "\n"
               << "  files generated: " << generatedFiles << "\n"
               << "  elapsed: " << elapsedWholeSec << ".";
  if (elapsedFractionMs < 100) {
    llvm::errs() << "0";
  }
  if (elapsedFractionMs < 10) {
    llvm::errs() << "0";
  }
  llvm::errs() << elapsedFractionMs << "s\n";
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  if (argc < 2) {
    printUsage();
    return 1;
  }

  const std::string command = argv[1];
  if (isHelpToken(command) || command == "help") {
    printHelp();
    return 0;
  }
  if (!isKnownCommand(command)) {
    llvm::errs() << "Unknown command: " << command << "\n";
    printUsage();
    return 1;
  }

  std::vector<std::string> roots;
  std::vector<std::string> lookups;
  std::string outDir;
  bool helpRequested = false;
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
    } else if (isHelpToken(arg)) {
      helpRequested = true;
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

  if (helpRequested) {
    printHelp(command);
    return 0;
  }

  if (roots.empty()) {
    llvm::errs() << "At least one --root-namespace-dir is required\n";
    return 1;
  }

  const auto startTime = std::chrono::steady_clock::now();
  llvmdsdl::DiagnosticEngine diagnostics;
  auto finish = [&](const std::string &outputRoot,
                    const std::uint64_t generatedFiles) -> int {
    printDiagnostics(diagnostics);
    printRunSummary(command, outputRoot, generatedFiles,
                    std::chrono::steady_clock::now() - startTime);
    return diagnostics.hasErrors() ? 1 : 0;
  };

  auto ast = llvmdsdl::parseDefinitions(roots, lookups, diagnostics);
  if (!ast) {
    llvm::consumeError(ast.takeError());
    printDiagnostics(diagnostics);
    return 1;
  }

  if (command == "ast") {
    llvm::outs() << llvmdsdl::printAST(*ast);
    return finish("stdout", 0);
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
    return finish("stdout", 0);
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

    return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
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

    return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
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

    return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
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

    return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
  }

  llvm::errs() << "Unhandled command path: " << command << "\n";
  return 1;
}
