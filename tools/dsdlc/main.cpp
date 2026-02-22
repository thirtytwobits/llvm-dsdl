//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// Entry point for the `dsdlc` command-line compiler front-end.
///
/// This tool parses DSDL definitions, runs semantic analysis, lowers to MLIR,
/// and dispatches to language backends (C, C++, Rust, Go, TypeScript) or text
/// output modes (`ast`, `mlir`).
///
//===----------------------------------------------------------------------===//

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <system_error>
#include <utility>

#include "llvmdsdl/CodeGen/CEmitter.h"
#include "llvmdsdl/CodeGen/CppEmitter.h"
#include "llvmdsdl/CodeGen/GoEmitter.h"
#include "llvmdsdl/CodeGen/PythonEmitter.h"
#include "llvmdsdl/CodeGen/RustEmitter.h"
#include "llvmdsdl/CodeGen/TsEmitter.h"
#include "llvmdsdl/Frontend/ASTPrinter.h"
#include "llvmdsdl/Frontend/Parser.h"
#include "llvmdsdl/Lowering/LowerToMLIR.h"
#include "llvmdsdl/Semantics/Analyzer.h"
#include "llvmdsdl/IR/DSDLDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvmdsdl/Frontend/SourceLocation.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"

namespace
{

/// @brief Checks whether a command token is implemented by `dsdlc`.
///
/// @param[in] command Command token from argv.
/// @return `true` if the command is one of the supported subcommands.
bool isKnownCommand(llvm::StringRef command)
{
    return command == "ast" || command == "mlir" || command == "c" || command == "cpp" || command == "rust" ||
           command == "go" || command == "ts" || command == "python";
}

/// @brief Checks whether a token is a help switch.
///
/// @param[in] arg Argument token from argv.
/// @return `true` when the argument is `--help` or `-h`.
bool isHelpToken(llvm::StringRef arg)
{
    return arg == "--help" || arg == "-h";
}

/// @brief Prints compact usage guidance for invalid CLI invocations.
void printUsage()
{
    llvm::errs() << "Usage: dsdlc <ast|mlir|c|cpp|rust|go|ts|python> --root-namespace-dir <dir> [options]\n"
                 << "Try: dsdlc --help\n";
}

/// @brief Prints the full help text and optional command-focused details.
///
/// @param[in] selectedCommand Optional command name used for focused help.
void printHelp(const std::string& selectedCommand = "")
{
    llvm::errs()
        << "NAME\n"
        << "  dsdlc - DSDL frontend, MLIR lowerer, and multi-language code generator\n\n"
        << "SYNOPSIS\n"
        << "  dsdlc <command> --root-namespace-dir <dir> [--root-namespace-dir <dir> ...] [options]\n"
        << "  dsdlc --help\n"
        << "  dsdlc <command> --help\n\n"
        << "DESCRIPTION\n"
        << "  dsdlc discovers and parses .dsdl definitions, performs semantic analysis, lowers into\n"
        << "  a DSDL MLIR dialect, and can emit output for C, C++, Rust, Go, TypeScript, and Python. Root namespaces "
           "are\n"
        << "  repeatable for multi-root projects. Lookup directories are optional dependency roots.\n\n"
        << "COMMANDS\n"
        << "  ast   Print parsed AST for all discovered definitions.\n"
        << "  mlir  Print lowered DSDL MLIR module.\n"
        << "  c     Generate C headers + per-definition C implementation units.\n"
        << "  cpp   Generate C++23 headers (std, pmr, or both profiles).\n"
        << "  rust  Generate Rust crate layout and SerDes/runtime integration.\n"
        << "  go    Generate Go module/package layout and SerDes/runtime integration.\n"
        << "  ts    Generate TypeScript module layout and type declarations.\n\n"
        << "  python Generate Python 3.10 package layout and dataclass SerDes/runtime integration.\n\n"
        << "COMMON OPTIONS\n"
        << "  --root-namespace-dir <dir>\n"
        << "      Primary input root. Repeat to add more top-level namespace roots.\n"
        << "      Required for all commands except --help.\n"
        << "  --lookup-dir <dir>\n"
        << "      Additional dependency lookup root. Repeat as needed.\n"
        << "  --optimize-lowered-serdes\n"
        << "      Enable optional semantics-preserving MLIR optimization on lowered SerDes IR\n"
        << "      before backend code generation.\n"
        << "  --help, -h\n"
        << "      Print this help text. With a command, prints command-focused guidance.\n\n"
        << "CODEGEN OPTIONS (c/cpp/rust/go/ts/python)\n"
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
        << "      std          -> Cargo default features include std (default).\n"
        << "      no-std-alloc -> emits a no_std + alloc-ready crate (Cargo default features empty).\n\n"
        << "  --rust-runtime-specialization <portable|fast>\n"
        << "      portable -> use conservative runtime defaults (default).\n"
        << "      fast     -> enable runtime-fast Cargo default feature for optimized bit-copy fast paths.\n\n"
        << "  --rust-memory-mode <max-inline|inline-then-pool>\n"
        << "      max-inline       -> variable-length fields use fixed-capacity inline storage sized to\n"
        << "                          DSDL maxima (default).\n"
        << "      inline-then-pool -> values up to --rust-inline-threshold-bytes remain inline; larger values\n"
        << "                          use dedicated per-type pools.\n\n"
        << "  --rust-inline-threshold-bytes <N>\n"
        << "      Inline threshold for inline-then-pool mode (positive integer, default: 256).\n"
        << "      The value is emitted into Cargo package metadata for auditability and runtime wiring.\n\n"
        << "GO OPTIONS (go)\n"
        << "  --go-module <name>\n"
        << "      Go module name written to go.mod (default: llvmdsdl_generated).\n\n"
        << "TS OPTIONS (ts)\n"
        << "  --ts-module <name>\n"
        << "      TypeScript package name written to package.json (default: llvmdsdl_generated).\n"
        << "  --ts-runtime-specialization <portable|fast>\n"
        << "      portable -> emit conservative bit-loop runtime helpers (default).\n"
        << "      fast     -> emit byte-aligned fast paths in the generated runtime helpers.\n"
        << "                  Wire semantics remain unchanged relative to portable.\n\n"
        << "PYTHON OPTIONS (python)\n"
        << "  --py-package <name>\n"
        << "      Python package name (dotted path allowed) used as the generated root package\n"
        << "      (default: dsdl_gen).\n"
        << "  --py-runtime-specialization <portable|fast>\n"
        << "      portable -> emit conservative pure-runtime bit helpers (default).\n"
        << "      fast     -> emit byte-aligned copy/extract fast paths in generated pure-runtime helpers.\n"
        << "                  Wire semantics on well-formed payloads remain unchanged relative to portable.\n"
        << "                  Malformed/truncated payload contract:\n"
        << "                    - portable pure runtime: tolerant reads (missing bits zero-extended).\n"
        << "                    - fast pure runtime: byte-aligned out-of-range copy/extract raises ValueError.\n"
        << "                    - accel runtime: follows accelerator helper behavior (currently tolerant extract,\n"
        << "                      strict range checks for copy).\n\n"
        << "RUN SUMMARY\n"
        << "  On successful command execution, dsdlc prints a summary to stderr with:\n"
        << "    - files generated\n"
        << "    - output root\n"
        << "    - elapsed wall time\n\n"
        << "EXAMPLES\n"
        << "  dsdlc ast --root-namespace-dir submodules/public_regulated_data_types/uavcan\n"
        << "  dsdlc mlir --root-namespace-dir submodules/public_regulated_data_types/uavcan\n"
        << "  dsdlc c --root-namespace-dir submodules/public_regulated_data_types/uavcan --out-dir build/uavcan-c\n"
        << "  dsdlc cpp --root-namespace-dir submodules/public_regulated_data_types/uavcan --cpp-profile both "
           "--out-dir "
           "build/uavcan-cpp\n"
        << "  dsdlc rust --root-namespace-dir submodules/public_regulated_data_types/uavcan --rust-profile std "
           "--rust-crate-name "
           "uavcan_dsdl_generated --out-dir build/uavcan-rust\n"
        << "  dsdlc rust --root-namespace-dir submodules/public_regulated_data_types/uavcan "
           "--rust-runtime-specialization fast "
           "--rust-profile std --rust-crate-name uavcan_dsdl_fast --out-dir build/uavcan-rust-fast\n"
        << "  dsdlc rust --root-namespace-dir submodules/public_regulated_data_types/uavcan --rust-profile "
           "no-std-alloc "
           "--rust-crate-name uavcan_dsdl_embedded --out-dir build/uavcan-rust-no-std\n"
        << "  dsdlc rust --root-namespace-dir submodules/public_regulated_data_types/uavcan --rust-profile "
           "no-std-alloc "
           "--rust-memory-mode max-inline --rust-crate-name uavcan_dsdl_embedded_inline --out-dir "
           "build/uavcan-rust-no-std-inline\n"
        << "  dsdlc rust --root-namespace-dir submodules/public_regulated_data_types/uavcan --rust-profile "
           "no-std-alloc "
           "--rust-memory-mode inline-then-pool --rust-inline-threshold-bytes 512 "
           "--rust-crate-name uavcan_dsdl_embedded_pool --out-dir build/uavcan-rust-no-std-pool\n"
        << "  dsdlc go --root-namespace-dir submodules/public_regulated_data_types/uavcan --go-module "
           "demo/uavcan/generated "
           "--out-dir build/uavcan-go\n"
        << "  dsdlc ts --root-namespace-dir submodules/public_regulated_data_types/uavcan --ts-module "
           "demo_uavcan_generated "
           "--out-dir build/uavcan-ts\n"
        << "  dsdlc python --root-namespace-dir submodules/public_regulated_data_types/uavcan --py-package "
           "demo_uavcan_generated "
           "--out-dir build/uavcan-python\n"
        << "  dsdlc python --root-namespace-dir submodules/public_regulated_data_types/uavcan --py-package "
           "demo_uavcan_generated_fast "
           "--py-runtime-specialization fast --out-dir build/uavcan-python-fast\n"
        << "  dsdlc ts --root-namespace-dir submodules/public_regulated_data_types/uavcan --ts-module "
           "demo_uavcan_generated_fast "
           "--ts-runtime-specialization fast --out-dir build/uavcan-ts-fast\n\n"
        << "EXIT STATUS\n"
        << "  0 on success, non-zero on parse/semantic/lowering/codegen failure or invalid CLI usage.\n";

    if (!selectedCommand.empty() && isKnownCommand(selectedCommand))
    {
        llvm::errs() << "\nCOMMAND FOCUS (" << selectedCommand << ")\n";
        if (selectedCommand == "ast")
        {
            llvm::errs() << "  Emits AST text to stdout. --out-dir is not used.\n";
        }
        else if (selectedCommand == "mlir")
        {
            llvm::errs() << "  Emits DSDL MLIR to stdout. --out-dir is not used.\n";
        }
        else if (selectedCommand == "c")
        {
            llvm::errs() << "  Requires --out-dir. Emits C headers, C source files, and runtime header.\n";
        }
        else if (selectedCommand == "cpp")
        {
            llvm::errs() << "  Requires --out-dir. Honors --cpp-profile (std|pmr|both).\n";
        }
        else if (selectedCommand == "rust")
        {
            llvm::errs() << "  Requires --out-dir. Honors --rust-crate-name, --rust-profile, and "
                            "--rust-runtime-specialization, --rust-memory-mode, and "
                            "--rust-inline-threshold-bytes.\n";
        }
        else if (selectedCommand == "go")
        {
            llvm::errs() << "  Requires --out-dir. Honors --go-module.\n";
        }
        else if (selectedCommand == "ts")
        {
            llvm::errs() << "  Requires --out-dir. Honors --ts-module and --ts-runtime-specialization.\n";
        }
        else if (selectedCommand == "python")
        {
            llvm::errs() << "  Requires --out-dir. Honors --py-package and --py-runtime-specialization.\n";
        }
    }
}

/// @brief Emits collected diagnostics to stderr.
///
/// @param[in] diag Diagnostic engine containing accumulated diagnostics.
void printDiagnostics(const llvmdsdl::DiagnosticEngine& diag)
{
    for (const auto& d : diag.diagnostics())
    {
        llvm::StringRef level = "note";
        if (d.level == llvmdsdl::DiagnosticLevel::Warning)
        {
            level = "warning";
        }
        else if (d.level == llvmdsdl::DiagnosticLevel::Error)
        {
            level = "error";
        }
        llvm::errs() << d.location.str() << ": " << level << ": " << d.message << "\n";
    }
}

/// @brief Resolves a path to an absolute output-root string when possible.
///
/// @param[in] root Requested output directory.
/// @return Absolute path string when resolution succeeds; otherwise the original
///         input string (or `"stdout"` for empty input).
std::string resolveOutputRoot(const std::string& root)
{
    if (root.empty())
    {
        return "stdout";
    }
    std::error_code ec;
    const auto      abs = std::filesystem::absolute(root, ec);
    if (!ec)
    {
        return abs.string();
    }
    return root;
}

/// @brief Counts regular files under a generated output tree.
///
/// @details Traversal skips entries that trigger permission or transient
/// filesystem errors so run-summary reporting remains best-effort.
///
/// @param[in] root Output root path.
/// @return Number of regular files reachable under the output root.
std::uint64_t countRegularFiles(const std::string& root)
{
    if (root.empty())
    {
        return 0;
    }
    std::error_code             ec;
    const std::filesystem::path outputRoot(root);
    if (!std::filesystem::exists(outputRoot, ec) || ec)
    {
        return 0;
    }

    std::uint64_t                                 count = 0;
    std::filesystem::recursive_directory_iterator it(outputRoot,
                                                     std::filesystem::directory_options::skip_permission_denied,
                                                     ec);
    std::filesystem::recursive_directory_iterator end;
    if (ec)
    {
        return 0;
    }
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            ec.clear();
            continue;
        }
        if (it->is_regular_file(ec) && !ec)
        {
            ++count;
        }
        ec.clear();
    }
    return count;
}

/// @brief Prints the post-run command summary.
///
/// @param[in] command Executed top-level command.
/// @param[in] outputRoot Resolved output root description.
/// @param[in] generatedFiles Number of generated regular files.
/// @param[in] elapsed Wall-clock execution duration.
void printRunSummary(llvm::StringRef                           command,
                     llvm::StringRef                           outputRoot,
                     const std::uint64_t                       generatedFiles,
                     const std::chrono::steady_clock::duration elapsed)
{
    const auto elapsedMs         = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    const auto elapsedWholeSec   = elapsedMs / 1000;
    const auto elapsedFractionMs = elapsedMs % 1000;
    llvm::errs() << "Run summary:\n"
                 << "  command: " << command << "\n"
                 << "  output root: " << outputRoot << "\n"
                 << "  files generated: " << generatedFiles << "\n"
                 << "  elapsed: " << elapsedWholeSec << ".";
    if (elapsedFractionMs < 100)
    {
        llvm::errs() << "0";
    }
    if (elapsedFractionMs < 10)
    {
        llvm::errs() << "0";
    }
    llvm::errs() << elapsedFractionMs << "s\n";
}

}  // namespace

/// @brief Program entry point for `dsdlc`.
///
/// @param[in] argc Argument count.
/// @param[in] argv Argument vector.
/// @return Zero on success, non-zero on CLI, parse, semantic, lowering, or
///         code-generation failure.
int main(int argc, char** argv)
{
    llvm::InitLLVM y(argc, argv);

    if (argc < 2)
    {
        printUsage();
        return 1;
    }

    const std::string command = argv[1];
    if (isHelpToken(command) || command == "help")
    {
        printHelp();
        return 0;
    }
    if (!isKnownCommand(command))
    {
        llvm::errs() << "Unknown command: " << command << "\n";
        printUsage();
        return 1;
    }

    std::vector<std::string>              roots;
    std::vector<std::string>              lookups;
    std::string                           outDir;
    bool                                  helpRequested             = false;
    bool                                  optimizeLoweredSerDes     = false;
    llvmdsdl::CppProfile                  cppProfile                = llvmdsdl::CppProfile::Both;
    std::string                           rustCrateName             = "llvmdsdl_generated";
    llvmdsdl::RustProfile                 rustProfile               = llvmdsdl::RustProfile::Std;
    llvmdsdl::RustRuntimeSpecialization   rustRuntimeSpecialization = llvmdsdl::RustRuntimeSpecialization::Portable;
    llvmdsdl::RustMemoryMode              rustMemoryMode            = llvmdsdl::RustMemoryMode::MaxInline;
    std::uint32_t                         rustInlineThresholdBytes  = 256U;
    std::string                           goModuleName              = "llvmdsdl_generated";
    std::string                           tsModuleName              = "llvmdsdl_generated";
    llvmdsdl::TsRuntimeSpecialization     tsRuntimeSpecialization   = llvmdsdl::TsRuntimeSpecialization::Portable;
    llvmdsdl::PythonRuntimeSpecialization pyRuntimeSpecialization   = llvmdsdl::PythonRuntimeSpecialization::Portable;
    std::string                           pyPackageName             = "dsdl_gen";

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg          = argv[i];
        auto              requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc)
            {
                llvm::errs() << "Missing value for " << name << "\n";
                printUsage();
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--root-namespace-dir")
        {
            roots.push_back(requireValue(arg));
        }
        else if (arg == "--lookup-dir")
        {
            lookups.push_back(requireValue(arg));
        }
        else if (isHelpToken(arg))
        {
            helpRequested = true;
        }
        else if (arg == "--out-dir")
        {
            outDir = requireValue(arg);
        }
        else if (arg == "--optimize-lowered-serdes")
        {
            optimizeLoweredSerDes = true;
        }
        else if (arg == "--cpp-profile")
        {
            const auto value = requireValue(arg);
            if (value == "std")
            {
                cppProfile = llvmdsdl::CppProfile::Std;
            }
            else if (value == "pmr")
            {
                cppProfile = llvmdsdl::CppProfile::Pmr;
            }
            else if (value == "both")
            {
                cppProfile = llvmdsdl::CppProfile::Both;
            }
            else
            {
                llvm::errs() << "Invalid --cpp-profile value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else if (arg == "--rust-crate-name")
        {
            rustCrateName = requireValue(arg);
        }
        else if (arg == "--rust-profile")
        {
            const auto value = requireValue(arg);
            if (value == "std")
            {
                rustProfile = llvmdsdl::RustProfile::Std;
            }
            else if (value == "no-std-alloc")
            {
                rustProfile = llvmdsdl::RustProfile::NoStdAlloc;
            }
            else
            {
                llvm::errs() << "Invalid --rust-profile value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else if (arg == "--rust-runtime-specialization")
        {
            const auto value = requireValue(arg);
            if (value == "portable")
            {
                rustRuntimeSpecialization = llvmdsdl::RustRuntimeSpecialization::Portable;
            }
            else if (value == "fast")
            {
                rustRuntimeSpecialization = llvmdsdl::RustRuntimeSpecialization::Fast;
            }
            else
            {
                llvm::errs() << "Invalid --rust-runtime-specialization value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else if (arg == "--rust-memory-mode")
        {
            const auto value = requireValue(arg);
            if (value == "max-inline")
            {
                rustMemoryMode = llvmdsdl::RustMemoryMode::MaxInline;
            }
            else if (value == "inline-then-pool")
            {
                rustMemoryMode = llvmdsdl::RustMemoryMode::InlineThenPool;
            }
            else
            {
                llvm::errs() << "Invalid --rust-memory-mode value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else if (arg == "--rust-inline-threshold-bytes")
        {
            const auto            value = requireValue(arg);
            std::uint64_t         parsedThreshold{};
            const llvm::StringRef valueRef(value);
            if (valueRef.getAsInteger(10, parsedThreshold) || parsedThreshold == 0U ||
                parsedThreshold > std::numeric_limits<std::uint32_t>::max())
            {
                llvm::errs() << "Invalid --rust-inline-threshold-bytes value: " << value << "\n";
                printUsage();
                return 1;
            }
            rustInlineThresholdBytes = static_cast<std::uint32_t>(parsedThreshold);
        }
        else if (arg == "--go-module")
        {
            goModuleName = requireValue(arg);
        }
        else if (arg == "--ts-module")
        {
            tsModuleName = requireValue(arg);
        }
        else if (arg == "--ts-runtime-specialization")
        {
            const auto value = requireValue(arg);
            if (value == "portable")
            {
                tsRuntimeSpecialization = llvmdsdl::TsRuntimeSpecialization::Portable;
            }
            else if (value == "fast")
            {
                tsRuntimeSpecialization = llvmdsdl::TsRuntimeSpecialization::Fast;
            }
            else
            {
                llvm::errs() << "Invalid --ts-runtime-specialization value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else if (arg == "--py-package")
        {
            pyPackageName = requireValue(arg);
        }
        else if (arg == "--py-runtime-specialization")
        {
            const auto value = requireValue(arg);
            if (value == "portable")
            {
                pyRuntimeSpecialization = llvmdsdl::PythonRuntimeSpecialization::Portable;
            }
            else if (value == "fast")
            {
                pyRuntimeSpecialization = llvmdsdl::PythonRuntimeSpecialization::Fast;
            }
            else
            {
                llvm::errs() << "Invalid --py-runtime-specialization value: " << value << "\n";
                printUsage();
                return 1;
            }
        }
        else
        {
            llvm::errs() << "Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (helpRequested)
    {
        printHelp(command);
        return 0;
    }

    if (roots.empty())
    {
        llvm::errs() << "At least one --root-namespace-dir is required\n";
        return 1;
    }

    const auto                 startTime = std::chrono::steady_clock::now();
    llvmdsdl::DiagnosticEngine diagnostics;
    auto                       finish = [&](const std::string& outputRoot, const std::uint64_t generatedFiles) -> int {
        printDiagnostics(diagnostics);
        printRunSummary(command, outputRoot, generatedFiles, std::chrono::steady_clock::now() - startTime);
        return diagnostics.hasErrors() ? 1 : 0;
    };

    auto ast = llvmdsdl::parseDefinitions(roots, lookups, diagnostics);
    if (!ast)
    {
        llvm::consumeError(ast.takeError());
        printDiagnostics(diagnostics);
        return 1;
    }

    if (command == "ast")
    {
        llvm::outs() << llvmdsdl::printAST(*ast);
        return finish("stdout", 0);
    }

    auto semantic = llvmdsdl::analyze(*ast, diagnostics);
    if (!semantic)
    {
        llvm::consumeError(semantic.takeError());
        printDiagnostics(diagnostics);
        return 1;
    }

    mlir::DialectRegistry registry;
    registry.insert<mlir::dsdl::DSDLDialect,
                    mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::emitc::EmitCDialect>();
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::dsdl::DSDLDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::emitc::EmitCDialect>();

    auto mlirModule = llvmdsdl::lowerToMLIR(*semantic, context, diagnostics);
    if (!mlirModule)
    {
        printDiagnostics(diagnostics);
        return 1;
    }

    if (command == "mlir")
    {
        mlirModule->print(llvm::outs());
        llvm::outs() << "\n";
        return finish("stdout", 0);
    }

    if (command == "c")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'c' command\n";
            return 1;
        }
        llvmdsdl::CEmitOptions options;
        options.outDir                = outDir;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitC(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    if (command == "cpp")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'cpp' command\n";
            return 1;
        }
        llvmdsdl::CppEmitOptions options;
        options.outDir                = outDir;
        options.profile               = cppProfile;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitCpp(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    if (command == "rust")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'rust' command\n";
            return 1;
        }
        llvmdsdl::RustEmitOptions options;
        options.outDir                = outDir;
        options.crateName             = rustCrateName;
        options.profile               = rustProfile;
        options.runtimeSpecialization = rustRuntimeSpecialization;
        options.memoryMode            = rustMemoryMode;
        options.inlineThresholdBytes  = rustInlineThresholdBytes;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitRust(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    if (command == "go")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'go' command\n";
            return 1;
        }
        llvmdsdl::GoEmitOptions options;
        options.outDir                = outDir;
        options.moduleName            = goModuleName;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitGo(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    if (command == "ts")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'ts' command\n";
            return 1;
        }
        llvmdsdl::TsEmitOptions options;
        options.outDir                = outDir;
        options.moduleName            = tsModuleName;
        options.runtimeSpecialization = tsRuntimeSpecialization;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitTs(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    if (command == "python")
    {
        if (outDir.empty())
        {
            llvm::errs() << "--out-dir is required for 'python' command\n";
            return 1;
        }
        llvmdsdl::PythonEmitOptions options;
        options.outDir                = outDir;
        options.packageName           = pyPackageName;
        options.runtimeSpecialization = pyRuntimeSpecialization;
        options.optimizeLoweredSerDes = optimizeLoweredSerDes;

        if (llvm::Error err = llvmdsdl::emitPython(*semantic, *mlirModule, options, diagnostics))
        {
            llvm::errs() << llvm::toString(std::move(err)) << "\n";
            printDiagnostics(diagnostics);
            return 1;
        }

        return finish(resolveOutputRoot(outDir), countRegularFiles(outDir));
    }

    llvm::errs() << "Unhandled command path: " << command << "\n";
    return 1;
}
