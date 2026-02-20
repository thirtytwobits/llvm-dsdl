//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "llvmdsdl/Frontend/Lexer.h"
#include "llvmdsdl/Frontend/Parser.h"
#include "llvmdsdl/Semantics/Analyzer.h"
#include "llvmdsdl/Semantics/Model.h"
#include "llvmdsdl/Support/Diagnostics.h"
#include "llvm/Support/Error.h"
#include "llvmdsdl/Frontend/AST.h"

bool runAnalyzerTests()
{
    const std::string text = "@union\n"
                             "uint8 a\n"
                             "uint16 b\n"
                             "@sealed\n"
                             "@assert _offset_.min == 16\n"
                             "@assert _offset_.max == 24\n"
                             "@assert _offset_ % 8 == {0}\n";

    llvmdsdl::DiagnosticEngine parseDiag;
    llvmdsdl::Lexer            lexer("uavcan.test.UnionOffset.1.0.dsdl", text);
    auto                       tokens = lexer.lex();
    llvmdsdl::Parser           parser("uavcan.test.UnionOffset.1.0.dsdl", std::move(tokens), parseDiag);
    auto                       def = parser.parseDefinition();
    if (!def)
    {
        llvm::consumeError(def.takeError());
        std::cerr << "analyzer fixture parse failed unexpectedly\n";
        return false;
    }
    if (parseDiag.hasErrors())
    {
        std::cerr << "analyzer fixture parse diagnostics contained errors\n";
        return false;
    }

    llvmdsdl::DiscoveredDefinition discovered;
    discovered.filePath            = "uavcan/test/UnionOffset.1.0.dsdl";
    discovered.rootNamespacePath   = "uavcan";
    discovered.fullName            = "uavcan.test.UnionOffset";
    discovered.shortName           = "UnionOffset";
    discovered.namespaceComponents = {"uavcan", "test"};
    discovered.majorVersion        = 1;
    discovered.minorVersion        = 0;
    discovered.text                = text;

    llvmdsdl::ASTModule module;
    module.definitions.push_back(llvmdsdl::ParsedDefinition{discovered, *def});

    llvmdsdl::DiagnosticEngine semDiag;
    auto                       semantic = llvmdsdl::analyze(module, semDiag);
    if (!semantic)
    {
        llvm::consumeError(semantic.takeError());
        std::cerr << "analyzer unexpectedly failed on union offset fixture\n";
        return false;
    }
    if (semDiag.hasErrors())
    {
        std::cerr << "analyzer produced unexpected errors on union offset fixture\n";
        return false;
    }

    if (semantic->definitions.size() != 1)
    {
        std::cerr << "unexpected semantic definition count\n";
        return false;
    }
    const auto& section = semantic->definitions.front().request;
    if (!section.isUnion)
    {
        std::cerr << "expected analyzed section to be a union\n";
        return false;
    }
    if (section.fields.size() != 2)
    {
        std::cerr << "unexpected union field count in semantic section\n";
        return false;
    }
    if (section.fields[0].unionOptionIndex != 0 || section.fields[1].unionOptionIndex != 1 ||
        section.fields[0].unionTagBits != 8 || section.fields[1].unionTagBits != 8)
    {
        std::cerr << "unexpected union tag metadata\n";
        return false;
    }
    if (section.minBitLength != 16 || section.maxBitLength != 24 || section.serializationBufferSizeBits != 24 ||
        section.fixedSize)
    {
        std::cerr << "unexpected union section size metadata\n";
        return false;
    }

    return true;
}
