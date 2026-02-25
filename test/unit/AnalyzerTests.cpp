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

    const std::string extentText = "uint16 sample\n"
                                   "@extent 8\n";

    llvmdsdl::DiagnosticEngine extentParseDiag;
    llvmdsdl::Lexer            extentLexer("uavcan.test.BadExtent.1.0.dsdl", extentText);
    auto                       extentTokens = extentLexer.lex();
    llvmdsdl::Parser           extentParser("uavcan.test.BadExtent.1.0.dsdl", std::move(extentTokens), extentParseDiag);
    auto                       extentDef = extentParser.parseDefinition();
    if (!extentDef)
    {
        llvm::consumeError(extentDef.takeError());
        std::cerr << "bad-extent fixture parse failed unexpectedly\n";
        return false;
    }
    if (extentParseDiag.hasErrors())
    {
        std::cerr << "bad-extent fixture parse diagnostics contained errors\n";
        return false;
    }

    llvmdsdl::DiscoveredDefinition extentDiscovered;
    extentDiscovered.filePath            = "uavcan/test/BadExtent.1.0.dsdl";
    extentDiscovered.rootNamespacePath   = "uavcan";
    extentDiscovered.fullName            = "uavcan.test.BadExtent";
    extentDiscovered.shortName           = "BadExtent";
    extentDiscovered.namespaceComponents = {"uavcan", "test"};
    extentDiscovered.majorVersion        = 1;
    extentDiscovered.minorVersion        = 0;
    extentDiscovered.text                = extentText;

    llvmdsdl::ASTModule extentModule;
    extentModule.definitions.push_back(llvmdsdl::ParsedDefinition{extentDiscovered, *extentDef});

    llvmdsdl::DiagnosticEngine extentSemDiag;
    auto                       extentSemantic = llvmdsdl::analyze(extentModule, extentSemDiag);
    if (extentSemantic)
    {
        std::cerr << "analyzer unexpectedly succeeded for invalid extent fixture\n";
        return false;
    }
    llvm::consumeError(extentSemantic.takeError());

    bool sawExtentDiagnosticAtExtentValue = false;
    for (const llvmdsdl::Diagnostic& diagnostic : extentSemDiag.diagnostics())
    {
        if (diagnostic.message == "extent smaller than maximal serialized length" && diagnostic.location.line == 2 &&
            diagnostic.location.column == 9 && diagnostic.length == 1)
        {
            sawExtentDiagnosticAtExtentValue = true;
            break;
        }
    }
    if (!sawExtentDiagnosticAtExtentValue)
    {
        std::cerr << "expected extent diagnostic to point at the @extent value span\n";
        return false;
    }

    const std::string extentMultipleDigitsText = "uint16 sample\n"
                                                 "@extent 13\n";

    llvmdsdl::DiagnosticEngine extentDigitsParseDiag;
    llvmdsdl::Lexer            extentDigitsLexer("uavcan.test.BadExtentDigits.1.0.dsdl", extentMultipleDigitsText);
    auto                       extentDigitsTokens = extentDigitsLexer.lex();
    llvmdsdl::Parser           extentDigitsParser("uavcan.test.BadExtentDigits.1.0.dsdl",
                                        std::move(extentDigitsTokens),
                                        extentDigitsParseDiag);
    auto                       extentDigitsDef = extentDigitsParser.parseDefinition();
    if (!extentDigitsDef)
    {
        llvm::consumeError(extentDigitsDef.takeError());
        std::cerr << "bad-extent-digits fixture parse failed unexpectedly\n";
        return false;
    }
    if (extentDigitsParseDiag.hasErrors())
    {
        std::cerr << "bad-extent-digits fixture parse diagnostics contained errors\n";
        return false;
    }

    llvmdsdl::DiscoveredDefinition extentDigitsDiscovered;
    extentDigitsDiscovered.filePath            = "uavcan/test/BadExtentDigits.1.0.dsdl";
    extentDigitsDiscovered.rootNamespacePath   = "uavcan";
    extentDigitsDiscovered.fullName            = "uavcan.test.BadExtentDigits";
    extentDigitsDiscovered.shortName           = "BadExtentDigits";
    extentDigitsDiscovered.namespaceComponents = {"uavcan", "test"};
    extentDigitsDiscovered.majorVersion        = 1;
    extentDigitsDiscovered.minorVersion        = 0;
    extentDigitsDiscovered.text                = extentMultipleDigitsText;

    llvmdsdl::ASTModule extentDigitsModule;
    extentDigitsModule.definitions.push_back(llvmdsdl::ParsedDefinition{extentDigitsDiscovered, *extentDigitsDef});

    llvmdsdl::DiagnosticEngine extentDigitsSemDiag;
    auto                       extentDigitsSemantic = llvmdsdl::analyze(extentDigitsModule, extentDigitsSemDiag);
    if (extentDigitsSemantic)
    {
        std::cerr << "analyzer unexpectedly succeeded for invalid multi-digit extent fixture\n";
        return false;
    }
    llvm::consumeError(extentDigitsSemantic.takeError());

    bool sawMultiDigitExtentSpan = false;
    for (const llvmdsdl::Diagnostic& diagnostic : extentDigitsSemDiag.diagnostics())
    {
        if (diagnostic.message == "extent must be a multiple of 8 bits" && diagnostic.location.line == 2 &&
            diagnostic.location.column == 9 && diagnostic.length == 2)
        {
            sawMultiDigitExtentSpan = true;
            break;
        }
    }
    if (!sawMultiDigitExtentSpan)
    {
        std::cerr << "expected multi-digit extent diagnostic span to match expression width\n";
        return false;
    }

    const std::string          docText = "# type docs\n"
                                         "uint8 field # field docs\n"
                                         "uint8 CONST = 1 # const docs\n"
                                         "@sealed\n";
    llvmdsdl::DiagnosticEngine docParseDiag;
    llvmdsdl::Lexer            docLexer("uavcan.test.Docs.1.0.dsdl", docText);
    auto                       docTokens = docLexer.lex();
    llvmdsdl::Parser           docParser("uavcan.test.Docs.1.0.dsdl", std::move(docTokens), docParseDiag);
    auto                       docDef = docParser.parseDefinition();
    if (!docDef)
    {
        llvm::consumeError(docDef.takeError());
        std::cerr << "doc fixture parse failed unexpectedly\n";
        return false;
    }
    if (docParseDiag.hasErrors())
    {
        std::cerr << "doc fixture parse diagnostics contained errors\n";
        return false;
    }

    llvmdsdl::DiscoveredDefinition docDiscovered;
    docDiscovered.filePath            = "uavcan/test/Docs.1.0.dsdl";
    docDiscovered.rootNamespacePath   = "uavcan";
    docDiscovered.fullName            = "uavcan.test.Docs";
    docDiscovered.shortName           = "Docs";
    docDiscovered.namespaceComponents = {"uavcan", "test"};
    docDiscovered.majorVersion        = 1;
    docDiscovered.minorVersion        = 0;
    docDiscovered.text                = docText;

    llvmdsdl::ASTModule docModule;
    docModule.definitions.push_back(llvmdsdl::ParsedDefinition{docDiscovered, *docDef});

    llvmdsdl::DiagnosticEngine docSemDiag;
    auto                       docSemantic = llvmdsdl::analyze(docModule, docSemDiag);
    if (!docSemantic)
    {
        llvm::consumeError(docSemantic.takeError());
        std::cerr << "analyzer unexpectedly failed on docs fixture\n";
        return false;
    }
    if (docSemDiag.hasErrors())
    {
        std::cerr << "analyzer produced diagnostics for docs fixture\n";
        return false;
    }
    if (docSemantic->definitions.empty())
    {
        std::cerr << "docs fixture produced no semantic definitions\n";
        return false;
    }
    const auto& docDefinition = docSemantic->definitions.front();
    if (docDefinition.doc.lines.size() != 1 || docDefinition.doc.lines[0].text != "type docs")
    {
        std::cerr << "definition docs did not propagate into semantic model\n";
        return false;
    }
    if (docDefinition.request.fields.empty() || docDefinition.request.fields.front().doc.lines.size() != 1 ||
        docDefinition.request.fields.front().doc.lines[0].text != "field docs")
    {
        std::cerr << "field docs did not propagate into semantic model\n";
        return false;
    }
    if (docDefinition.request.constants.empty() || docDefinition.request.constants.front().doc.lines.size() != 1 ||
        docDefinition.request.constants.front().doc.lines[0].text != "const docs")
    {
        std::cerr << "constant docs did not propagate into semantic model\n";
        return false;
    }

    return true;
}
