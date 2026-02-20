//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>

bool runParserTests();
bool runBitLengthSetTests();
bool runEvaluatorTests();
bool runAnalyzerTests();
bool runRuntimeTests();
bool runArrayWirePlanTests();
bool runLoweredBodyPlanTests();
bool runLoweredRenderIRTests();
bool runSectionHelperBindingPlanTests();
bool runSerDesStatementPlanTests();
bool runHelperBindingRenderTests();
bool runTsLoweredPlanTests();
bool runHelperSymbolResolverTests();
bool runWireLayoutFactsTests();
bool runTypeStorageTests();
bool runLoweredMetadataHardeningTests();

int main()
{
    bool ok = true;
    ok      = runParserTests() && ok;
    ok      = runBitLengthSetTests() && ok;
    ok      = runEvaluatorTests() && ok;
    ok      = runAnalyzerTests() && ok;
    ok      = runRuntimeTests() && ok;
    ok      = runArrayWirePlanTests() && ok;
    ok      = runLoweredBodyPlanTests() && ok;
    ok      = runLoweredRenderIRTests() && ok;
    ok      = runSectionHelperBindingPlanTests() && ok;
    ok      = runSerDesStatementPlanTests() && ok;
    ok      = runHelperBindingRenderTests() && ok;
    ok      = runTsLoweredPlanTests() && ok;
    ok      = runHelperSymbolResolverTests() && ok;
    ok      = runWireLayoutFactsTests() && ok;
    ok      = runTypeStorageTests() && ok;
    ok      = runLoweredMetadataHardeningTests() && ok;
    if (!ok)
    {
        std::cerr << "unit tests failed\n";
        return 1;
    }
    std::cout << "unit tests passed\n";
    return 0;
}
