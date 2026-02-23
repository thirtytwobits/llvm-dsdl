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
bool runCodegenDiagnosticTextTests();
bool runConstantLiteralRenderTests();
bool runCompositeImportGraphTests();
bool runDefinitionIndexTests();
bool runDefinitionPathProjectionTests();
bool runNativeHelperContractTests();
bool runLoweredBodyPlanTests();
bool runLoweredRenderIRTests();
bool runNativeEmitterTraversalTests();
bool runSectionHelperBindingPlanTests();
bool runSerDesStatementPlanTests();
bool runScriptedBodyPlanTests();
bool runHelperBindingRenderTests();
bool runRuntimeHelperBindingsTests();
bool runNamingPolicyTests();
bool runRuntimeLoweredPlanTests();
bool runRuntimeLoweredOrderingTests();
bool runHelperSymbolResolverTests();
bool runWireLayoutFactsTests();
bool runTypeStorageTests();
bool runStorageTypeTokensTests();
bool runLoweredMetadataHardeningTests();
bool runLspDocumentStoreTests();
bool runLspRequestSchedulerTests();
bool runLspAnalysisTests();
bool runLspIndexTests();
bool runLspLintTests();
bool runLspRankingTests();
bool runLspServerTests();
bool runLspRobustnessTests();
bool runLspJsonRpcFuzzTests();

int main()
{
    bool ok = true;
    ok      = runParserTests() && ok;
    ok      = runBitLengthSetTests() && ok;
    ok      = runEvaluatorTests() && ok;
    ok      = runAnalyzerTests() && ok;
    ok      = runRuntimeTests() && ok;
    ok      = runArrayWirePlanTests() && ok;
    ok      = runCodegenDiagnosticTextTests() && ok;
    ok      = runConstantLiteralRenderTests() && ok;
    ok      = runCompositeImportGraphTests() && ok;
    ok      = runDefinitionIndexTests() && ok;
    ok      = runDefinitionPathProjectionTests() && ok;
    ok      = runNativeHelperContractTests() && ok;
    ok      = runLoweredBodyPlanTests() && ok;
    ok      = runLoweredRenderIRTests() && ok;
    ok      = runNativeEmitterTraversalTests() && ok;
    ok      = runSectionHelperBindingPlanTests() && ok;
    ok      = runSerDesStatementPlanTests() && ok;
    ok      = runScriptedBodyPlanTests() && ok;
    ok      = runHelperBindingRenderTests() && ok;
    ok      = runRuntimeHelperBindingsTests() && ok;
    ok      = runNamingPolicyTests() && ok;
    ok      = runRuntimeLoweredPlanTests() && ok;
    ok      = runRuntimeLoweredOrderingTests() && ok;
    ok      = runHelperSymbolResolverTests() && ok;
    ok      = runWireLayoutFactsTests() && ok;
    ok      = runTypeStorageTests() && ok;
    ok      = runStorageTypeTokensTests() && ok;
    ok      = runLoweredMetadataHardeningTests() && ok;
    ok      = runLspDocumentStoreTests() && ok;
    ok      = runLspRequestSchedulerTests() && ok;
    ok      = runLspAnalysisTests() && ok;
    ok      = runLspIndexTests() && ok;
    ok      = runLspLintTests() && ok;
    ok      = runLspRankingTests() && ok;
    ok      = runLspServerTests() && ok;
    ok      = runLspRobustnessTests() && ok;
    ok      = runLspJsonRpcFuzzTests() && ok;
    if (!ok)
    {
        std::cerr << "unit tests failed\n";
        return 1;
    }
    std::cout << "unit tests passed\n";
    return 0;
}
