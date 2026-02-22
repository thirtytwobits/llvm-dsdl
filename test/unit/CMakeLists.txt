add_executable(llvmdsdl-unit-tests
  UnitMain.cpp
  ParserTests.cpp
  BitLengthSetTests.cpp
  EvaluatorTests.cpp
  AnalyzerTests.cpp
  RuntimeTests.cpp
  ArrayWirePlanTests.cpp
  LoweredBodyPlanTests.cpp
  LoweredRenderIRTests.cpp
  SectionHelperBindingPlanTests.cpp
  SerDesStatementPlanTests.cpp
  HelperBindingRenderTests.cpp
  TsLoweredPlanTests.cpp
  HelperSymbolResolverTests.cpp
  WireLayoutFactsTests.cpp
  TypeStorageTests.cpp
  LoweredMetadataHardeningTests.cpp
  LspDocumentStoreTests.cpp
  LspRequestSchedulerTests.cpp
  LspAnalysisTests.cpp
  LspIndexTests.cpp
  LspLintTests.cpp
  LspRankingTests.cpp
  LspServerTests.cpp
)

target_link_libraries(llvmdsdl-unit-tests
  PRIVATE
  llvmdsdlCodeGen
  llvmdsdlLSP
)

target_include_directories(llvmdsdl-unit-tests
  PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/runtime
)

add_test(NAME llvmdsdl-unit-tests COMMAND llvmdsdl-unit-tests)
