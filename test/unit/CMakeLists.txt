add_executable(llvmdsdl-unit-tests
  UnitMain.cpp
  ParserTests.cpp
  BitLengthSetTests.cpp
  EvaluatorTests.cpp
  AnalyzerTests.cpp
  RuntimeTests.cpp
  ArrayWirePlanTests.cpp
  SectionHelperBindingPlanTests.cpp
  SerDesStatementPlanTests.cpp
  HelperBindingRenderTests.cpp
  HelperSymbolResolverTests.cpp
  WireLayoutFactsTests.cpp
  TypeStorageTests.cpp
)

target_link_libraries(llvmdsdl-unit-tests
  PRIVATE
  llvmdsdlCodeGen
)

target_include_directories(llvmdsdl-unit-tests
  PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/runtime
)

add_test(NAME llvmdsdl-unit-tests COMMAND llvmdsdl-unit-tests)
