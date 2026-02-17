add_library(llvmdsdlCodeGen
  ArrayWirePlan.cpp
  CEmitter.cpp
  CppEmitter.cpp
  HelperBindingRender.cpp
  HelperSymbolResolver.cpp
  MlirLoweredFacts.cpp
  RustEmitter.cpp
  SectionHelperBindingPlan.cpp
  SerDesStatementPlan.cpp
  SerDesHelperDescriptors.cpp
  TypeStorage.cpp
  WireLayoutFacts.cpp
)

target_link_libraries(llvmdsdlCodeGen
  PUBLIC
  llvmdsdlLowering
  llvmdsdlTransforms
  llvmdsdlSemantics
  llvmdsdlIR
  MLIRFuncToEmitC
  MLIRArithToEmitC
  MLIRSCFToEmitC
  MLIRTargetCpp
  MLIRTransforms
  MLIRPass
  MLIRIR
)

target_include_directories(llvmdsdlCodeGen PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)

target_compile_definitions(llvmdsdlCodeGen
  PRIVATE
  LLVMDSDL_SOURCE_DIR="${LLVMDSDL_SOURCE_DIR}"
)
