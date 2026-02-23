#===----------------------------------------------------------------------===#
##
## @file
## Build rules for language emitters and lowering planners.
##
## The target aggregates shared code-generation utilities and links MLIR conversion components required by emitters.
##
#===----------------------------------------------------------------------===#

add_library(llvmdsdlCodeGen
  ArrayWirePlan.cpp
  CEmitter.cpp
  CppEmitter.cpp
  GoEmitter.cpp
  CodegenDiagnosticText.cpp
  HelperBindingRender.cpp
  HelperSymbolResolver.cpp
  LoweredBodyPlan.cpp
  LoweredRenderIR.cpp
  NativeHelperContract.cpp
  NativeEmitterTraversal.cpp
  MlirLoweredFacts.cpp
  RuntimeHelperBindings.cpp
  RuntimeLoweredPlan.cpp
  ScriptedBodyPlan.cpp
  PythonEmitter.cpp
  RustEmitter.cpp
  SectionHelperBindingPlan.cpp
  SerDesStatementPlan.cpp
  SerDesHelperDescriptors.cpp
  TsEmitter.cpp
  TypeStorage.cpp
  WireLayoutFacts.cpp
)

target_link_libraries(llvmdsdlCodeGen
  PUBLIC
  llvmdsdlLowering
  llvmdsdlTransforms
  MLIRFuncToEmitC
  MLIRArithToEmitC
  MLIRSCFToEmitC
  MLIRTargetCpp
  MLIRTransforms
)

target_include_directories(llvmdsdlCodeGen PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)

target_compile_definitions(llvmdsdlCodeGen
  PRIVATE
  LLVMDSDL_SOURCE_DIR="${LLVMDSDL_SOURCE_DIR}"
)
