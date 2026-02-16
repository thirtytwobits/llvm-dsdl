add_library(llvmdsdlCodeGen
  CEmitter.cpp
  CppEmitter.cpp
  RustEmitter.cpp
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
