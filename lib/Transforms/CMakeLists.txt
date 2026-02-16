add_library(llvmdsdlTransforms
  Passes.cpp
  ConvertDSDLToEmitC.cpp
)

target_link_libraries(llvmdsdlTransforms
  PUBLIC
  llvmdsdlIR
  MLIRIR
  MLIRPass
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)

target_include_directories(llvmdsdlTransforms PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
