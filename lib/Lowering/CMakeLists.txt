add_library(llvmdsdlLowering
  LowerToMLIR.cpp
)

target_link_libraries(llvmdsdlLowering
  PUBLIC
  llvmdsdlSemantics
  llvmdsdlIR
  MLIRIR
  MLIRFuncDialect
)

target_include_directories(llvmdsdlLowering PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
