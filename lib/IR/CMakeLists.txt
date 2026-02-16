add_library(llvmdsdlIR
  DSDLDialect.cpp
  DSDLOps.cpp
  DSDLTypes.cpp
  DSDLAttrs.cpp
)
add_dependencies(llvmdsdlIR LLVMDSDLIRIncGen)

target_link_libraries(llvmdsdlIR
  PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIREmitCDialect
)

target_include_directories(llvmdsdlIR PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
