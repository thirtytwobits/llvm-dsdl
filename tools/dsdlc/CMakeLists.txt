add_executable(dsdlc
  main.cpp
)

target_link_libraries(dsdlc
  PRIVATE
  llvmdsdlFrontend
  llvmdsdlSemantics
  llvmdsdlLowering
  llvmdsdlCodeGen
  llvmdsdlTransforms
  llvmdsdlIR
  llvmdsdlSupport
  MLIRIR
  MLIRSupport
)

target_include_directories(dsdlc PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
