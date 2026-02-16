add_executable(dsdl-opt
  main.cpp
)

target_link_libraries(dsdl-opt
  PRIVATE
  llvmdsdlIR
  llvmdsdlTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRMlirOptMain
)

target_include_directories(dsdl-opt PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
