add_library(llvmdsdlSemantics
  BitLengthSet.cpp
  Evaluator.cpp
  Analyzer.cpp
)

target_link_libraries(llvmdsdlSemantics
  PUBLIC
  llvmdsdlFrontend
  llvmdsdlSupport
  LLVMSupport
)

target_include_directories(llvmdsdlSemantics PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
