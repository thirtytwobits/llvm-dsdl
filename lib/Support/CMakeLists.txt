add_library(llvmdsdlSupport
  Diagnostics.cpp
  Rational.cpp
  SourceLocation.cpp
)

target_link_libraries(llvmdsdlSupport
  PUBLIC
  LLVMSupport
)

target_include_directories(llvmdsdlSupport PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
