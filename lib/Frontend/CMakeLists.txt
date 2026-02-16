add_library(llvmdsdlFrontend
  Discovery.cpp
  Lexer.cpp
  Parser.cpp
  ASTPrinter.cpp
)

target_link_libraries(llvmdsdlFrontend
  PUBLIC
  llvmdsdlSupport
  LLVMSupport
)

target_include_directories(llvmdsdlFrontend PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
