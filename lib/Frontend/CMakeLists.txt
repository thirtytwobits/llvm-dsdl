#===----------------------------------------------------------------------===#
##
## @file
## Build rules for the frontend parsing and discovery library.
##
## The frontend target provides lexing, parsing, discovery, and AST rendering capabilities to later pipeline stages.
##
#===----------------------------------------------------------------------===#

add_library(llvmdsdlFrontend
  Discovery.cpp
  Lexer.cpp
  Parser.cpp
  ASTPrinter.cpp
)

target_link_libraries(llvmdsdlFrontend
  PUBLIC
  llvmdsdlSupport
)

target_include_directories(llvmdsdlFrontend PUBLIC
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
