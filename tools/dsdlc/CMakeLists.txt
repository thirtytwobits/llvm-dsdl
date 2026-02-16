add_executable(dsdlc
  main.cpp
)

target_link_libraries(dsdlc
  PRIVATE
  llvmdsdlCodeGen
)

target_include_directories(dsdlc PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
