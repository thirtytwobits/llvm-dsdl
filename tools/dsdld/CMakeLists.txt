add_executable(dsdld
  main.cpp
)

target_link_libraries(dsdld
  PRIVATE
  llvmdsdlLSP
)

target_include_directories(dsdld PRIVATE
  ${LLVMDSDL_SOURCE_DIR}/include
  ${LLVMDSDL_BINARY_DIR}/include
)
