set(LLVMDSDL_UAVCAN_ROOT
    "${LLVMDSDL_SOURCE_DIR}/public_regulated_data_types/uavcan")

set(LLVMDSDL_FIXTURES_ROOT
    "${LLVMDSDL_SOURCE_DIR}/test/lit/fixtures")
if(EXISTS "${LLVMDSDL_FIXTURES_ROOT}")
  add_test(
    NAME llvmdsdl-dsdl-opt-sanity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLOPT=$<TARGET_FILE:dsdl-opt>
        -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/dsdl-opt-sanity-out
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunDsdlOptSanity.cmake
  )

  set_tests_properties(
    llvmdsdl-dsdl-opt-sanity
    PROPERTIES
      LABELS "integration;tools;mlir"
  )
endif()

if(NOT EXISTS "${LLVMDSDL_UAVCAN_ROOT}")
  message(WARNING "Skipping uavcan integration test: ${LLVMDSDL_UAVCAN_ROOT} missing")
  return()
endif()

if(NOT CMAKE_C_COMPILER)
  message(WARNING "Skipping uavcan integration test: C compiler unavailable")
  return()
endif()

if(NOT CMAKE_CXX_COMPILER)
  message(WARNING "Skipping uavcan C++ integration test: C++ compiler unavailable")
  return()
endif()

add_test(
  NAME llvmdsdl-uavcan-strict-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-strict-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanStrictGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-strict-generation
  PROPERTIES
    LABELS "integration;uavcan;slow"
)

add_test(
  NAME llvmdsdl-uavcan-rust-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-generation
  PROPERTIES
    LABELS "integration;uavcan;rust"
)

add_test(
  NAME llvmdsdl-uavcan-cpp-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-cpp-out
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanCppGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-cpp-generation
  PROPERTIES
    LABELS "integration;uavcan;cpp;slow"
)
