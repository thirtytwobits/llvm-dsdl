set(LLVMDSDL_UAVCAN_ROOT
    "${LLVMDSDL_SOURCE_DIR}/public_regulated_data_types/uavcan")

if(NOT EXISTS "${LLVMDSDL_UAVCAN_ROOT}")
  message(WARNING "Skipping uavcan integration test: ${LLVMDSDL_UAVCAN_ROOT} missing")
  return()
endif()

if(NOT CMAKE_C_COMPILER)
  message(WARNING "Skipping uavcan integration test: C compiler unavailable")
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
