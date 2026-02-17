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

if(CMAKE_C_COMPILER AND CMAKE_CXX_COMPILER)
  add_test(
    NAME llvmdsdl-signed-narrow-cpp-c-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-cpp-c-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCPP_PROFILE=std
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCppCParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-cpp-c-parity
    PROPERTIES
      LABELS "integration;fixtures;cpp;differential"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-cpp-pmr-c-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-cpp-pmr-c-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCPP_PROFILE=pmr
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCppCParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-cpp-pmr-c-parity
    PROPERTIES
      LABELS "integration;fixtures;cpp;differential"
  )
else()
  message(STATUS "Skipping signed_narrow C/C++ parity tests: C/C++ compiler unavailable")
endif()

find_program(CARGO_EXECUTABLE cargo QUIET)
find_program(RUSTC_EXECUTABLE rustc QUIET)
if(CMAKE_C_COMPILER AND CARGO_EXECUTABLE AND RUSTC_EXECUTABLE)
  add_test(
    NAME llvmdsdl-signed-narrow-c-rust-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-rust-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-rust-parity
    PROPERTIES
      LABELS "integration;fixtures;rust;differential"
  )
else()
  message(STATUS "Skipping signed_narrow C/Rust parity test: C compiler or cargo/rustc unavailable")
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
  NAME llvmdsdl-uavcan-mlir-lowering
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DDSDLOPT=$<TARGET_FILE:dsdl-opt>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-mlir-lowering-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanMlirLowering.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-mlir-lowering
  PROPERTIES
    LABELS "integration;uavcan;mlir;slow"
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

find_program(CARGO_EXECUTABLE cargo QUIET)
find_program(RUSTC_EXECUTABLE rustc QUIET)
if(CARGO_EXECUTABLE AND RUSTC_EXECUTABLE)
  add_test(
    NAME llvmdsdl-uavcan-rust-cargo-check
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-cargo-check-out
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustCargoCheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-rust-cargo-check
    PROPERTIES
      LABELS "integration;uavcan;rust;slow"
  )
endif()

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

add_test(
  NAME llvmdsdl-uavcan-cpp-c-parity
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-cpp-c-parity-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCPP_PROFILE=std
      -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCppCParity.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-cpp-c-parity
  PROPERTIES
    LABELS "integration;uavcan;cpp;differential;slow"
)

add_test(
  NAME llvmdsdl-uavcan-cpp-pmr-c-parity
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-cpp-pmr-c-parity-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCPP_PROFILE=pmr
      -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCppCParity.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-cpp-pmr-c-parity
  PROPERTIES
    LABELS "integration;uavcan;cpp;differential;slow"
)

if(CARGO_EXECUTABLE AND RUSTC_EXECUTABLE)
  add_test(
    NAME llvmdsdl-uavcan-c-rust-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-rust-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-c-rust-parity
    PROPERTIES
      LABELS "integration;uavcan;rust;differential;slow"
  )
else()
  message(STATUS "Skipping C/Rust parity test: cargo/rustc executable not found")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
set(LLVMDSDL_NUNAVUT_REPO "${LLVMDSDL_SOURCE_DIR}/../nunavut")
set(LLVMDSDL_PYDSDL_REPO "${LLVMDSDL_SOURCE_DIR}/../pydsdl")
if(Python3_Interpreter_FOUND AND
   EXISTS "${LLVMDSDL_NUNAVUT_REPO}/src/nunavut" AND
   EXISTS "${LLVMDSDL_PYDSDL_REPO}/pydsdl")
  add_test(
    NAME llvmdsdl-differential-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/differential-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DNUNAVUT_REPO=${LLVMDSDL_NUNAVUT_REPO}
        -DPYDSDL_REPO=${LLVMDSDL_PYDSDL_REPO}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunDifferentialParity.cmake
  )

  set_tests_properties(
    llvmdsdl-differential-parity
    PROPERTIES
      LABELS "integration;uavcan;differential;slow"
  )
else()
  message(STATUS
    "Skipping differential parity test: requires Python3 interpreter and sibling nunavut/pydsdl repos.")
endif()
