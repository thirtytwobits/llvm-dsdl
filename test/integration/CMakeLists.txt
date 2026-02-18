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
    NAME llvmdsdl-signed-narrow-cpp-c-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-cpp-c-parity-optimized-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCPP_PROFILE=std
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCppCParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-cpp-c-parity-optimized
    PROPERTIES
      LABELS "integration;fixtures;cpp;differential;optimized"
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
find_program(GO_EXECUTABLE go QUIET)
find_program(TSC_EXECUTABLE tsc QUIET)
find_program(NODE_EXECUTABLE node QUIET)

if(GO_EXECUTABLE)
  add_test(
    NAME llvmdsdl-go-runtime-unit-tests
    COMMAND
      ${CMAKE_COMMAND}
        -DGO_EXECUTABLE=${GO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/go-runtime-unit-tests-out
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunGoRuntimeUnitTests.cmake
  )

  set_tests_properties(
    llvmdsdl-go-runtime-unit-tests
    PROPERTIES
      LABELS "integration;go;runtime;go-differential"
  )
else()
  message(STATUS "Skipping Go runtime unit tests: go unavailable")
endif()

if(CMAKE_C_COMPILER AND TSC_EXECUTABLE AND NODE_EXECUTABLE)
  add_test(
    NAME llvmdsdl-fixtures-c-ts-runtime-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-runtime-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsRuntimeParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-runtime-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )
else()
  message(STATUS "Skipping fixtures C/TypeScript runtime parity test: C compiler, tsc, or node unavailable")
endif()

if(TSC_EXECUTABLE AND NODE_EXECUTABLE)
  add_test(
    NAME llvmdsdl-ts-runtime-fixed-array-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-fixed-array-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeFixedArraySmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-fixed-array-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )
else()
  message(STATUS "Skipping TypeScript fixed-array runtime smoke test: tsc or node unavailable")
endif()

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

  add_test(
    NAME llvmdsdl-signed-narrow-c-rust-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-rust-parity-optimized-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-rust-parity-optimized
    PROPERTIES
      LABELS "integration;fixtures;rust;differential;optimized"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-c-rust-parity-no-std-alloc
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DRUST_PROFILE=no-std-alloc
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-rust-parity-no-std-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-rust-parity-no-std-alloc
    PROPERTIES
      LABELS "integration;fixtures;rust;differential;rust-no-std"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-c-rust-parity-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DRUST_RUNTIME_SPECIALIZATION=fast
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-rust-parity-runtime-fast-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-rust-parity-runtime-fast
    PROPERTIES
      LABELS "integration;fixtures;rust;differential;rust-runtime-specialization"
  )
else()
  message(STATUS "Skipping signed_narrow C/Rust parity test: C compiler or cargo/rustc unavailable")
endif()

if(CMAKE_C_COMPILER AND GO_EXECUTABLE AND CMAKE_AR)
  add_test(
    NAME llvmdsdl-signed-narrow-c-go-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-go-parity-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DGO_EXECUTABLE=${GO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCGoParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-go-parity
    PROPERTIES
      LABELS "integration;fixtures;go;differential;go-differential"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-c-go-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-go-parity-optimized-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DGO_EXECUTABLE=${GO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCGoParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-go-parity-optimized
    PROPERTIES
      LABELS "integration;fixtures;go;differential;go-differential;optimized"
  )
else()
  message(STATUS "Skipping signed_narrow C/Go parity test: C compiler, ar, or go unavailable")
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

add_test(
  NAME llvmdsdl-uavcan-rust-generation-no-std-alloc
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-no-std-out
      -DRUST_PROFILE=no-std-alloc
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-generation-no-std-alloc
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-no-std"
)

add_test(
  NAME llvmdsdl-uavcan-rust-generation-runtime-fast
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-runtime-fast-out
      -DRUST_RUNTIME_SPECIALIZATION=fast
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-generation-runtime-fast
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-runtime-specialization"
)

add_test(
  NAME llvmdsdl-uavcan-rust-generation-no-std-runtime-fast
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-no-std-fast-out
      -DRUST_PROFILE=no-std-alloc
      -DRUST_RUNTIME_SPECIALIZATION=fast
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-generation-no-std-runtime-fast
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-no-std;rust-runtime-specialization"
)

add_test(
  NAME llvmdsdl-uavcan-rust-profile-semantic-diff
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-profile-diff-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunRustProfileSemanticDiff.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-profile-semantic-diff
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-no-std"
)

add_test(
  NAME llvmdsdl-uavcan-rust-runtime-specialization-diff
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-runtime-diff-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunRustRuntimeSpecializationDiff.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-runtime-specialization-diff
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-runtime-specialization"
)

add_test(
  NAME llvmdsdl-uavcan-rust-runtime-specialization-diff-no-std-alloc
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-runtime-diff-no-std-out
      -DRUST_PROFILE=no-std-alloc
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunRustRuntimeSpecializationDiff.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-rust-runtime-specialization-diff-no-std-alloc
  PROPERTIES
    LABELS "integration;uavcan;rust;slow;rust-no-std;rust-runtime-specialization"
)

add_test(
  NAME llvmdsdl-uavcan-ts-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-ts-generation
  PROPERTIES
    LABELS "integration;uavcan;ts;slow"
)

add_test(
  NAME llvmdsdl-uavcan-ts-determinism
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-determinism-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsDeterminism.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-ts-determinism
  PROPERTIES
    LABELS "integration;uavcan;ts;slow"
)

add_test(
  NAME llvmdsdl-uavcan-ts-index-contract
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-index-contract-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsIndexContract.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-ts-index-contract
  PROPERTIES
    LABELS "integration;uavcan;ts;slow"
)

if(TSC_EXECUTABLE)
  add_test(
    NAME llvmdsdl-uavcan-ts-typecheck
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-typecheck-out
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsTypecheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-ts-typecheck
    PROPERTIES
      LABELS "integration;uavcan;ts;slow"
  )

  add_test(
    NAME llvmdsdl-uavcan-ts-consumer-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-consumer-smoke-out
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsConsumerSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-ts-consumer-smoke
    PROPERTIES
      LABELS "integration;uavcan;ts;slow"
  )
endif()

add_test(
  NAME llvmdsdl-uavcan-go-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-go-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanGoGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-go-generation
  PROPERTIES
    LABELS "integration;uavcan;go;go-differential"
)

add_test(
  NAME llvmdsdl-uavcan-go-determinism
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-go-determinism-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanGoDeterminism.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-go-determinism
  PROPERTIES
    LABELS "integration;uavcan;go;go-differential;slow"
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

  add_test(
    NAME llvmdsdl-uavcan-rust-cargo-check-no-std-alloc
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-cargo-check-no-std-out
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DRUST_PROFILE=no-std-alloc
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustCargoCheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-rust-cargo-check-no-std-alloc
    PROPERTIES
      LABELS "integration;uavcan;rust;slow;rust-no-std"
  )

  add_test(
    NAME llvmdsdl-uavcan-rust-cargo-check-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-cargo-check-fast-out
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DRUST_RUNTIME_SPECIALIZATION=fast
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustCargoCheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-rust-cargo-check-runtime-fast
    PROPERTIES
      LABELS "integration;uavcan;rust;slow;rust-runtime-specialization"
  )

  add_test(
    NAME llvmdsdl-uavcan-rust-cargo-check-no-std-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-rust-cargo-check-no-std-fast-out
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DRUST_PROFILE=no-std-alloc
        -DRUST_RUNTIME_SPECIALIZATION=fast
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanRustCargoCheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-rust-cargo-check-no-std-runtime-fast
    PROPERTIES
      LABELS "integration;uavcan;rust;slow;rust-no-std;rust-runtime-specialization"
  )
endif()

if(GO_EXECUTABLE)
  add_test(
    NAME llvmdsdl-uavcan-go-build
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-go-build-out
        -DGO_EXECUTABLE=${GO_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanGoBuild.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-go-build
    PROPERTIES
      LABELS "integration;uavcan;go;go-differential;slow"
  )

  if(CMAKE_AR)
    add_test(
      NAME llvmdsdl-uavcan-c-go-parity
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-go-parity-out
          -DC_COMPILER=${CMAKE_C_COMPILER}
          -DAR_EXECUTABLE=${CMAKE_AR}
          -DGO_EXECUTABLE=${GO_EXECUTABLE}
          -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCGoParity.cmake
    )

    set_tests_properties(
      llvmdsdl-uavcan-c-go-parity
      PROPERTIES
        LABELS "integration;uavcan;go;differential;go-differential;slow"
    )

    add_test(
      NAME llvmdsdl-uavcan-c-go-parity-optimized
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
          -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-go-parity-optimized-out
          -DC_COMPILER=${CMAKE_C_COMPILER}
          -DAR_EXECUTABLE=${CMAKE_AR}
          -DGO_EXECUTABLE=${GO_EXECUTABLE}
          -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCGoParity.cmake
    )

    set_tests_properties(
      llvmdsdl-uavcan-c-go-parity-optimized
      PROPERTIES
        LABELS "integration;uavcan;go;differential;go-differential;slow;optimized"
    )
  endif()
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
  NAME llvmdsdl-uavcan-cpp-c-parity-optimized
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-cpp-c-parity-optimized-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCPP_PROFILE=std
      -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCppCParity.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-cpp-c-parity-optimized
  PROPERTIES
    LABELS "integration;uavcan;cpp;differential;slow;optimized"
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

add_test(
  NAME llvmdsdl-uavcan-cpp-pmr-c-parity-optimized
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-cpp-pmr-c-parity-optimized-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCPP_PROFILE=pmr
      -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCppCParity.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-cpp-pmr-c-parity-optimized
  PROPERTIES
    LABELS "integration;uavcan;cpp;differential;slow;optimized"
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

  add_test(
    NAME llvmdsdl-uavcan-c-rust-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-rust-parity-optimized-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-c-rust-parity-optimized
    PROPERTIES
      LABELS "integration;uavcan;rust;differential;slow;optimized"
  )

  add_test(
    NAME llvmdsdl-uavcan-c-rust-parity-no-std-alloc
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DRUST_PROFILE=no-std-alloc
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-rust-parity-no-std-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-c-rust-parity-no-std-alloc
    PROPERTIES
      LABELS "integration;uavcan;rust;differential;slow;rust-no-std"
  )

  add_test(
    NAME llvmdsdl-uavcan-c-rust-parity-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DRUST_RUNTIME_SPECIALIZATION=fast
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-rust-parity-runtime-fast-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-c-rust-parity-runtime-fast
    PROPERTIES
      LABELS "integration;uavcan;rust;differential;slow;rust-runtime-specialization"
  )

  add_test(
    NAME llvmdsdl-uavcan-c-rust-parity-no-std-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DRUST_PROFILE=no-std-alloc
        -DRUST_RUNTIME_SPECIALIZATION=fast
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-c-rust-parity-no-std-fast-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DAR_EXECUTABLE=${CMAKE_AR}
        -DCARGO_EXECUTABLE=${CARGO_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCRustParity.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-c-rust-parity-no-std-runtime-fast
    PROPERTIES
      LABELS "integration;uavcan;rust;differential;slow;rust-no-std;rust-runtime-specialization"
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

  add_test(
    NAME llvmdsdl-differential-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/differential-parity-optimized-out
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DNUNAVUT_REPO=${LLVMDSDL_NUNAVUT_REPO}
        -DPYDSDL_REPO=${LLVMDSDL_PYDSDL_REPO}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunDifferentialParity.cmake
  )

  set_tests_properties(
    llvmdsdl-differential-parity-optimized
    PROPERTIES
      LABELS "integration;uavcan;differential;slow;optimized"
  )
else()
  message(STATUS
    "Skipping differential parity test: requires Python3 interpreter and sibling nunavut/pydsdl repos.")
endif()
