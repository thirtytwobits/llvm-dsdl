set(LLVMDSDL_UAVCAN_ROOT
    "${LLVMDSDL_SOURCE_DIR}/submodules/public_regulated_data_types/uavcan")

set(LLVMDSDL_FIXTURES_ROOT
    "${LLVMDSDL_SOURCE_DIR}/test/lit/fixtures")
find_program(PYTHON_EXECUTABLE python3 QUIET)
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

  add_test(
    NAME llvmdsdl-fixtures-ts-generation-hardening
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-ts-generation-hardening-out
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesTsGenerationHardening.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-ts-generation-hardening
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  if(PYTHON_EXECUTABLE)
    add_test(
      NAME llvmdsdl-fixtures-python-generation-hardening
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-python-generation-hardening-out
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesPythonGenerationHardening.cmake
    )

    set_tests_properties(
      llvmdsdl-fixtures-python-generation-hardening
      PROPERTIES
        LABELS "integration;fixtures;python"
    )

    add_test(
      NAME llvmdsdl-fixtures-python-runtime-smoke
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-python-runtime-smoke-out
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunPythonRuntimeSmoke.cmake
    )

    set_tests_properties(
      llvmdsdl-fixtures-python-runtime-smoke
      PROPERTIES
        LABELS "integration;fixtures;python;runtime"
    )

    add_test(
      NAME llvmdsdl-fixtures-python-runtime-smoke-optimized
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
          -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-python-runtime-smoke-optimized-out
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunPythonRuntimeSmoke.cmake
    )

    set_tests_properties(
      llvmdsdl-fixtures-python-runtime-smoke-optimized
      PROPERTIES
        LABELS "integration;fixtures;python;runtime;optimized"
    )

    add_test(
      NAME llvmdsdl-fixtures-python-runtime-backend-selection
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-python-runtime-backend-selection-out
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
          -DACCEL_MODULE=$<$<TARGET_EXISTS:llvmdsdl-python-runtime-accelerator>:$<TARGET_FILE:llvmdsdl-python-runtime-accelerator>>
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunPythonRuntimeBackendSelection.cmake
    )

    set_tests_properties(
      llvmdsdl-fixtures-python-runtime-backend-selection
      PROPERTIES
        LABELS "integration;fixtures;python;runtime"
    )

    add_test(
      NAME llvmdsdl-fixtures-python-runtime-parity
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-python-runtime-parity-out
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
          -DACCEL_MODULE=$<$<TARGET_EXISTS:llvmdsdl-python-runtime-accelerator>:$<TARGET_FILE:llvmdsdl-python-runtime-accelerator>>
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunPythonRuntimeParity.cmake
    )

    set_tests_properties(
      llvmdsdl-fixtures-python-runtime-parity
      PROPERTIES
        LABELS "integration;fixtures;python;runtime;differential"
    )
  else()
    message(STATUS "Skipping Python fixture integration tests: python3 unavailable")
  endif()
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
    NAME llvmdsdl-c-ts-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/c-ts-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCTsParity.cmake
  )

  set_tests_properties(
    llvmdsdl-c-ts-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-c-ts-parity-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DTS_RUNTIME_SPECIALIZATION=fast
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/c-ts-parity-runtime-fast-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCTsParity.cmake
  )

  set_tests_properties(
    llvmdsdl-c-ts-parity-runtime-fast
    PROPERTIES
      LABELS "integration;fixtures;ts;differential;ts-runtime-specialization"
  )

  add_test(
    NAME llvmdsdl-c-ts-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/c-ts-parity-optimized-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunCTsParity.cmake
  )

  set_tests_properties(
    llvmdsdl-c-ts-parity-optimized
    PROPERTIES
      LABELS "integration;fixtures;ts;differential;optimized"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-c-ts-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-ts-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCTsParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-ts-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-signed-narrow-c-ts-parity-optimized
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLC_EXTRA_ARGS=--optimize-lowered-serdes
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/signed-narrow-c-ts-parity-optimized-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunSignedNarrowCTsParity.cmake
  )

  set_tests_properties(
    llvmdsdl-signed-narrow-c-ts-parity-optimized
    PROPERTIES
      LABELS "integration;fixtures;ts;differential;optimized"
  )

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

  add_test(
    NAME llvmdsdl-fixtures-c-ts-service-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-service-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsServiceParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-service-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-variable-array-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-variable-array-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsVariableArrayParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-variable-array-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-bigint-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-bigint-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsBigIntParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-bigint-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-float-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-float-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsFloatParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-float-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-utf8-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-utf8-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsUtf8Parity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-utf8-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-union-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-union-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsUnionParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-union-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-union-composite-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-union-composite-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsUnionCompositeParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-union-composite-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-union-empty-composite-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-union-empty-composite-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsUnionEmptyCompositeParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-union-empty-composite-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-union-array-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-union-array-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsUnionArrayParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-union-array-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-truncated-decode-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-truncated-decode-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsTruncatedDecodeParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-truncated-decode-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-padding-alignment-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-padding-alignment-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsPaddingAlignmentParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-padding-alignment-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-composite-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-composite-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsCompositeParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-composite-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-delimited-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-delimited-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsDelimitedParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-delimited-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-delimited-invalid-header-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-delimited-invalid-header-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsDelimitedInvalidHeaderParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-delimited-invalid-header-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-composite-fixed-array-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-composite-fixed-array-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsCompositeFixedArrayParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-composite-fixed-array-parity
    PROPERTIES
      LABELS "integration;fixtures;ts;differential"
  )

  add_test(
    NAME llvmdsdl-fixtures-c-ts-composite-variable-array-parity
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures-c-ts-composite-variable-array-parity-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFixturesCTsCompositeVariableArrayParity.cmake
  )

  set_tests_properties(
    llvmdsdl-fixtures-c-ts-composite-variable-array-parity
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

  add_test(
    NAME llvmdsdl-ts-runtime-variable-array-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-variable-array-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeVariableArraySmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-variable-array-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-bigint-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-bigint-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeBigIntSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-bigint-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-float-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-float-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeFloatSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-float-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-utf8-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-utf8-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeUtf8Smoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-utf8-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-service-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-service-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeServiceSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-service-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-union-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-union-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeUnionSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-union-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-union-composite-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-union-composite-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeUnionCompositeSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-union-composite-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-union-empty-composite-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-union-empty-composite-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeUnionEmptyCompositeSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-union-empty-composite-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-union-array-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-union-array-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeUnionArraySmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-union-array-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-truncated-decode-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-truncated-decode-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeTruncatedDecodeSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-truncated-decode-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-padding-alignment-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-padding-alignment-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimePaddingAlignmentSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-padding-alignment-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-composite-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-composite-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeCompositeSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-composite-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-delimited-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-delimited-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeDelimitedSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-delimited-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-delimited-invalid-header-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-delimited-invalid-header-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeDelimitedInvalidHeaderSmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-delimited-invalid-header-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-composite-fixed-array-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-composite-fixed-array-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeCompositeFixedArraySmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-composite-fixed-array-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )

  add_test(
    NAME llvmdsdl-ts-runtime-composite-variable-array-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ts-runtime-composite-variable-array-smoke-out
        -DSOURCE_ROOT=${LLVMDSDL_SOURCE_DIR}
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeCompositeVariableArraySmoke.cmake
  )

  set_tests_properties(
    llvmdsdl-ts-runtime-composite-variable-array-smoke
    PROPERTIES
      LABELS "integration;fixtures;ts"
  )
else()
  message(STATUS "Skipping TypeScript runtime smoke tests: tsc or node unavailable")
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
  NAME llvmdsdl-uavcan-generation
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-out
      -DC_COMPILER=${CMAKE_C_COMPILER}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-generation
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
  NAME llvmdsdl-uavcan-ts-generation-runtime-fast
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-runtime-fast-out
      -DTS_RUNTIME_SPECIALIZATION=fast
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsGeneration.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-ts-generation-runtime-fast
  PROPERTIES
    LABELS "integration;uavcan;ts;slow;ts-runtime-specialization"
)

add_test(
  NAME llvmdsdl-uavcan-ts-runtime-specialization-diff
  COMMAND
    ${CMAKE_COMMAND}
      -DDSDLC=$<TARGET_FILE:dsdlc>
      -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-runtime-specialization-diff-out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/RunTsRuntimeSpecializationDiff.cmake
)

set_tests_properties(
  llvmdsdl-uavcan-ts-runtime-specialization-diff
  PROPERTIES
    LABELS "integration;uavcan;ts;slow;ts-runtime-specialization"
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

if(PYTHON_EXECUTABLE)
  add_test(
    NAME llvmdsdl-uavcan-python-generation
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-python-out
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanPythonGeneration.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-python-generation
    PROPERTIES
      LABELS "integration;uavcan;python;slow"
  )

  add_test(
    NAME llvmdsdl-uavcan-python-determinism
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-python-determinism-out
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanPythonDeterminism.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-python-determinism
    PROPERTIES
      LABELS "integration;uavcan;python;slow"
  )

  add_test(
    NAME llvmdsdl-uavcan-python-runtime-bench
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DFIXTURES_ROOT=${LLVMDSDL_FIXTURES_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-python-runtime-bench-out
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -DACCEL_MODULE=$<$<TARGET_EXISTS:llvmdsdl-python-runtime-accelerator>:$<TARGET_FILE:llvmdsdl-python-runtime-accelerator>>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunPythonRuntimeBench.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-python-runtime-bench
    PROPERTIES
      LABELS "integration;uavcan;python;bench;slow"
  )
else()
  message(STATUS "Skipping uavcan Python integration tests: python3 unavailable")
endif()

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
    NAME llvmdsdl-uavcan-ts-typecheck-runtime-fast
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-typecheck-runtime-fast-out
        -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
        -DTS_RUNTIME_SPECIALIZATION=fast
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsTypecheck.cmake
  )

  set_tests_properties(
    llvmdsdl-uavcan-ts-typecheck-runtime-fast
    PROPERTIES
      LABELS "integration;uavcan;ts;slow;ts-runtime-specialization"
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

  if(NODE_EXECUTABLE)
    add_test(
      NAME llvmdsdl-uavcan-ts-runtime-execution-smoke
      COMMAND
        ${CMAKE_COMMAND}
          -DDSDLC=$<TARGET_FILE:dsdlc>
          -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
          -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/uavcan-ts-runtime-execution-smoke-out
          -DTSC_EXECUTABLE=${TSC_EXECUTABLE}
          -DNODE_EXECUTABLE=${NODE_EXECUTABLE}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/RunUavcanTsRuntimeExecution.cmake
    )

    set_tests_properties(
      llvmdsdl-uavcan-ts-runtime-execution-smoke
      PROPERTIES
        LABELS "integration;uavcan;ts;slow"
    )
  endif()
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

find_program(NPM_EXECUTABLE npm QUIET)
if(NPM_EXECUTABLE)
  add_test(
    NAME llvmdsdl-vscode-extension-smoke
    COMMAND
      ${CMAKE_COMMAND}
        -DNPM_EXECUTABLE=${NPM_EXECUTABLE}
        -DEXTENSION_DIR=${LLVMDSDL_SOURCE_DIR}/editors/vscode/dsdld-client
        -DDSDLD_BINARY=$<TARGET_FILE:dsdld>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/RunVsCodeExtensionTests.cmake
  )

  set_tests_properties(
    llvmdsdl-vscode-extension-smoke
    PROPERTIES
      LABELS "integration;vscode;lsp"
  )
else()
  message(STATUS "Skipping VSCode extension integration test: npm not found")
endif()
