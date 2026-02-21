#===----------------------------------------------------------------------===#
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===#

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  message(STATUS
    "Skipping benchmark target setup: Python3 interpreter not available.")
  return()
endif()

set(LLVMDSDL_COMPLEX_BENCHMARK_ROOT
    "${LLVMDSDL_SOURCE_DIR}/test/benchmark/complex")
set(LLVMDSDL_COMPLEX_BENCHMARK_PRDT_LOOKUP_ROOT
    "${LLVMDSDL_SOURCE_DIR}/submodules/public_regulated_data_types")
set(LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_codegen.py")
set(LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS
    "${CMAKE_CURRENT_SOURCE_DIR}/complex_codegen_thresholds.json")

set(LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/complex-codegen")
set(LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR
    "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/generated")
set(LLVMDSDL_COMPLEX_BENCHMARK_REPORT_RECORD
    "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/record-report.json")
set(LLVMDSDL_COMPLEX_BENCHMARK_REPORT_CHECK_CI
    "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/check-ci-oom-report.json")
set(LLVMDSDL_COMPLEX_BENCHMARK_REPORT_CHECK_DEV
    "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/check-dev-ab-report.json")
set(LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS_GENERATED
    "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/complex_codegen_thresholds.generated.json")

set(LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS)
if(EXISTS "${LLVMDSDL_COMPLEX_BENCHMARK_PRDT_LOOKUP_ROOT}")
  list(APPEND LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS
       --lookup-dir "${LLVMDSDL_COMPLEX_BENCHMARK_PRDT_LOOKUP_ROOT}")
else()
  message(WARNING
    "Benchmark PRDT lookup root missing at "
    "${LLVMDSDL_COMPLEX_BENCHMARK_PRDT_LOOKUP_ROOT}; "
    "uavcan namespace resolution may fail if not present under benchmark root.")
endif()

set(LLVMDSDL_CODEGEN_BENCHMARK_LANGUAGES
    "all"
    CACHE STRING
    "Comma-separated benchmark language keys (default: all).")
set(LLVMDSDL_CODEGEN_BENCHMARK_ITERATIONS
    1
    CACHE STRING
    "Iterations per language for benchmark runs.")
set(LLVMDSDL_CODEGEN_BENCHMARK_PER_LANGUAGE_TIMEOUT_SEC
    0
    CACHE STRING
    "Per-language benchmark timeout in seconds (0 disables timeout).")
set(LLVMDSDL_CODEGEN_BENCHMARK_STATUS_INTERVAL_SEC
    15
    CACHE STRING
    "Heartbeat status print interval while each language benchmark runs (0 disables).")

option(LLVMDSDL_ENABLE_BENCHMARK_TESTS
       "Enable benchmark throughput CTest checks (timing thresholds)"
       OFF)

if(EXISTS "${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}")
  add_custom_target(benchmark-codegen-record
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}"
    COMMAND
      ${Python3_EXECUTABLE}
        ${LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT}
        record
        --dsdlc $<TARGET_FILE:dsdlc>
        --root-namespace-dir ${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}
        ${LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS}
        --out-base-dir ${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}
        --report-json ${LLVMDSDL_COMPLEX_BENCHMARK_REPORT_RECORD}
        --iterations ${LLVMDSDL_CODEGEN_BENCHMARK_ITERATIONS}
        --languages ${LLVMDSDL_CODEGEN_BENCHMARK_LANGUAGES}
        --per-language-timeout-sec ${LLVMDSDL_CODEGEN_BENCHMARK_PER_LANGUAGE_TIMEOUT_SEC}
        --status-interval-sec ${LLVMDSDL_CODEGEN_BENCHMARK_STATUS_INTERVAL_SEC}
    DEPENDS dsdlc
    USES_TERMINAL
    VERBATIM
    COMMENT "Recording complex multi-language codegen benchmark timings"
  )

  add_custom_target(benchmark-codegen-init-thresholds
    COMMAND
      ${Python3_EXECUTABLE}
        ${LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT}
        init-thresholds
        --report-json ${LLVMDSDL_COMPLEX_BENCHMARK_REPORT_RECORD}
        --thresholds-json ${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS_GENERATED}
        --dev-relative-multiplier 1.35
        --dev-absolute-margin-sec 2.0
        --ci-relative-multiplier 10.0
        --ci-absolute-margin-sec 30.0
    DEPENDS benchmark-codegen-record
    USES_TERMINAL
    VERBATIM
    COMMENT "Generating benchmark threshold template from recorded timings"
  )

  if(EXISTS "${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS}")
    add_custom_target(benchmark-codegen-check-ci-oom
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}"
      COMMAND
        ${Python3_EXECUTABLE}
          ${LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT}
          check
          --dsdlc $<TARGET_FILE:dsdlc>
          --root-namespace-dir ${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}
          ${LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS}
          --out-base-dir ${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}
          --report-json ${LLVMDSDL_COMPLEX_BENCHMARK_REPORT_CHECK_CI}
          --thresholds-json ${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS}
          --profile ci_oom
          --iterations ${LLVMDSDL_CODEGEN_BENCHMARK_ITERATIONS}
          --languages ${LLVMDSDL_CODEGEN_BENCHMARK_LANGUAGES}
          --per-language-timeout-sec ${LLVMDSDL_CODEGEN_BENCHMARK_PER_LANGUAGE_TIMEOUT_SEC}
          --status-interval-sec ${LLVMDSDL_CODEGEN_BENCHMARK_STATUS_INTERVAL_SEC}
      DEPENDS dsdlc
      USES_TERMINAL
      VERBATIM
      COMMENT "Checking complex benchmark timings against ci_oom threshold profile"
    )

    add_custom_target(benchmark-codegen-check-dev-ab
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}"
      COMMAND
        ${Python3_EXECUTABLE}
          ${LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT}
          check
          --dsdlc $<TARGET_FILE:dsdlc>
          --root-namespace-dir ${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}
          ${LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS}
          --out-base-dir ${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}
          --report-json ${LLVMDSDL_COMPLEX_BENCHMARK_REPORT_CHECK_DEV}
          --thresholds-json ${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS}
          --profile dev_ab
          --iterations ${LLVMDSDL_CODEGEN_BENCHMARK_ITERATIONS}
          --languages ${LLVMDSDL_CODEGEN_BENCHMARK_LANGUAGES}
          --per-language-timeout-sec ${LLVMDSDL_CODEGEN_BENCHMARK_PER_LANGUAGE_TIMEOUT_SEC}
          --status-interval-sec ${LLVMDSDL_CODEGEN_BENCHMARK_STATUS_INTERVAL_SEC}
      DEPENDS dsdlc
      USES_TERMINAL
      VERBATIM
      COMMENT "Checking complex benchmark timings against dev_ab threshold profile"
    )

    if(LLVMDSDL_ENABLE_BENCHMARK_TESTS)
      add_test(
        NAME llvmdsdl-codegen-benchmark-ci-oom
        COMMAND
          ${Python3_EXECUTABLE}
            ${LLVMDSDL_COMPLEX_BENCHMARK_SCRIPT}
            check
            --dsdlc $<TARGET_FILE:dsdlc>
            --root-namespace-dir ${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}
            ${LLVMDSDL_COMPLEX_BENCHMARK_LOOKUP_ARGS}
            --out-base-dir ${LLVMDSDL_COMPLEX_BENCHMARK_GENERATED_DIR}/ctest-ci-oom
            --report-json ${LLVMDSDL_COMPLEX_BENCHMARK_OUT_DIR}/ctest-check-ci-oom-report.json
            --thresholds-json ${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS}
            --profile ci_oom
            --iterations ${LLVMDSDL_CODEGEN_BENCHMARK_ITERATIONS}
            --languages ${LLVMDSDL_CODEGEN_BENCHMARK_LANGUAGES}
            --per-language-timeout-sec ${LLVMDSDL_CODEGEN_BENCHMARK_PER_LANGUAGE_TIMEOUT_SEC}
            --status-interval-sec ${LLVMDSDL_CODEGEN_BENCHMARK_STATUS_INTERVAL_SEC}
      )
      set_tests_properties(
        llvmdsdl-codegen-benchmark-ci-oom
        PROPERTIES
          LABELS "benchmark;performance;integration;slow"
      )
    endif()
  else()
    message(WARNING
      "Benchmark thresholds file is missing at "
      "${LLVMDSDL_COMPLEX_BENCHMARK_THRESHOLDS}. "
      "benchmark-codegen-check-* targets will not be created.")
  endif()
else()
  message(STATUS
    "Skipping complex benchmark target setup: root namespace not found at "
    "${LLVMDSDL_COMPLEX_BENCHMARK_ROOT}")
endif()
