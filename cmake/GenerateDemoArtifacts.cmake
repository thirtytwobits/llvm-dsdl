cmake_minimum_required(VERSION 3.24)

foreach(var
    DSDLC
    DSDLOPT
    UAVCAN_ROOT
    OUT_DIR
    BINARY_DIR
    CTEST_COMMAND
    C_COMPILER
    COMPILE_C_SCRIPT
    SOURCE_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()
if(NOT EXISTS "${DSDLOPT}")
  message(FATAL_ERROR "dsdl-opt executable not found: ${DSDLOPT}")
endif()
if(NOT EXISTS "${UAVCAN_ROOT}")
  message(FATAL_ERROR "uavcan root not found: ${UAVCAN_ROOT}")
endif()
if(NOT EXISTS "${CTEST_COMMAND}")
  message(FATAL_ERROR "ctest executable not found: ${CTEST_COMMAND}")
endif()
if(NOT EXISTS "${C_COMPILER}")
  message(FATAL_ERROR "C compiler not found: ${C_COMPILER}")
endif()
if(NOT EXISTS "${COMPILE_C_SCRIPT}")
  message(FATAL_ERROR "compile helper script not found: ${COMPILE_C_SCRIPT}")
endif()
if(NOT EXISTS "${SOURCE_DIR}")
  message(FATAL_ERROR "source root not found: ${SOURCE_DIR}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}/logs")

function(run_step label)
  set(cmd ${ARGN})
  string(REPLACE ";" " " cmd_string "${cmd}")
  message(STATUS "[demo] ${label}: ${cmd_string}")
  execute_process(
    COMMAND ${cmd}
    RESULT_VARIABLE rv
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
  )
  file(WRITE "${OUT_DIR}/logs/${label}.stdout.log" "${out}")
  file(WRITE "${OUT_DIR}/logs/${label}.stderr.log" "${err}")
  if(NOT rv EQUAL 0)
    message(STATUS "stdout (${label}):\n${out}")
    message(STATUS "stderr (${label}):\n${err}")
    message(FATAL_ERROR "Demo step failed: ${label}")
  endif()
endfunction()

set(C_OUT "${OUT_DIR}/c")
set(CPP_OUT "${OUT_DIR}/cpp")
set(RUST_OUT "${OUT_DIR}/rust")

run_step(
  "dsdlc-c"
  "${DSDLC}" c
  --root-namespace-dir "${UAVCAN_ROOT}"
  --out-dir "${C_OUT}"
)

run_step(
  "compile-generated-c"
  "${CMAKE_COMMAND}"
  -DC_COMPILER=${C_COMPILER}
  -DSOURCE_ROOT=${C_OUT}
  -DINCLUDE_ROOT=${C_OUT}
  -P "${COMPILE_C_SCRIPT}"
)

run_step(
  "dsdlc-cpp-both"
  "${DSDLC}" cpp
  --root-namespace-dir "${UAVCAN_ROOT}"
  --cpp-profile both
  --out-dir "${CPP_OUT}"
)

run_step(
  "dsdlc-rust-std"
  "${DSDLC}" rust
  --root-namespace-dir "${UAVCAN_ROOT}"
  --rust-profile std
  --rust-crate-name uavcan_dsdl_generated
  --out-dir "${RUST_OUT}"
)

set(MLIR_OUT "${OUT_DIR}/uavcan.mlir")
execute_process(
  COMMAND
    "${DSDLC}" mlir
      --root-namespace-dir "${UAVCAN_ROOT}"
  RESULT_VARIABLE mlir_rv
  OUTPUT_FILE "${MLIR_OUT}"
  ERROR_VARIABLE mlir_err
)
file(WRITE "${OUT_DIR}/logs/dsdlc-mlir.stderr.log" "${mlir_err}")
if(NOT mlir_rv EQUAL 0)
  message(STATUS "stderr (dsdlc-mlir):\n${mlir_err}")
  message(FATAL_ERROR "Failed to generate MLIR artifact")
endif()

set(LOWERED_MLIR_OUT "${OUT_DIR}/uavcan.lowered.mlir")
execute_process(
  COMMAND
    "${DSDLOPT}"
      --pass-pipeline
      "builtin.module(lower-dsdl-serialization)"
      "${MLIR_OUT}"
  RESULT_VARIABLE lowered_mlir_rv
  OUTPUT_FILE "${LOWERED_MLIR_OUT}"
  ERROR_VARIABLE lowered_mlir_err
)
file(WRITE "${OUT_DIR}/logs/dsdl-opt-lower.stderr.log" "${lowered_mlir_err}")
if(NOT lowered_mlir_rv EQUAL 0)
  message(STATUS "stderr (dsdl-opt-lower):\n${lowered_mlir_err}")
  message(FATAL_ERROR "Failed to lower full uavcan MLIR artifact")
endif()

file(GLOB_RECURSE dsdl_files "${UAVCAN_ROOT}/*.dsdl")
list(LENGTH dsdl_files dsdl_count)

file(GLOB_RECURSE c_headers "${C_OUT}/*.h")
set(c_type_header_count 0)
foreach(h IN LISTS c_headers)
  if(NOT h MATCHES "dsdl_runtime\\.h$")
    math(EXPR c_type_header_count "${c_type_header_count} + 1")
  endif()
endforeach()

file(GLOB_RECURSE c_impl_sources "${C_OUT}/*.c")
list(LENGTH c_impl_sources c_impl_count)

file(GLOB_RECURSE cpp_std_headers "${CPP_OUT}/std/*.hpp")
set(cpp_std_type_header_count 0)
foreach(h IN LISTS cpp_std_headers)
  if(NOT h MATCHES "dsdl_runtime\\.hpp$")
    math(EXPR cpp_std_type_header_count "${cpp_std_type_header_count} + 1")
  endif()
endforeach()

file(GLOB_RECURSE cpp_pmr_headers "${CPP_OUT}/pmr/*.hpp")
set(cpp_pmr_type_header_count 0)
foreach(h IN LISTS cpp_pmr_headers)
  if(NOT h MATCHES "dsdl_runtime\\.hpp$")
    math(EXPR cpp_pmr_type_header_count "${cpp_pmr_type_header_count} + 1")
  endif()
endforeach()

if(NOT c_type_header_count EQUAL dsdl_count)
  message(FATAL_ERROR
    "C type header count mismatch: dsdl=${dsdl_count} generated=${c_type_header_count}")
endif()

if(NOT cpp_std_type_header_count EQUAL dsdl_count)
  message(FATAL_ERROR
    "C++ std header count mismatch: dsdl=${dsdl_count} generated=${cpp_std_type_header_count}")
endif()

if(NOT cpp_pmr_type_header_count EQUAL dsdl_count)
  message(FATAL_ERROR
    "C++ pmr header count mismatch: dsdl=${dsdl_count} generated=${cpp_pmr_type_header_count}")
endif()

execute_process(
  COMMAND "${CTEST_COMMAND}" --test-dir "${BINARY_DIR}" -N
  RESULT_VARIABLE ctest_list_rv
  OUTPUT_VARIABLE ctest_list_out
  ERROR_VARIABLE ctest_list_err
)
if(NOT ctest_list_rv EQUAL 0)
  message(STATUS "ctest -N stdout:\n${ctest_list_out}")
  message(STATUS "ctest -N stderr:\n${ctest_list_err}")
  message(FATAL_ERROR "Failed to list tests from build directory")
endif()

set(demo_tests
  llvmdsdl-uavcan-generation
  llvmdsdl-uavcan-mlir-lowering
  llvmdsdl-uavcan-cpp-generation
  llvmdsdl-uavcan-cpp-c-parity
  llvmdsdl-uavcan-cpp-pmr-c-parity
  llvmdsdl-uavcan-rust-generation
  llvmdsdl-uavcan-rust-cargo-check
  llvmdsdl-uavcan-c-rust-parity
  llvmdsdl-differential-parity
)
set(test_summary "")
foreach(test_name IN LISTS demo_tests)
  string(FIND "${ctest_list_out}" "${test_name}" test_index)
  if(test_index EQUAL -1)
    string(APPEND test_summary "- `${test_name}`: skipped (not registered in this build)\n")
    continue()
  endif()

  run_step(
    "ctest-${test_name}"
    "${CTEST_COMMAND}"
      --test-dir "${BINARY_DIR}"
      -R "^${test_name}$"
      --output-on-failure
  )
  string(APPEND test_summary "- `${test_name}`: PASS\n")
endforeach()

set(demo_md "")
string(APPEND demo_md "# llvm-dsdl Demo Artifact (2026-02-16)\n\n")
string(APPEND demo_md "This folder was generated automatically by:\n\n")
string(APPEND demo_md "```bash\n")
string(APPEND demo_md "cmake --build <build-dir> --target generate-demo-2026-02-16\n")
string(APPEND demo_md "```\n\n")
string(APPEND demo_md "## Inputs\n\n")
string(APPEND demo_md "- Repository: `${SOURCE_DIR}`\n")
string(APPEND demo_md "- Build directory: `${BINARY_DIR}`\n")
string(APPEND demo_md "- DSDL root: `${UAVCAN_ROOT}`\n\n")
string(APPEND demo_md "## Commands Executed\n\n")
string(APPEND demo_md "```bash\n")
string(APPEND demo_md "${DSDLC} c --root-namespace-dir ${UAVCAN_ROOT} --out-dir ${C_OUT}\n")
string(APPEND demo_md "${CMAKE_COMMAND} -DC_COMPILER=${C_COMPILER} -DSOURCE_ROOT=${C_OUT} -DINCLUDE_ROOT=${C_OUT} -P ${COMPILE_C_SCRIPT}\n")
string(APPEND demo_md "${DSDLC} cpp --root-namespace-dir ${UAVCAN_ROOT} --cpp-profile both --out-dir ${CPP_OUT}\n")
string(APPEND demo_md "${DSDLC} rust --root-namespace-dir ${UAVCAN_ROOT} --rust-profile std --rust-crate-name uavcan_dsdl_generated --out-dir ${RUST_OUT}\n")
string(APPEND demo_md "${DSDLC} mlir --root-namespace-dir ${UAVCAN_ROOT} > ${MLIR_OUT}\n")
string(APPEND demo_md "${DSDLOPT} --pass-pipeline 'builtin.module(lower-dsdl-serialization)' ${MLIR_OUT} > ${LOWERED_MLIR_OUT}\n")
string(APPEND demo_md "```\n\n")
string(APPEND demo_md "## Artifact Summary\n\n")
string(APPEND demo_md "- DSDL definitions discovered: `${dsdl_count}`\n")
string(APPEND demo_md "- C headers (excluding runtime): `${c_type_header_count}`\n")
string(APPEND demo_md "- C implementation translation units: `${c_impl_count}`\n")
string(APPEND demo_md "- C++ std headers (excluding runtime): `${cpp_std_type_header_count}`\n")
string(APPEND demo_md "- C++ pmr headers (excluding runtime): `${cpp_pmr_type_header_count}`\n")
string(APPEND demo_md "- MLIR snapshot: `${MLIR_OUT}`\n\n")
string(APPEND demo_md "- Lowered MLIR snapshot: `${LOWERED_MLIR_OUT}`\n\n")
string(APPEND demo_md "## Test Results\n\n")
string(APPEND demo_md "${test_summary}\n")
string(APPEND demo_md "Detailed command logs are under `${OUT_DIR}/logs/`.\n")

file(WRITE "${OUT_DIR}/DEMO.md" "${demo_md}")
message(STATUS "Demo artifacts generated at: ${OUT_DIR}")
