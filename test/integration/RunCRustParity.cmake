cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC UAVCAN_ROOT OUT_DIR C_COMPILER AR_EXECUTABLE CARGO_EXECUTABLE SOURCE_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()

if(NOT EXISTS "${UAVCAN_ROOT}")
  message(FATAL_ERROR "uavcan root not found: ${UAVCAN_ROOT}")
endif()

if(NOT EXISTS "${C_COMPILER}")
  message(FATAL_ERROR "C compiler not found: ${C_COMPILER}")
endif()

if(NOT EXISTS "${AR_EXECUTABLE}")
  message(FATAL_ERROR "archive tool not found: ${AR_EXECUTABLE}")
endif()

if(NOT EXISTS "${CARGO_EXECUTABLE}")
  message(FATAL_ERROR "cargo executable not found: ${CARGO_EXECUTABLE}")
endif()

set(cargo_toml_template "${SOURCE_ROOT}/test/integration/CRustParityCargo.toml.in")
set(main_rs_template "${SOURCE_ROOT}/test/integration/CRustParityMain.rs")
set(c_harness_template "${SOURCE_ROOT}/test/integration/CRustParityCHarness.c")
foreach(path "${cargo_toml_template}" "${main_rs_template}" "${c_harness_template}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "C/Rust parity harness input missing: ${path}")
  endif()
endforeach()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(c_out "${OUT_DIR}/c")
set(rust_out "${OUT_DIR}/rust")
set(build_out "${OUT_DIR}/build")
set(harness_out "${OUT_DIR}/harness")
file(MAKE_DIRECTORY "${c_out}")
file(MAKE_DIRECTORY "${rust_out}")
file(MAKE_DIRECTORY "${build_out}")
file(MAKE_DIRECTORY "${harness_out}")
file(MAKE_DIRECTORY "${harness_out}/src")

execute_process(
  COMMAND
    "${DSDLC}" c
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --out-dir "${c_out}"
  RESULT_VARIABLE c_result
  OUTPUT_VARIABLE c_stdout
  ERROR_VARIABLE c_stderr
)
if(NOT c_result EQUAL 0)
  message(STATUS "dsdlc c stdout:\n${c_stdout}")
  message(STATUS "dsdlc c stderr:\n${c_stderr}")
  message(FATAL_ERROR "failed to generate C output for C/Rust parity harness")
endif()

execute_process(
  COMMAND
    "${DSDLC}" rust
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --rust-profile std
      --rust-crate-name uavcan_dsdl_generated
      --out-dir "${rust_out}"
  RESULT_VARIABLE rust_result
  OUTPUT_VARIABLE rust_stdout
  ERROR_VARIABLE rust_stderr
)
if(NOT rust_result EQUAL 0)
  message(STATUS "dsdlc rust stdout:\n${rust_stdout}")
  message(STATUS "dsdlc rust stderr:\n${rust_stderr}")
  message(FATAL_ERROR "failed to generate Rust output for C/Rust parity harness")
endif()

set(RUST_OUT "${rust_out}")
configure_file("${cargo_toml_template}" "${harness_out}/Cargo.toml" @ONLY)
configure_file("${main_rs_template}" "${harness_out}/src/main.rs" COPYONLY)
configure_file("${c_harness_template}" "${harness_out}/src/c_harness.c" COPYONLY)

set(harness_obj "${build_out}/c_harness.o")
execute_process(
  COMMAND
    "${C_COMPILER}"
      -std=c11
      -Wall
      -Wextra
      -Werror
      -I "${c_out}"
      -c "${harness_out}/src/c_harness.c"
      -o "${harness_obj}"
  RESULT_VARIABLE harness_cc_result
  OUTPUT_VARIABLE harness_cc_stdout
  ERROR_VARIABLE harness_cc_stderr
)
if(NOT harness_cc_result EQUAL 0)
  message(STATUS "C harness compile stdout:\n${harness_cc_stdout}")
  message(STATUS "C harness compile stderr:\n${harness_cc_stderr}")
  message(FATAL_ERROR "failed to compile C/Rust parity C harness")
endif()

file(GLOB_RECURSE generated_c_sources "${c_out}/*.c")
list(LENGTH generated_c_sources generated_c_count)
if(generated_c_count EQUAL 0)
  message(FATAL_ERROR "no generated C implementation sources under ${c_out}")
endif()

set(generated_obj_dir "${build_out}/generated-obj")
file(MAKE_DIRECTORY "${generated_obj_dir}")
set(generated_objs "")
set(c_index 0)
foreach(src IN LISTS generated_c_sources)
  math(EXPR c_index "${c_index} + 1")
  set(obj "${generated_obj_dir}/generated_${c_index}.o")
  execute_process(
    COMMAND
      "${C_COMPILER}"
        -std=c11
        -Wall
        -Wextra
        -Werror
        -I "${c_out}"
        -c "${src}"
        -o "${obj}"
    RESULT_VARIABLE c_cc_result
    OUTPUT_VARIABLE c_cc_stdout
    ERROR_VARIABLE c_cc_stderr
  )
  if(NOT c_cc_result EQUAL 0)
    message(STATUS "failed source: ${src}")
    message(STATUS "generated C compile stdout:\n${c_cc_stdout}")
    message(STATUS "generated C compile stderr:\n${c_cc_stderr}")
    message(FATAL_ERROR "failed to compile generated C implementation for C/Rust parity")
  endif()
  list(APPEND generated_objs "${obj}")
endforeach()

set(static_lib "${build_out}/libllvmdsdl_c_rust_parity.a")
execute_process(
  COMMAND "${AR_EXECUTABLE}" rcs "${static_lib}" "${harness_obj}" ${generated_objs}
  RESULT_VARIABLE ar_result
  OUTPUT_VARIABLE ar_stdout
  ERROR_VARIABLE ar_stderr
)
if(NOT ar_result EQUAL 0)
  message(STATUS "archive stdout:\n${ar_stdout}")
  message(STATUS "archive stderr:\n${ar_stderr}")
  message(FATAL_ERROR "failed to archive C/Rust parity support library")
endif()
if(NOT EXISTS "${static_lib}")
  message(FATAL_ERROR "C/Rust parity archive missing after creation: ${static_lib}")
endif()

set(rustflags "-L native=${build_out} -l static=llvmdsdl_c_rust_parity")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "RUSTFLAGS=${rustflags}"
      "CARGO_TARGET_DIR=${build_out}/cargo-target"
      "${CARGO_EXECUTABLE}" run --quiet --manifest-path "${harness_out}/Cargo.toml" -- 128
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
)
if(NOT run_result EQUAL 0)
  message(STATUS "cargo run stdout:\n${run_stdout}")
  message(STATUS "cargo run stderr:\n${run_stderr}")
  message(FATAL_ERROR "C/Rust parity harness reported mismatches")
endif()

set(min_random 128)
set(min_cases 8)
set(min_directed 1)
string(REGEX MATCH
  "PASS c/rust parity random=([0-9]+) cases=([0-9]+) directed=([0-9]+)"
  parity_summary_line
  "${run_stdout}")
if(NOT parity_summary_line)
  message(FATAL_ERROR
    "failed to parse C/Rust parity summary line from harness output")
endif()
set(observed_random "${CMAKE_MATCH_1}")
set(observed_cases "${CMAKE_MATCH_2}")
set(observed_directed "${CMAKE_MATCH_3}")
if(observed_random LESS min_random)
  message(FATAL_ERROR
    "C/Rust parity random-iteration regression: observed=${observed_random}, required>=${min_random}")
endif()
if(observed_cases LESS min_cases)
  message(FATAL_ERROR
    "C/Rust parity case count regression: observed=${observed_cases}, required>=${min_cases}")
endif()
if(observed_directed LESS min_directed)
  message(FATAL_ERROR
    "C/Rust parity directed count regression: observed=${observed_directed}, required>=${min_directed}")
endif()

string(REGEX MATCH
  "PASS c/rust inventory cases=([0-9]+) directed=([0-9]+)"
  inventory_summary_match
  "${run_stdout}")
if(NOT inventory_summary_match)
  message(FATAL_ERROR "missing C/Rust parity inventory summary marker")
endif()
set(inventory_cases "${CMAKE_MATCH_1}")
set(inventory_directed "${CMAKE_MATCH_2}")
if(NOT inventory_cases EQUAL observed_cases OR
   NOT inventory_directed EQUAL observed_directed)
  message(FATAL_ERROR
    "C/Rust parity inventory mismatch: inventory cases=${inventory_cases}, "
    "inventory directed=${inventory_directed}, summary cases=${observed_cases}, "
    "summary directed=${observed_directed}")
endif()

string(REGEX MATCHALL
  "PASS [A-Za-z0-9_.]+ random \\([0-9]+ iterations\\)"
  random_pass_lines
  "${run_stdout}")
list(LENGTH random_pass_lines observed_random_pass_lines)
if(NOT observed_random_pass_lines EQUAL observed_cases)
  message(FATAL_ERROR
    "C/Rust random execution count mismatch: pass-lines=${observed_random_pass_lines}, "
    "summary cases=${observed_cases}")
endif()

string(REGEX MATCHALL
  "PASS [A-Za-z0-9_]+ directed"
  directed_pass_lines
  "${run_stdout}")
list(LENGTH directed_pass_lines observed_directed_pass_lines)
if(NOT observed_directed_pass_lines EQUAL observed_directed)
  message(FATAL_ERROR
    "C/Rust directed execution count mismatch: pass-lines=${observed_directed_pass_lines}, "
    "summary directed=${observed_directed}")
endif()

set(summary_file "${OUT_DIR}/c-rust-parity-summary.txt")
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef summary_nonce)
set(summary_tmp "${summary_file}.tmp-${summary_nonce}")
file(WRITE "${summary_tmp}" "${run_stdout}\n")
file(RENAME "${summary_tmp}" "${summary_file}")
message(STATUS "C/Rust parity summary:\n${run_stdout}")
