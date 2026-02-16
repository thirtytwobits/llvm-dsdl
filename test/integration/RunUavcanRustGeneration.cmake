cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC UAVCAN_ROOT OUT_DIR)
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

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

execute_process(
  COMMAND
    "${DSDLC}" rust
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --out-dir "${OUT_DIR}"
      --rust-crate-name "uavcan_dsdl_generated"
      --rust-profile "std"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "uavcan rust generation failed")
endif()

foreach(required
    "${OUT_DIR}/Cargo.toml"
    "${OUT_DIR}/src/lib.rs"
    "${OUT_DIR}/src/dsdl_runtime.rs")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing required generated file: ${required}")
  endif()
endforeach()

file(GLOB_RECURSE dsdl_files "${UAVCAN_ROOT}/*.dsdl")
list(LENGTH dsdl_files dsdl_count)

file(GLOB_RECURSE rust_files "${OUT_DIR}/src/*.rs")
set(type_rs_files "")
foreach(rs IN LISTS rust_files)
  get_filename_component(name "${rs}" NAME)
  if(NOT name STREQUAL "lib.rs" AND
     NOT name STREQUAL "mod.rs" AND
     NOT name STREQUAL "dsdl_runtime.rs")
    list(APPEND type_rs_files "${rs}")
  endif()
endforeach()
list(LENGTH type_rs_files type_rs_count)

if(NOT dsdl_count EQUAL type_rs_count)
  message(FATAL_ERROR
    "Rust type file count mismatch: dsdl=${dsdl_count}, generated=${type_rs_count}")
endif()

message(STATUS
  "uavcan rust generation check passed: ${dsdl_count} DSDL -> ${type_rs_count} Rust type files")
