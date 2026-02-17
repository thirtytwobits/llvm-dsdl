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

set(found_mlir_union_helper FALSE)
set(found_mlir_union_validate_helper FALSE)
set(found_mlir_union_validate_call FALSE)
set(found_mlir_capacity_check_helper FALSE)
set(found_mlir_capacity_check_call FALSE)
set(found_mlir_array_prefix_helper FALSE)
set(found_mlir_array_validate_helper FALSE)
set(found_mlir_scalar_unsigned_helper FALSE)
set(found_mlir_scalar_signed_helper FALSE)
set(found_mlir_scalar_float_helper FALSE)
set(found_mlir_delimiter_validate_helper FALSE)
foreach(rs IN LISTS type_rs_files)
  file(READ "${rs}" rs_text)
  if(rs_text MATCHES "mlir___llvmdsdl_plan_union_tag__")
    set(found_mlir_union_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_validate_union_tag__")
    set(found_mlir_union_validate_helper TRUE)
  endif()
  if(rs_text MATCHES "let _err_union_tag = mlir___llvmdsdl_plan_validate_union_tag__")
    set(found_mlir_union_validate_call TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_capacity_check__")
    set(found_mlir_capacity_check_helper TRUE)
  endif()
  if(rs_text MATCHES "let _err_capacity = mlir___llvmdsdl_plan_capacity_check__")
    set(found_mlir_capacity_check_call TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_array_length_prefix__")
    set(found_mlir_array_prefix_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_validate_array_length__")
    set(found_mlir_array_validate_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_scalar_unsigned__")
    set(found_mlir_scalar_unsigned_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_scalar_signed__")
    set(found_mlir_scalar_signed_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_scalar_float__")
    set(found_mlir_scalar_float_helper TRUE)
  endif()
  if(rs_text MATCHES "mlir___llvmdsdl_plan_validate_delimiter_header__")
    set(found_mlir_delimiter_validate_helper TRUE)
  endif()
endforeach()

if(NOT found_mlir_union_helper)
  message(FATAL_ERROR "Missing MLIR union-tag helper bindings in generated Rust files")
endif()
if(NOT found_mlir_union_validate_helper)
  message(FATAL_ERROR "Missing MLIR union-tag-validate helper bindings in generated Rust files")
endif()
if(NOT found_mlir_union_validate_call)
  message(FATAL_ERROR "Missing MLIR union-tag-validate helper call sites in generated Rust files")
endif()
if(NOT found_mlir_capacity_check_helper)
  message(FATAL_ERROR "Missing MLIR capacity-check helper bindings in generated Rust files")
endif()
if(NOT found_mlir_capacity_check_call)
  message(FATAL_ERROR "Missing MLIR capacity-check helper call sites in generated Rust files")
endif()
if(NOT found_mlir_array_prefix_helper)
  message(FATAL_ERROR "Missing MLIR array-prefix helper bindings in generated Rust files")
endif()
if(NOT found_mlir_array_validate_helper)
  message(FATAL_ERROR "Missing MLIR array-validate helper bindings in generated Rust files")
endif()
if(NOT found_mlir_scalar_unsigned_helper)
  message(FATAL_ERROR "Missing MLIR scalar-unsigned helper bindings in generated Rust files")
endif()
if(NOT found_mlir_scalar_signed_helper)
  message(FATAL_ERROR "Missing MLIR scalar-signed helper bindings in generated Rust files")
endif()
if(NOT found_mlir_scalar_float_helper)
  message(FATAL_ERROR "Missing MLIR scalar-float helper bindings in generated Rust files")
endif()
if(NOT found_mlir_delimiter_validate_helper)
  message(FATAL_ERROR "Missing MLIR delimiter-validate helper bindings in generated Rust files")
endif()

if(NOT dsdl_count EQUAL type_rs_count)
  message(FATAL_ERROR
    "Rust type file count mismatch: dsdl=${dsdl_count}, generated=${type_rs_count}")
endif()

message(STATUS
  "uavcan rust generation check passed: ${dsdl_count} DSDL -> ${type_rs_count} Rust type files")
