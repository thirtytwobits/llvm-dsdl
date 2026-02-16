cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC UAVCAN_ROOT OUT_DIR C_COMPILER)
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
    "${DSDLC}" c
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --out-dir "${OUT_DIR}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "uavcan strict generation failed")
endif()

file(GLOB_RECURSE dsdl_files "${UAVCAN_ROOT}/*.dsdl")
list(LENGTH dsdl_files dsdl_count)

file(GLOB_RECURSE generated_headers "${OUT_DIR}/*.h")
set(filtered_headers "")
foreach(h IN LISTS generated_headers)
  get_filename_component(name "${h}" NAME)
  if(NOT name STREQUAL "dsdl_runtime.h")
    list(APPEND filtered_headers "${h}")
  endif()
endforeach()
list(LENGTH filtered_headers header_count)

if(NOT dsdl_count EQUAL header_count)
  message(FATAL_ERROR
    "header count mismatch: dsdl=${dsdl_count}, generated=${header_count}")
endif()

set(stub_hits "")
foreach(h IN LISTS filtered_headers)
  file(READ "${h}" header_text)
  string(FIND "${header_text}" "dsdl_runtime_stub_" hit_pos)
  if(NOT hit_pos EQUAL -1)
    list(APPEND stub_hits "${h}")
  endif()
endforeach()
if(stub_hits)
  message(FATAL_ERROR
    "generated headers still reference dsdl_runtime_stub_: ${stub_hits}")
endif()

file(GLOB_RECURSE generated_impls "${OUT_DIR}/*.c")
set(generic_lowering_hits "")
set(missing_capacity_call_hits "")
set(missing_capacity_helper_hits "")
set(found_union_tag_call 0)
set(found_union_tag_helper 0)
set(found_scalar_unsigned_call 0)
set(found_scalar_unsigned_helper 0)
foreach(c_file IN LISTS generated_impls)
  file(READ "${c_file}" impl_text)
  string(FIND "${impl_text}" "Generic bitstream mapping" hit_pos)
  if(NOT hit_pos EQUAL -1)
    list(APPEND generic_lowering_hits "${c_file}")
  endif()
  string(FIND "${impl_text}" "_err_capacity = __llvmdsdl_plan_capacity_check__"
         capacity_call_pos)
  if(capacity_call_pos EQUAL -1)
    list(APPEND missing_capacity_call_hits "${c_file}")
  endif()
  string(FIND "${impl_text}" "int8_t __llvmdsdl_plan_capacity_check__"
         capacity_helper_pos)
  if(capacity_helper_pos EQUAL -1)
    list(APPEND missing_capacity_helper_hits "${c_file}")
  endif()
  string(FIND "${impl_text}" "_err_union_tag = __llvmdsdl_plan_validate_union_tag__"
         union_tag_call_pos)
  if(NOT union_tag_call_pos EQUAL -1)
    set(found_union_tag_call 1)
  endif()
  string(FIND "${impl_text}" "int8_t __llvmdsdl_plan_validate_union_tag__"
         union_tag_helper_pos)
  if(NOT union_tag_helper_pos EQUAL -1)
    set(found_union_tag_helper 1)
  endif()
  string(FIND "${impl_text}" "= (uint64_t)__llvmdsdl_plan_scalar_unsigned__"
         scalar_call_pos)
  if(NOT scalar_call_pos EQUAL -1)
    set(found_scalar_unsigned_call 1)
  endif()
  string(FIND "${impl_text}" "int64_t __llvmdsdl_plan_scalar_unsigned__"
         scalar_helper_pos)
  if(NOT scalar_helper_pos EQUAL -1)
    set(found_scalar_unsigned_helper 1)
  endif()
endforeach()
if(generic_lowering_hits)
  message(FATAL_ERROR
    "generated C implementations unexpectedly used generic lowering: ${generic_lowering_hits}")
endif()
if(missing_capacity_call_hits)
  message(FATAL_ERROR
    "generated C implementations missing capacity-helper call: ${missing_capacity_call_hits}")
endif()
if(missing_capacity_helper_hits)
  message(FATAL_ERROR
    "generated C implementations missing lowered capacity-helper body: ${missing_capacity_helper_hits}")
endif()
if(NOT found_union_tag_call)
  message(FATAL_ERROR
    "generated C implementations did not use lowered union-tag validation helper")
endif()
if(NOT found_union_tag_helper)
  message(FATAL_ERROR
    "generated C implementations did not emit lowered union-tag validation helper body")
endif()
if(NOT found_scalar_unsigned_call)
  message(FATAL_ERROR
    "generated C implementations did not use lowered scalar unsigned helper")
endif()
if(NOT found_scalar_unsigned_helper)
  message(FATAL_ERROR
    "generated C implementations did not emit lowered scalar unsigned helper body")
endif()

set(scratch_dir "${OUT_DIR}/.compile-check")
file(MAKE_DIRECTORY "${scratch_dir}")

set(index 0)
foreach(h IN LISTS filtered_headers)
  math(EXPR index "${index} + 1")
  file(RELATIVE_PATH rel_header "${OUT_DIR}" "${h}")

  set(tu "${scratch_dir}/tu_${index}.c")
  set(obj "${scratch_dir}/tu_${index}.o")
  file(WRITE "${tu}" "#include \"${rel_header}\"\nint main(void) { return 0; }\n")

  execute_process(
    COMMAND
      "${C_COMPILER}"
        -std=c11
        -Wall
        -Wextra
        -Werror
        -I
        "${OUT_DIR}"
        "${tu}"
        -c
        -o "${obj}"
    RESULT_VARIABLE cc_result
    OUTPUT_VARIABLE cc_stdout
    ERROR_VARIABLE cc_stderr
  )
  if(NOT cc_result EQUAL 0)
    message(STATUS "Failed header: ${rel_header}")
    message(STATUS "compiler stdout:\n${cc_stdout}")
    message(STATUS "compiler stderr:\n${cc_stderr}")
    message(FATAL_ERROR "Generated header compile check failed")
  endif()
endforeach()

message(STATUS
  "uavcan strict generation check passed: ${dsdl_count} DSDL -> ${header_count} headers")
