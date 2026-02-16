cmake_minimum_required(VERSION 3.24)

foreach(var C_COMPILER SOURCE_ROOT INCLUDE_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${C_COMPILER}")
  message(FATAL_ERROR "C compiler not found: ${C_COMPILER}")
endif()

if(NOT EXISTS "${SOURCE_ROOT}")
  message(FATAL_ERROR "Generated source root not found: ${SOURCE_ROOT}")
endif()

file(GLOB_RECURSE generated_c_files "${SOURCE_ROOT}/*.c")
list(LENGTH generated_c_files generated_count)

if(generated_count EQUAL 0)
  message(FATAL_ERROR "No generated C sources found under: ${SOURCE_ROOT}")
endif()

set(scratch_dir "${SOURCE_ROOT}/.compile-check")
file(REMOVE_RECURSE "${scratch_dir}")
file(MAKE_DIRECTORY "${scratch_dir}")

set(index 0)
foreach(src IN LISTS generated_c_files)
  math(EXPR index "${index} + 1")
  set(obj "${scratch_dir}/tu_${index}.o")

  execute_process(
    COMMAND
      "${C_COMPILER}"
        -std=c11
        -Wall
        -Wextra
        -Werror
        -I "${INCLUDE_ROOT}"
        "${src}"
        -c
        -o "${obj}"
    RESULT_VARIABLE cc_result
    OUTPUT_VARIABLE cc_stdout
    ERROR_VARIABLE cc_stderr
  )
  if(NOT cc_result EQUAL 0)
    message(STATUS "Failed source: ${src}")
    message(STATUS "compiler stdout:\n${cc_stdout}")
    message(STATUS "compiler stderr:\n${cc_stderr}")
    message(FATAL_ERROR "Generated C source compile check failed")
  endif()
endforeach()

message(STATUS
  "Compiled ${generated_count} generated C source file(s) under ${SOURCE_ROOT}")
