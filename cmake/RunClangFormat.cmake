#===----------------------------------------------------------------------===#
##
## @file
## CMake script that rewrites project source formatting with clang-format.
##
#===----------------------------------------------------------------------===#

if(NOT DEFINED CLANG_FORMAT OR CLANG_FORMAT STREQUAL "" OR
   CLANG_FORMAT MATCHES "-NOTFOUND$")
  message(FATAL_ERROR
    "clang-format executable was not provided. "
    "Install clang-format and re-run target format-source.")
endif()

if(NOT DEFINED LLVMDSDL_SOURCE_DIR OR LLVMDSDL_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "LLVMDSDL_SOURCE_DIR must be provided.")
endif()

string(REGEX REPLACE "([][+.*()^$?{}|\\\\])" "\\\\\\1"
  _llvmdsdl_source_dir_regex "${LLVMDSDL_SOURCE_DIR}")

set(_llvmdsdl_format_dirs
  include
  lib
  runtime
  test
  tools
)

set(_llvmdsdl_format_files)
foreach(_dir IN LISTS _llvmdsdl_format_dirs)
  if(NOT EXISTS "${LLVMDSDL_SOURCE_DIR}/${_dir}")
    continue()
  endif()
  file(GLOB_RECURSE _dir_files
    LIST_DIRECTORIES FALSE
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.c"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.cc"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.cpp"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.cxx"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.h"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.hh"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.hpp"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.inc"
    "${LLVMDSDL_SOURCE_DIR}/${_dir}/*.td"
  )
  list(APPEND _llvmdsdl_format_files ${_dir_files})
endforeach()

list(REMOVE_DUPLICATES _llvmdsdl_format_files)
list(SORT _llvmdsdl_format_files)

set(_llvmdsdl_filtered_format_files)
foreach(_file IN LISTS _llvmdsdl_format_files)
  if(_file MATCHES "^${_llvmdsdl_source_dir_regex}/examples/")
    continue()
  endif()
  list(APPEND _llvmdsdl_filtered_format_files "${_file}")
endforeach()
set(_llvmdsdl_format_files "${_llvmdsdl_filtered_format_files}")

if(NOT _llvmdsdl_format_files)
  message(STATUS "No source files found for clang-format rewrite.")
  return()
endif()

set(_failed_files)
foreach(_file IN LISTS _llvmdsdl_format_files)
  execute_process(
    COMMAND "${CLANG_FORMAT}" -i --style=file "${_file}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(NOT _result EQUAL 0)
    list(APPEND _failed_files "${_file}")
    if(NOT DEFINED _first_error AND NOT _stderr STREQUAL "")
      set(_first_error "${_stderr}")
    endif()
  endif()
endforeach()

if(_failed_files)
  list(LENGTH _failed_files _failed_count)
  list(JOIN _failed_files "\n  " _failed_list)
  message(FATAL_ERROR
    "clang-format rewrite failed for ${_failed_count} file(s):\n"
    "  ${_failed_list}\n"
    "First clang-format error:\n${_first_error}")
endif()

list(LENGTH _llvmdsdl_format_files _rewritten_count)
message(STATUS
  "clang-format rewrite completed for ${_rewritten_count} file(s).")
