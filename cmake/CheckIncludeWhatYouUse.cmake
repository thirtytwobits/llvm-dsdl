#===----------------------------------------------------------------------===#
##
## @file
## CMake script that validates project includes with include-what-you-use.
##
#===----------------------------------------------------------------------===#

if(NOT DEFINED INCLUDE_WHAT_YOU_USE OR INCLUDE_WHAT_YOU_USE STREQUAL "" OR
   INCLUDE_WHAT_YOU_USE MATCHES "-NOTFOUND$")
  message(FATAL_ERROR
    "include-what-you-use executable was not provided. "
    "Install include-what-you-use and re-run target check-iwyu.")
endif()

if(NOT DEFINED LLVMDSDL_SOURCE_DIR OR LLVMDSDL_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "LLVMDSDL_SOURCE_DIR must be provided.")
endif()

if(NOT DEFINED LLVMDSDL_BINARY_DIR OR LLVMDSDL_BINARY_DIR STREQUAL "")
  message(FATAL_ERROR "LLVMDSDL_BINARY_DIR must be provided.")
endif()

set(_llvmdsdl_compile_commands
  "${LLVMDSDL_BINARY_DIR}/compile_commands.json")
if(NOT EXISTS "${_llvmdsdl_compile_commands}")
  message(FATAL_ERROR
    "compile_commands.json not found at:\n"
    "  ${_llvmdsdl_compile_commands}\n"
    "Configure the build directory first with "
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON.")
endif()

if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    RESULT_VARIABLE _sdk_result
    OUTPUT_VARIABLE _sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

file(READ "${_llvmdsdl_compile_commands}" _llvmdsdl_compile_db)
string(JSON _llvmdsdl_entry_count LENGTH "${_llvmdsdl_compile_db}")

if(_llvmdsdl_entry_count EQUAL 0)
  message(STATUS "No compile commands found for include-what-you-use check.")
  return()
endif()

math(EXPR _llvmdsdl_last_index "${_llvmdsdl_entry_count} - 1")

string(REGEX REPLACE "([][+.*()^$?{}|\\\\])" "\\\\\\1"
  _llvmdsdl_source_dir_regex "${LLVMDSDL_SOURCE_DIR}")

set(_checked_files)
set(_failed_files)

set(_llvmdsdl_iwyu_args)
set(_llvmdsdl_mapping_file
  "${LLVMDSDL_SOURCE_DIR}/cmake/IwyuMappings.imp")
if(EXISTS "${_llvmdsdl_mapping_file}")
  list(APPEND _llvmdsdl_iwyu_args
    -Xiwyu
    --mapping_file=${_llvmdsdl_mapping_file}
  )
endif()

foreach(_index RANGE 0 ${_llvmdsdl_last_index})
  string(JSON _file GET "${_llvmdsdl_compile_db}" ${_index} file)
  if(_file STREQUAL "")
    continue()
  endif()

  if(IS_ABSOLUTE "${_file}")
    set(_abs_file "${_file}")
  else()
    string(JSON _entry_dir GET "${_llvmdsdl_compile_db}" ${_index} directory)
    if(_entry_dir STREQUAL "")
      continue()
    endif()
    cmake_path(ABSOLUTE_PATH _file
      BASE_DIRECTORY "${_entry_dir}"
      OUTPUT_VARIABLE _abs_file)
  endif()

  cmake_path(NORMAL_PATH _abs_file OUTPUT_VARIABLE _abs_file)

  if(NOT _abs_file MATCHES "^${_llvmdsdl_source_dir_regex}/")
    continue()
  endif()

  if(NOT _abs_file MATCHES "\\.(c|cc|cpp|cxx)$")
    continue()
  endif()

  string(JSON _entry_dir GET "${_llvmdsdl_compile_db}" ${_index} directory)
  string(JSON _command GET "${_llvmdsdl_compile_db}" ${_index} command)
  if(_entry_dir STREQUAL "" OR _command STREQUAL "")
    continue()
  endif()

  separate_arguments(_compile_args UNIX_COMMAND "${_command}")
  if(NOT _compile_args)
    continue()
  endif()

  list(POP_FRONT _compile_args _compiler)

  set(_iwyu_command
    "${INCLUDE_WHAT_YOU_USE}"
    ${_llvmdsdl_iwyu_args}
    ${_compile_args}
  )

  if(APPLE AND _sdk_result EQUAL 0 AND NOT _sdk_path STREQUAL "")
    list(APPEND _iwyu_command
      -isysroot
      "${_sdk_path}"
    )
  endif()

  execute_process(
    COMMAND ${_iwyu_command}
    WORKING_DIRECTORY "${_entry_dir}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )

  set(_combined_output "${_stdout}")
  if(NOT _stderr STREQUAL "")
    if(NOT _combined_output STREQUAL "")
      string(APPEND _combined_output "\n")
    endif()
    string(APPEND _combined_output "${_stderr}")
  endif()

  string(REGEX REPLACE "([][+.*()^$?{}|\\\\])" "\\\\\\1"
    _abs_file_regex "${_abs_file}")

  set(_has_include_fixes FALSE)
  if(_combined_output MATCHES
      "${_abs_file_regex}[^\n\r]*should add these lines:" OR
     _combined_output MATCHES
      "${_abs_file_regex}[^\n\r]*should remove these lines:")
    set(_has_include_fixes TRUE)
  endif()

  list(APPEND _checked_files "${_abs_file}")
  if((NOT _result EQUAL 0) OR _has_include_fixes)
    list(APPEND _failed_files "${_abs_file}")
    if(NOT DEFINED _first_error)
      if(NOT _combined_output STREQUAL "")
        set(_first_error "${_combined_output}")
      else()
        set(_first_error
          "include-what-you-use exited with code ${_result}.")
      endif()
    endif()
  endif()
endforeach()

list(REMOVE_DUPLICATES _checked_files)
list(REMOVE_DUPLICATES _failed_files)
list(SORT _checked_files)
list(SORT _failed_files)

if(NOT _checked_files)
  message(STATUS
    "No project C/C++ translation units found for include-what-you-use check.")
  return()
endif()

if(_failed_files)
  list(LENGTH _failed_files _failed_count)
  list(JOIN _failed_files "\n  " _failed_list)
  message(FATAL_ERROR
    "include-what-you-use check failed for ${_failed_count} file(s):\n"
    "  ${_failed_list}\n"
    "First include-what-you-use diagnostic:\n${_first_error}")
endif()

list(LENGTH _checked_files _checked_count)
message(STATUS
  "include-what-you-use check passed for ${_checked_count} file(s).")
