#===----------------------------------------------------------------------===#
##
## @file
## CMake script that validates project sources with clang-tidy.
##
#===----------------------------------------------------------------------===#

if(NOT DEFINED CLANG_TIDY OR CLANG_TIDY STREQUAL "" OR
   CLANG_TIDY MATCHES "-NOTFOUND$")
  message(FATAL_ERROR
    "clang-tidy executable was not provided. "
    "Install clang-tidy and re-run target check-clang-tidy.")
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

file(READ "${_llvmdsdl_compile_commands}" _llvmdsdl_compile_db)
string(JSON _llvmdsdl_entry_count LENGTH "${_llvmdsdl_compile_db}")

if(_llvmdsdl_entry_count EQUAL 0)
  message(STATUS "No compile commands found for clang-tidy check.")
  return()
endif()

math(EXPR _llvmdsdl_last_index "${_llvmdsdl_entry_count} - 1")

string(REGEX REPLACE "([][+.*()^$?{}|\\\\])" "\\\\\\1"
  _llvmdsdl_source_dir_regex "${LLVMDSDL_SOURCE_DIR}")

set(_llvmdsdl_tidy_files)
set(_llvmdsdl_tidy_excluded_files
  "${LLVMDSDL_SOURCE_DIR}/lib/IR/DSDLDialect.cpp"
)
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

  if(_abs_file MATCHES "^${_llvmdsdl_source_dir_regex}/submodules/")
    continue()
  endif()

  if(_abs_file MATCHES "^${_llvmdsdl_source_dir_regex}/examples/")
    continue()
  endif()

  list(FIND _llvmdsdl_tidy_excluded_files "${_abs_file}" _llvmdsdl_excluded_index)
  if(NOT _llvmdsdl_excluded_index EQUAL -1)
    continue()
  endif()

  if(NOT _abs_file MATCHES "\\.(c|cc|cpp|cxx)$")
    continue()
  endif()

  list(APPEND _llvmdsdl_tidy_files "${_abs_file}")
endforeach()

list(REMOVE_DUPLICATES _llvmdsdl_tidy_files)
list(SORT _llvmdsdl_tidy_files)

if(NOT _llvmdsdl_tidy_files)
  message(STATUS
    "No project C/C++ source files found in compile commands for clang-tidy.")
  return()
endif()

set(_llvmdsdl_tidy_args
  -p "${LLVMDSDL_BINARY_DIR}"
  --warnings-as-errors=*
  "--header-filter=^${_llvmdsdl_source_dir_regex}/"
)

if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    RESULT_VARIABLE _sdk_result
    OUTPUT_VARIABLE _sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_sdk_result EQUAL 0 AND NOT _sdk_path STREQUAL "")
    list(APPEND _llvmdsdl_tidy_args
      --extra-arg-before=-isysroot
      --extra-arg-before=${_sdk_path}
    )
  endif()
endif()

set(_failed_files)
foreach(_file IN LISTS _llvmdsdl_tidy_files)
  execute_process(
    COMMAND "${CLANG_TIDY}" ${_llvmdsdl_tidy_args} "${_file}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(NOT _result EQUAL 0)
    list(APPEND _failed_files "${_file}")
    if(NOT DEFINED _first_error)
      set(_combined_output "${_stdout}")
      if(NOT _stderr STREQUAL "")
        if(NOT _combined_output STREQUAL "")
          string(APPEND _combined_output "\n")
        endif()
        string(APPEND _combined_output "${_stderr}")
      endif()
      if(NOT _combined_output STREQUAL "")
        set(_first_error "${_combined_output}")
      endif()
    endif()
  endif()
endforeach()

if(_failed_files)
  list(LENGTH _failed_files _failed_count)
  list(JOIN _failed_files "\n  " _failed_list)
  message(FATAL_ERROR
    "clang-tidy check failed for ${_failed_count} file(s):\n"
    "  ${_failed_list}\n"
    "First clang-tidy diagnostic:\n${_first_error}")
endif()

list(LENGTH _llvmdsdl_tidy_files _checked_count)
message(STATUS
  "clang-tidy check passed for ${_checked_count} file(s).")
