cmake_minimum_required(VERSION 3.24)

foreach(var
        LLVM_PROFDATA
        LLVM_COV
        PROFILE_DIR
        PROFILE_DATA
        REPORT_TXT
        REPORT_LCOV
        REPORT_HTML_DIR
        SOURCE_DIR
        BINARY_DIR
        OBJECTS)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

file(GLOB_RECURSE profile_raw_files "${PROFILE_DIR}/*.profraw")
if(NOT profile_raw_files)
  message(FATAL_ERROR
    "No *.profraw files were found in ${PROFILE_DIR}. "
    "Run coverage-run first.")
endif()

set(coverage_objects "")
foreach(obj IN LISTS OBJECTS)
  if(EXISTS "${obj}")
    list(APPEND coverage_objects "${obj}")
  else()
    message(WARNING "Skipping missing coverage object: ${obj}")
  endif()
endforeach()

if(NOT coverage_objects)
  message(FATAL_ERROR "No coverage objects exist for llvm-cov report generation.")
endif()

get_filename_component(profile_data_dir "${PROFILE_DATA}" DIRECTORY)
file(MAKE_DIRECTORY "${profile_data_dir}")
file(MAKE_DIRECTORY "${REPORT_HTML_DIR}")
get_filename_component(report_txt_dir "${REPORT_TXT}" DIRECTORY)
file(MAKE_DIRECTORY "${report_txt_dir}")
get_filename_component(report_lcov_dir "${REPORT_LCOV}" DIRECTORY)
file(MAKE_DIRECTORY "${report_lcov_dir}")

execute_process(
  COMMAND "${LLVM_PROFDATA}" merge -sparse ${profile_raw_files} -o "${PROFILE_DATA}"
  RESULT_VARIABLE prof_merge_result
  OUTPUT_VARIABLE prof_merge_stdout
  ERROR_VARIABLE prof_merge_stderr
)
if(NOT prof_merge_result EQUAL 0)
  message(FATAL_ERROR
    "llvm-profdata merge failed.\n${prof_merge_stdout}\n${prof_merge_stderr}")
endif()

set(common_cov_args
    "-instr-profile=${PROFILE_DATA}"
    "-compilation-dir=${BINARY_DIR}"
    "-path-equivalence=${BINARY_DIR},${SOURCE_DIR}")

if(DEFINED IGNORE_FILENAME_REGEX AND NOT "${IGNORE_FILENAME_REGEX}" STREQUAL "")
  list(APPEND common_cov_args "-ignore-filename-regex=${IGNORE_FILENAME_REGEX}")
endif()

set(cov_object_args "")
foreach(obj IN LISTS coverage_objects)
  list(APPEND cov_object_args "-object=${obj}")
endforeach()

execute_process(
  COMMAND "${LLVM_COV}" report ${common_cov_args} ${cov_object_args}
  RESULT_VARIABLE report_result
  OUTPUT_FILE "${REPORT_TXT}"
  ERROR_VARIABLE report_stderr
)
if(NOT report_result EQUAL 0)
  message(FATAL_ERROR "llvm-cov report failed.\n${report_stderr}")
endif()

execute_process(
  COMMAND "${LLVM_COV}" export -format=lcov ${common_cov_args} ${cov_object_args}
  RESULT_VARIABLE export_result
  OUTPUT_FILE "${REPORT_LCOV}"
  ERROR_VARIABLE export_stderr
)
if(NOT export_result EQUAL 0)
  message(FATAL_ERROR "llvm-cov export failed.\n${export_stderr}")
endif()

execute_process(
  COMMAND "${LLVM_COV}" show -format=html "-output-dir=${REPORT_HTML_DIR}" ${common_cov_args} ${cov_object_args}
  RESULT_VARIABLE show_result
  OUTPUT_VARIABLE show_stdout
  ERROR_VARIABLE show_stderr
)
if(NOT show_result EQUAL 0)
  message(FATAL_ERROR "llvm-cov show failed.\n${show_stdout}\n${show_stderr}")
endif()

message(STATUS "Coverage profile data: ${PROFILE_DATA}")
message(STATUS "Coverage summary report: ${REPORT_TXT}")
message(STATUS "Coverage LCOV report: ${REPORT_LCOV}")
message(STATUS "Coverage HTML report directory: ${REPORT_HTML_DIR}")
