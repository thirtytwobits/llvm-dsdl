# ===----------------------------------------------------------------------===//
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------===//

if(NOT DEFINED PYTHON_EXECUTABLE OR PYTHON_EXECUTABLE STREQUAL "")
  message(FATAL_ERROR "PYTHON_EXECUTABLE is required for convergence reporting")
endif()

if(NOT DEFINED REPO_ROOT OR REPO_ROOT STREQUAL "")
  message(FATAL_ERROR "REPO_ROOT is required for convergence reporting")
endif()

if(NOT DEFINED OUTPUT_JSON OR OUTPUT_JSON STREQUAL "")
  message(FATAL_ERROR "OUTPUT_JSON is required for convergence reporting")
endif()

if(NOT DEFINED OUTPUT_MD OR OUTPUT_MD STREQUAL "")
  message(FATAL_ERROR "OUTPUT_MD is required for convergence reporting")
endif()

if(NOT DEFINED BASELINE_JSON OR BASELINE_JSON STREQUAL "")
  message(FATAL_ERROR "BASELINE_JSON is required for convergence reporting")
endif()

if(NOT EXISTS "${BASELINE_JSON}")
  message(FATAL_ERROR "convergence baseline file does not exist: ${BASELINE_JSON}")
endif()

if(NOT DEFINED CONVERGENCE_SCRIPT OR CONVERGENCE_SCRIPT STREQUAL "")
  set(CONVERGENCE_SCRIPT "${REPO_ROOT}/tools/convergence/convergence_report.py")
endif()

if(NOT EXISTS "${CONVERGENCE_SCRIPT}")
  message(FATAL_ERROR "convergence report script does not exist: ${CONVERGENCE_SCRIPT}")
endif()

get_filename_component(output_json_dir "${OUTPUT_JSON}" DIRECTORY)
get_filename_component(output_md_dir "${OUTPUT_MD}" DIRECTORY)
file(MAKE_DIRECTORY "${output_json_dir}")
file(MAKE_DIRECTORY "${output_md_dir}")

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${CONVERGENCE_SCRIPT}"
      --repo-root "${REPO_ROOT}"
      --output-json "${OUTPUT_JSON}"
      --output-md "${OUTPUT_MD}"
      --baseline "${BASELINE_JSON}"
      --check-regressions
  RESULT_VARIABLE convergence_result
  OUTPUT_VARIABLE convergence_stdout
  ERROR_VARIABLE convergence_stderr
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

if(convergence_stdout)
  message(STATUS "convergence report output:\n${convergence_stdout}")
endif()

if(convergence_result AND NOT convergence_result EQUAL 0)
  message(FATAL_ERROR
    "convergence report failed with exit code ${convergence_result}\n"
    "${convergence_stderr}")
endif()

if(convergence_stderr)
  message(STATUS "convergence report diagnostics:\n${convergence_stderr}")
endif()

message(STATUS "convergence report generated at ${OUTPUT_JSON}")
message(STATUS "convergence scorecard generated at ${OUTPUT_MD}")
