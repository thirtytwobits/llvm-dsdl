#===----------------------------------------------------------------------===#
#
# Part of the OpenCyphal project, under the MIT licence
# SPDX-License-Identifier: MIT
#
#===----------------------------------------------------------------------===#

foreach(var PYTHON_EXECUTABLE REPO_ROOT CTEST_TEST_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "${var} is required for convergence matrix report self-test")
  endif()
endforeach()

set(selftest_script "${REPO_ROOT}/tools/convergence/test_matrix_reports.py")
if(NOT EXISTS "${selftest_script}")
  message(FATAL_ERROR "convergence matrix report self-test script not found: ${selftest_script}")
endif()

set(selftest_args
  "${PYTHON_EXECUTABLE}"
  "${selftest_script}"
  --repo-root
  "${REPO_ROOT}"
  --ctest-test-dir
  "${CTEST_TEST_DIR}"
)

if(DEFINED CTEST_CONFIG AND NOT CTEST_CONFIG STREQUAL "")
  list(APPEND selftest_args --ctest-config "${CTEST_CONFIG}")
endif()

execute_process(
  COMMAND ${selftest_args}
  RESULT_VARIABLE selftest_result
  OUTPUT_VARIABLE selftest_stdout
  ERROR_VARIABLE selftest_stderr
)

if(selftest_stdout)
  message(STATUS "convergence matrix report self-test output:\n${selftest_stdout}")
endif()

if(selftest_result AND NOT selftest_result EQUAL 0)
  message(FATAL_ERROR
    "convergence matrix report self-test failed with exit code ${selftest_result}\n${selftest_stderr}")
endif()

if(selftest_stderr)
  message(STATUS "convergence matrix report self-test diagnostics:\n${selftest_stderr}")
endif()

message(STATUS "convergence matrix report self-test passed")
