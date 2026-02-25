foreach(var PYTHON_EXECUTABLE REPO_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "${var} is required for runtime allowlist self-test")
  endif()
endforeach()

set(selftest_script "${REPO_ROOT}/tools/runtime/test_validate_semantic_wrapper_allowlist.py")
if(NOT EXISTS "${selftest_script}")
  message(FATAL_ERROR "runtime allowlist self-test script not found: ${selftest_script}")
endif()

set(generation_selftest_script "${REPO_ROOT}/tools/runtime/test_generate_runtime_semantic_wrappers.py")
if(NOT EXISTS "${generation_selftest_script}")
  message(FATAL_ERROR "runtime generator self-test script not found: ${generation_selftest_script}")
endif()

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${selftest_script}"
      --repo-root "${REPO_ROOT}"
  RESULT_VARIABLE selftest_result
  OUTPUT_VARIABLE selftest_stdout
  ERROR_VARIABLE selftest_stderr
)

if(selftest_stdout)
  message(STATUS "runtime allowlist self-test output:\n${selftest_stdout}")
endif()

if(selftest_result AND NOT selftest_result EQUAL 0)
  message(FATAL_ERROR
    "runtime allowlist self-test failed with exit code ${selftest_result}\n${selftest_stderr}")
endif()

if(selftest_stderr)
  message(STATUS "runtime allowlist self-test diagnostics:\n${selftest_stderr}")
endif()

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${generation_selftest_script}"
      --repo-root "${REPO_ROOT}"
  RESULT_VARIABLE generation_selftest_result
  OUTPUT_VARIABLE generation_selftest_stdout
  ERROR_VARIABLE generation_selftest_stderr
)

if(generation_selftest_stdout)
  message(STATUS "runtime generator self-test output:\n${generation_selftest_stdout}")
endif()

if(generation_selftest_result AND NOT generation_selftest_result EQUAL 0)
  message(FATAL_ERROR
    "runtime generator self-test failed with exit code ${generation_selftest_result}\n${generation_selftest_stderr}")
endif()

if(generation_selftest_stderr)
  message(STATUS "runtime generator self-test diagnostics:\n${generation_selftest_stderr}")
endif()

message(STATUS "runtime semantic-wrapper allowlist/generation self-tests passed")
