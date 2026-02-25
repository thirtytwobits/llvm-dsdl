foreach(var PYTHON_EXECUTABLE REPO_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "${var} is required for runtime semantic-wrapper allowlist validation")
  endif()
endforeach()

set(allowlist_script "${REPO_ROOT}/tools/runtime/validate_semantic_wrapper_allowlist.py")
if(NOT EXISTS "${allowlist_script}")
  message(FATAL_ERROR "runtime allowlist validator script not found: ${allowlist_script}")
endif()

set(generation_script "${REPO_ROOT}/tools/runtime/generate_runtime_semantic_wrappers.py")
if(NOT EXISTS "${generation_script}")
  message(FATAL_ERROR "runtime semantic-wrapper generator script not found: ${generation_script}")
endif()

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${generation_script}"
      --repo-root "${REPO_ROOT}"
      --check
  RESULT_VARIABLE generation_result
  OUTPUT_VARIABLE generation_stdout
  ERROR_VARIABLE generation_stderr
)

if(generation_stdout)
  message(STATUS "runtime semantic-wrapper generation output:\n${generation_stdout}")
endif()

if(generation_result AND NOT generation_result EQUAL 0)
  message(FATAL_ERROR
    "runtime semantic-wrapper generation check failed with exit code ${generation_result}\n${generation_stderr}")
endif()

if(generation_stderr)
  message(STATUS "runtime semantic-wrapper generation diagnostics:\n${generation_stderr}")
endif()

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${allowlist_script}"
      --repo-root "${REPO_ROOT}"
  RESULT_VARIABLE allowlist_result
  OUTPUT_VARIABLE allowlist_stdout
  ERROR_VARIABLE allowlist_stderr
)

if(allowlist_stdout)
  message(STATUS "runtime allowlist validator output:\n${allowlist_stdout}")
endif()

if(allowlist_result AND NOT allowlist_result EQUAL 0)
  message(FATAL_ERROR
    "runtime allowlist validator failed with exit code ${allowlist_result}\n${allowlist_stderr}")
endif()

if(allowlist_stderr)
  message(STATUS "runtime allowlist validator diagnostics:\n${allowlist_stderr}")
endif()

message(STATUS "runtime semantic-wrapper allowlist validation passed")
