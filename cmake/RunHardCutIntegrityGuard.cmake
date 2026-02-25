foreach(var PYTHON_EXECUTABLE REPO_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "${var} is required for hard-cut integrity validation")
  endif()
endforeach()

set(validator_script "${REPO_ROOT}/tools/convergence/validate_hard_cut_integrity.py")
if(NOT EXISTS "${validator_script}")
  message(FATAL_ERROR "hard-cut integrity validator script not found: ${validator_script}")
endif()

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "${validator_script}"
      --repo-root "${REPO_ROOT}"
  RESULT_VARIABLE validator_result
  OUTPUT_VARIABLE validator_stdout
  ERROR_VARIABLE validator_stderr
)

if(validator_stdout)
  message(STATUS "hard-cut integrity validator output:\n${validator_stdout}")
endif()

if(validator_result AND NOT validator_result EQUAL 0)
  message(FATAL_ERROR
    "hard-cut integrity validator failed with exit code ${validator_result}\n${validator_stderr}")
endif()

if(validator_stderr)
  message(STATUS "hard-cut integrity validator diagnostics:\n${validator_stderr}")
endif()

message(STATUS "hard-cut integrity validation passed")
