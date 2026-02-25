foreach(var PYTHON_EXECUTABLE REPO_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "${var} is required for execution-engine boundary validation")
  endif()
endforeach()

set(validator_script "${REPO_ROOT}/tools/convergence/validate_execution_engine_boundaries.py")
if(NOT EXISTS "${validator_script}")
  message(FATAL_ERROR "execution-engine boundary validator script not found: ${validator_script}")
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
  message(STATUS "execution-engine boundary validator output:\n${validator_stdout}")
endif()

if(validator_result AND NOT validator_result EQUAL 0)
  message(FATAL_ERROR
    "execution-engine boundary validator failed with exit code ${validator_result}\n${validator_stderr}")
endif()

if(validator_stderr)
  message(STATUS "execution-engine boundary validator diagnostics:\n${validator_stderr}")
endif()

message(STATUS "execution-engine boundary validation passed")
