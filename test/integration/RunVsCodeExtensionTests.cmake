cmake_minimum_required(VERSION 3.24)

foreach(var NPM_EXECUTABLE EXTENSION_DIR DSDLD_BINARY)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${NPM_EXECUTABLE}")
  message(FATAL_ERROR "npm executable not found: ${NPM_EXECUTABLE}")
endif()
if(NOT EXISTS "${EXTENSION_DIR}/package.json")
  message(FATAL_ERROR "VSCode extension package.json not found: ${EXTENSION_DIR}/package.json")
endif()
if(NOT EXISTS "${DSDLD_BINARY}")
  message(FATAL_ERROR "dsdld executable not found: ${DSDLD_BINARY}")
endif()

execute_process(
  COMMAND "${NPM_EXECUTABLE}" install --no-fund --no-audit
  WORKING_DIRECTORY "${EXTENSION_DIR}"
  RESULT_VARIABLE npm_install_result
  OUTPUT_VARIABLE npm_install_stdout
  ERROR_VARIABLE npm_install_stderr
)
if(NOT npm_install_result EQUAL 0)
  message(STATUS "npm install stdout:\n${npm_install_stdout}")
  message(STATUS "npm install stderr:\n${npm_install_stderr}")
  message(FATAL_ERROR "VSCode extension dependency install failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DSDLD_BINARY=${DSDLD_BINARY}"
    "${NPM_EXECUTABLE}" test
  WORKING_DIRECTORY "${EXTENSION_DIR}"
  RESULT_VARIABLE npm_test_result
  OUTPUT_VARIABLE npm_test_stdout
  ERROR_VARIABLE npm_test_stderr
)
if(NOT npm_test_result EQUAL 0)
  message(STATUS "npm test stdout:\n${npm_test_stdout}")
  message(STATUS "npm test stderr:\n${npm_test_stderr}")
  message(FATAL_ERROR "VSCode extension smoke tests failed")
endif()

message(STATUS "VSCode extension smoke tests passed")
