cmake_minimum_required(VERSION 3.24)

foreach(var GO_EXECUTABLE SOURCE_ROOT OUT_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${GO_EXECUTABLE}")
  message(FATAL_ERROR "go executable not found: ${GO_EXECUTABLE}")
endif()

set(runtime_dir "${SOURCE_ROOT}/runtime/go")
if(NOT EXISTS "${runtime_dir}/go.mod")
  message(FATAL_ERROR "runtime go.mod missing: ${runtime_dir}/go.mod")
endif()
if(NOT EXISTS "${runtime_dir}/dsdl_runtime.go")
  message(FATAL_ERROR "runtime source missing: ${runtime_dir}/dsdl_runtime.go")
endif()
if(NOT EXISTS "${runtime_dir}/dsdl_runtime_test.go")
  message(FATAL_ERROR "runtime test missing: ${runtime_dir}/dsdl_runtime_test.go")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(go_cache "${OUT_DIR}/.gocache")
set(go_mod_cache "${OUT_DIR}/.gomodcache")
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "GOCACHE=${go_cache}"
      "GOMODCACHE=${go_mod_cache}"
      "${GO_EXECUTABLE}" test ./...
  WORKING_DIRECTORY "${runtime_dir}"
  RESULT_VARIABLE test_result
  OUTPUT_VARIABLE test_stdout
  ERROR_VARIABLE test_stderr
)
if(NOT test_result EQUAL 0)
  message(STATUS "go test stdout:\n${test_stdout}")
  message(STATUS "go test stderr:\n${test_stderr}")
  message(FATAL_ERROR "Go runtime unit tests failed")
endif()

file(WRITE "${OUT_DIR}/go-runtime-unit-summary.txt" "${test_stdout}\n")
message(STATUS "Go runtime unit tests summary:\n${test_stdout}")
