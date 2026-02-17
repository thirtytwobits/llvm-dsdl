cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC UAVCAN_ROOT OUT_DIR C_COMPILER CXX_COMPILER SOURCE_ROOT)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()

if(NOT EXISTS "${UAVCAN_ROOT}")
  message(FATAL_ERROR "uavcan root not found: ${UAVCAN_ROOT}")
endif()

if(NOT EXISTS "${C_COMPILER}")
  message(FATAL_ERROR "C compiler not found: ${C_COMPILER}")
endif()

if(NOT EXISTS "${CXX_COMPILER}")
  message(FATAL_ERROR "C++ compiler not found: ${CXX_COMPILER}")
endif()

if(NOT DEFINED CPP_PROFILE OR "${CPP_PROFILE}" STREQUAL "")
  set(CPP_PROFILE "std")
endif()
if(NOT (CPP_PROFILE STREQUAL "std" OR CPP_PROFILE STREQUAL "pmr"))
  message(FATAL_ERROR "CPP_PROFILE must be one of: std, pmr")
endif()

set(parity_main "${SOURCE_ROOT}/test/integration/CppCParityMain.cpp")
if(NOT EXISTS "${parity_main}")
  message(FATAL_ERROR "parity harness source missing: ${parity_main}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(c_out "${OUT_DIR}/c")
set(cpp_out "${OUT_DIR}/cpp")
set(build_out "${OUT_DIR}/build")
file(MAKE_DIRECTORY "${c_out}")
file(MAKE_DIRECTORY "${cpp_out}")
file(MAKE_DIRECTORY "${build_out}")

execute_process(
  COMMAND
    "${DSDLC}" c
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --out-dir "${c_out}"
  RESULT_VARIABLE c_result
  OUTPUT_VARIABLE c_stdout
  ERROR_VARIABLE c_stderr
)
if(NOT c_result EQUAL 0)
  message(STATUS "dsdlc c stdout:\n${c_stdout}")
  message(STATUS "dsdlc c stderr:\n${c_stderr}")
  message(FATAL_ERROR "failed to generate C output for parity harness")
endif()

execute_process(
  COMMAND
    "${DSDLC}" cpp
      --root-namespace-dir "${UAVCAN_ROOT}"
      --strict
      --cpp-profile "${CPP_PROFILE}"
      --out-dir "${cpp_out}"
  RESULT_VARIABLE cpp_result
  OUTPUT_VARIABLE cpp_stdout
  ERROR_VARIABLE cpp_stderr
)
if(NOT cpp_result EQUAL 0)
  message(STATUS "dsdlc cpp stdout:\n${cpp_stdout}")
  message(STATUS "dsdlc cpp stderr:\n${cpp_stderr}")
  message(FATAL_ERROR "failed to generate C++ ${CPP_PROFILE} output for parity harness")
endif()

set(main_obj "${build_out}/cpp_c_parity_main.o")
execute_process(
  COMMAND
    "${CXX_COMPILER}"
      -std=c++23
      -Wall
      -Wextra
      -Werror
      -I "${c_out}"
      -I "${cpp_out}"
      -c "${parity_main}"
      -o "${main_obj}"
  RESULT_VARIABLE main_cc_result
  OUTPUT_VARIABLE main_cc_stdout
  ERROR_VARIABLE main_cc_stderr
)
if(NOT main_cc_result EQUAL 0)
  message(STATUS "C++ compile stdout:\n${main_cc_stdout}")
  message(STATUS "C++ compile stderr:\n${main_cc_stderr}")
  message(FATAL_ERROR "failed to compile C/C++ parity harness main")
endif()

file(GLOB_RECURSE generated_c_sources "${c_out}/*.c")
list(LENGTH generated_c_sources generated_c_count)
if(generated_c_count EQUAL 0)
  message(FATAL_ERROR "no generated C implementation sources under ${c_out}")
endif()

set(c_obj_dir "${build_out}/cobj")
file(MAKE_DIRECTORY "${c_obj_dir}")
set(c_objs "")
set(c_index 0)
foreach(src IN LISTS generated_c_sources)
  math(EXPR c_index "${c_index} + 1")
  set(obj "${c_obj_dir}/generated_${c_index}.o")
  execute_process(
    COMMAND
      "${C_COMPILER}"
        -std=c11
        -Wall
        -Wextra
        -Werror
        -I "${c_out}"
        -c "${src}"
        -o "${obj}"
    RESULT_VARIABLE c_cc_result
    OUTPUT_VARIABLE c_cc_stdout
    ERROR_VARIABLE c_cc_stderr
  )
  if(NOT c_cc_result EQUAL 0)
    message(STATUS "failed source: ${src}")
    message(STATUS "C compile stdout:\n${c_cc_stdout}")
    message(STATUS "C compile stderr:\n${c_cc_stderr}")
    message(FATAL_ERROR "failed to compile generated C implementation source")
  endif()
  list(APPEND c_objs "${obj}")
endforeach()

set(exe "${build_out}/cpp_c_parity_runner")
execute_process(
  COMMAND
    "${CXX_COMPILER}"
      "${main_obj}"
      ${c_objs}
      -o "${exe}"
  RESULT_VARIABLE link_result
  OUTPUT_VARIABLE link_stdout
  ERROR_VARIABLE link_stderr
)
if(NOT link_result EQUAL 0)
  message(STATUS "link stdout:\n${link_stdout}")
  message(STATUS "link stderr:\n${link_stderr}")
  message(FATAL_ERROR "failed to link C/C++ parity harness")
endif()

execute_process(
  COMMAND "${exe}" 128
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
)
if(NOT run_result EQUAL 0)
  message(STATUS "runner stdout:\n${run_stdout}")
  message(STATUS "runner stderr:\n${run_stderr}")
  message(FATAL_ERROR "C/C++ parity harness reported mismatches")
endif()

file(WRITE "${OUT_DIR}/cpp-c-parity-${CPP_PROFILE}-summary.txt" "${run_stdout}\n")
if(CPP_PROFILE STREQUAL "std")
  # Backward-compatible summary path used by existing scripts/docs.
  file(WRITE "${OUT_DIR}/cpp-c-parity-summary.txt" "${run_stdout}\n")
endif()
message(STATUS "C/C++ (${CPP_PROFILE}) parity summary:\n${run_stdout}")
