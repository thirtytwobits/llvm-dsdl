cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC UAVCAN_ROOT OUT_DIR)
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

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(out_a "${OUT_DIR}/run-a")
set(out_b "${OUT_DIR}/run-b")
file(MAKE_DIRECTORY "${out_a}")
file(MAKE_DIRECTORY "${out_b}")

foreach(out_dir "${out_a}" "${out_b}")
  execute_process(
    COMMAND
      "${DSDLC}" go
        --root-namespace-dir "${UAVCAN_ROOT}"
        --strict
        --out-dir "${out_dir}"
        --go-module "uavcan_dsdl_generated"
    RESULT_VARIABLE gen_result
    OUTPUT_VARIABLE gen_stdout
    ERROR_VARIABLE gen_stderr
  )
  if(NOT gen_result EQUAL 0)
    message(STATUS "dsdlc stdout:\n${gen_stdout}")
    message(STATUS "dsdlc stderr:\n${gen_stderr}")
    message(FATAL_ERROR "uavcan Go determinism generation failed for ${out_dir}")
  endif()
endforeach()

file(GLOB_RECURSE run_a_entries RELATIVE "${out_a}" "${out_a}/*")
file(GLOB_RECURSE run_b_entries RELATIVE "${out_b}" "${out_b}/*")

set(run_a_files "")
foreach(entry IN LISTS run_a_entries)
  if(NOT IS_DIRECTORY "${out_a}/${entry}")
    list(APPEND run_a_files "${entry}")
  endif()
endforeach()

set(run_b_files "")
foreach(entry IN LISTS run_b_entries)
  if(NOT IS_DIRECTORY "${out_b}/${entry}")
    list(APPEND run_b_files "${entry}")
  endif()
endforeach()

list(SORT run_a_files)
list(SORT run_b_files)

string(JOIN "\n" run_a_manifest ${run_a_files})
string(JOIN "\n" run_b_manifest ${run_b_files})
if(NOT run_a_manifest STREQUAL run_b_manifest)
  file(WRITE "${OUT_DIR}/run-a-manifest.txt" "${run_a_manifest}\n")
  file(WRITE "${OUT_DIR}/run-b-manifest.txt" "${run_b_manifest}\n")
  message(FATAL_ERROR
    "Generated file lists differ between deterministic runs. "
    "See ${OUT_DIR}/run-a-manifest.txt and ${OUT_DIR}/run-b-manifest.txt.")
endif()

foreach(rel IN LISTS run_a_files)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${out_a}/${rel}" "${out_b}/${rel}"
    RESULT_VARIABLE cmp_result
  )
  if(NOT cmp_result EQUAL 0)
    message(FATAL_ERROR "Generated file differs between runs: ${rel}")
  endif()
endforeach()

list(LENGTH run_a_files file_count)
message(STATUS "uavcan Go determinism check passed: ${file_count} files identical across two runs")
