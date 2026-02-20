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

execute_process(
  COMMAND
    "${DSDLC}" ts
      --root-namespace-dir "${UAVCAN_ROOT}"
      --out-dir "${OUT_DIR}"
      --ts-module "uavcan_dsdl_generated_ts"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "uavcan TypeScript generation failed")
endif()

set(index_file "${OUT_DIR}/index.ts")
if(NOT EXISTS "${index_file}")
  message(FATAL_ERROR "Missing generated root index file: ${index_file}")
endif()

file(READ "${index_file}" index_text)
if(index_text MATCHES "export \\* from ")
  message(FATAL_ERROR "Unexpected wildcard root re-export in index.ts")
endif()

file(STRINGS
  "${index_file}"
  index_export_lines
  REGEX "^export \\* as [A-Za-z_][A-Za-z0-9_]* from \"\\./.+\";$")

if(index_export_lines STREQUAL "")
  message(FATAL_ERROR "Expected namespace alias export lines in index.ts")
endif()

file(GLOB_RECURSE generated_ts RELATIVE "${OUT_DIR}" "${OUT_DIR}/*.ts")
set(type_ts_files "")
foreach(ts_rel IN LISTS generated_ts)
  if(NOT ts_rel STREQUAL "index.ts" AND NOT ts_rel STREQUAL "dsdl_runtime.ts")
    list(APPEND type_ts_files "${ts_rel}")
  endif()
endforeach()
list(LENGTH type_ts_files type_ts_count)
list(LENGTH index_export_lines index_export_count)

if(NOT type_ts_count EQUAL index_export_count)
  message(FATAL_ERROR
    "index.ts export count mismatch: index=${index_export_count}, type_ts=${type_ts_count}")
endif()

set(index_aliases "")
set(index_modules "")
foreach(line IN LISTS index_export_lines)
  string(REGEX REPLACE
    "^export \\* as ([A-Za-z_][A-Za-z0-9_]*) from \"\\./([^\"]+)\";$"
    "\\1"
    alias
    "${line}")
  string(REGEX REPLACE
    "^export \\* as ([A-Za-z_][A-Za-z0-9_]*) from \"\\./([^\"]+)\";$"
    "\\2"
    module_rel
    "${line}")

  if(alias STREQUAL line OR module_rel STREQUAL line)
    message(FATAL_ERROR "Failed to parse index.ts export line: ${line}")
  endif()

  set(module_file "${OUT_DIR}/${module_rel}.ts")
  if(NOT EXISTS "${module_file}")
    message(FATAL_ERROR
      "index.ts exports missing module target: ${module_rel} (expected ${module_file})")
  endif()

  list(APPEND index_aliases "${alias}")
  list(APPEND index_modules "${module_rel}")
endforeach()

set(index_aliases_unique ${index_aliases})
list(REMOVE_DUPLICATES index_aliases_unique)
list(LENGTH index_aliases_unique index_aliases_unique_count)
if(NOT index_aliases_unique_count EQUAL index_export_count)
  message(FATAL_ERROR "Duplicate namespace aliases detected in index.ts")
endif()

set(index_modules_unique ${index_modules})
list(REMOVE_DUPLICATES index_modules_unique)
list(LENGTH index_modules_unique index_modules_unique_count)
if(NOT index_modules_unique_count EQUAL index_export_count)
  message(FATAL_ERROR "Duplicate module exports detected in index.ts")
endif()

set(expected_modules "")
foreach(ts_rel IN LISTS type_ts_files)
  string(REGEX REPLACE "\\.ts$" "" module_rel "${ts_rel}")
  list(APPEND expected_modules "${module_rel}")
endforeach()
list(SORT expected_modules)
list(SORT index_modules)
string(JOIN "\n" expected_manifest ${expected_modules})
string(JOIN "\n" index_manifest ${index_modules})
if(NOT expected_manifest STREQUAL index_manifest)
  file(WRITE "${OUT_DIR}/expected-index-modules.txt" "${expected_manifest}\n")
  file(WRITE "${OUT_DIR}/actual-index-modules.txt" "${index_manifest}\n")
  message(FATAL_ERROR
    "index.ts module inventory does not match generated type files. "
    "See ${OUT_DIR}/expected-index-modules.txt and ${OUT_DIR}/actual-index-modules.txt.")
endif()

message(STATUS
  "uavcan TypeScript index contract check passed: ${index_export_count} alias exports covering all type modules")
