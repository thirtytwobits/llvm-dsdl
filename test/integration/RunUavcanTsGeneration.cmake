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

if(NOT DEFINED TS_RUNTIME_SPECIALIZATION OR "${TS_RUNTIME_SPECIALIZATION}" STREQUAL "")
  set(TS_RUNTIME_SPECIALIZATION "portable")
endif()
if(NOT "${TS_RUNTIME_SPECIALIZATION}" STREQUAL "portable" AND
   NOT "${TS_RUNTIME_SPECIALIZATION}" STREQUAL "fast")
  message(FATAL_ERROR "Invalid TS_RUNTIME_SPECIALIZATION value: ${TS_RUNTIME_SPECIALIZATION}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

execute_process(
  COMMAND
    "${DSDLC}" --target-language ts
      "${UAVCAN_ROOT}"
      --outdir "${OUT_DIR}"
      --ts-module "uavcan_dsdl_generated_ts"
      --ts-runtime-specialization "${TS_RUNTIME_SPECIALIZATION}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "uavcan TypeScript generation failed")
endif()

foreach(required
    "${OUT_DIR}/package.json"
    "${OUT_DIR}/index.ts"
    "${OUT_DIR}/dsdl_runtime.ts")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing required generated file: ${required}")
  endif()
endforeach()

file(READ "${OUT_DIR}/index.ts" ts_index)
if(NOT ts_index MATCHES "export \\* as ")
  message(FATAL_ERROR "Expected index.ts to export collision-safe namespace aliases")
endif()
if(ts_index MATCHES "export \\* from ")
  message(FATAL_ERROR "Unexpected wildcard re-export in index.ts; collisions may occur")
endif()

file(READ "${OUT_DIR}/package.json" package_json)
if(NOT package_json MATCHES "\"name\"[ \t\r\n]*:[ \t\r\n]*\"uavcan_dsdl_generated_ts\"")
  message(FATAL_ERROR "Expected TypeScript package.json name to match requested module")
endif()
if(NOT package_json MATCHES
      "\"tsRuntimeSpecialization\"[ \t\r\n]*:[ \t\r\n]*\"${TS_RUNTIME_SPECIALIZATION}\"")
  message(FATAL_ERROR
    "Expected TypeScript package.json runtime specialization metadata to match requested profile")
endif()

file(GLOB_RECURSE dsdl_files "${UAVCAN_ROOT}/*.dsdl")
list(LENGTH dsdl_files dsdl_count)

file(GLOB_RECURSE ts_files "${OUT_DIR}/*.ts")
set(type_ts_files "")
foreach(ts IN LISTS ts_files)
  get_filename_component(name "${ts}" NAME)
  if(NOT name STREQUAL "index.ts" AND NOT name STREQUAL "dsdl_runtime.ts")
    list(APPEND type_ts_files "${ts}")
  endif()
endforeach()
list(LENGTH type_ts_files type_ts_count)

if(NOT dsdl_count EQUAL type_ts_count)
  message(FATAL_ERROR
    "TypeScript type file count mismatch: dsdl=${dsdl_count}, generated=${type_ts_count}")
endif()

set(found_type_declaration FALSE)
set(found_metadata_constant FALSE)
set(found_service_alias FALSE)
set(found_runtime_serialize_fn FALSE)
set(found_runtime_deserialize_fn FALSE)
set(found_runtime_unsupported_stub FALSE)
foreach(ts IN LISTS type_ts_files)
  file(READ "${ts}" ts_text)
  if(ts_text MATCHES "export (interface|type) ")
    set(found_type_declaration TRUE)
  endif()
  if(ts_text MATCHES "export const DSDL_FULL_NAME = ")
    set(found_metadata_constant TRUE)
  endif()
  if(ts_text MATCHES "_Request" AND ts_text MATCHES "_Response")
    set(found_service_alias TRUE)
  endif()
  if(ts_text MATCHES "export function serialize[A-Za-z0-9_]+\\(")
    set(found_runtime_serialize_fn TRUE)
  endif()
  if(ts_text MATCHES "export function deserialize[A-Za-z0-9_]+\\(")
    set(found_runtime_deserialize_fn TRUE)
  endif()
  if(ts_text MATCHES "TypeScript runtime path is not yet available for this DSDL type")
    set(found_runtime_unsupported_stub TRUE)
  endif()
endforeach()

if(NOT found_type_declaration)
  message(FATAL_ERROR "Generated TypeScript files are missing exported type declarations")
endif()
if(NOT found_metadata_constant)
  message(FATAL_ERROR "Generated TypeScript files are missing DSDL metadata constants")
endif()
if(NOT found_service_alias)
  message(FATAL_ERROR "Generated TypeScript files are missing service request/response type aliases")
endif()
if(NOT found_runtime_serialize_fn)
  message(FATAL_ERROR "Generated TypeScript files are missing runtime serialize function exports")
endif()
if(NOT found_runtime_deserialize_fn)
  message(FATAL_ERROR "Generated TypeScript files are missing runtime deserialize function exports")
endif()
if(found_runtime_unsupported_stub)
  message(FATAL_ERROR
    "Generated TypeScript files still contain unsupported runtime stubs; expected runtime-backed serializers/deserializers for UAVCAN corpus")
endif()

message(STATUS
  "uavcan TypeScript generation check passed: ${dsdl_count} DSDL -> ${type_ts_count} TypeScript type files")
