cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC OUT_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(fixture_ns_root "${OUT_DIR}/compatdemo")
set(ts_out "${OUT_DIR}/ts-compat")
file(MAKE_DIRECTORY "${fixture_ns_root}")
file(MAKE_DIRECTORY "${ts_out}")

file(WRITE
  "${fixture_ns_root}/CompatArray.1.0.dsdl"
  "uint8[0] fixed_bad\n"
  "uint8[<=0] var_inc_bad\n"
  "uint8[<1] var_exc_bad\n"
  "@sealed\n"
)

execute_process(
  COMMAND
    "${DSDLC}" ts
      --root-namespace-dir "${fixture_ns_root}"
      --strict
      --out-dir "${OUT_DIR}/ts-strict"
      --ts-module "ts_compat_generation_strict"
  RESULT_VARIABLE strict_result
  OUTPUT_VARIABLE strict_stdout
  ERROR_VARIABLE strict_stderr
)
if(strict_result EQUAL 0)
  message(FATAL_ERROR "strict generation unexpectedly succeeded for compat-dependent fixture")
endif()
if(NOT strict_stderr MATCHES "fixed-length array capacity must be positive")
  message(FATAL_ERROR "strict diagnostics missing fixed-array capacity failure")
endif()
if(NOT strict_stderr MATCHES "inclusive variable-length array bound must be >= 1")
  message(FATAL_ERROR "strict diagnostics missing inclusive-bound failure")
endif()
if(NOT strict_stderr MATCHES "exclusive variable-length array bound must be > 1")
  message(FATAL_ERROR "strict diagnostics missing exclusive-bound failure")
endif()

execute_process(
  COMMAND
    "${DSDLC}" ts
      --root-namespace-dir "${fixture_ns_root}"
      --compat-mode
      --out-dir "${ts_out}"
      --ts-module "ts_compat_generation"
  RESULT_VARIABLE compat_result
  OUTPUT_VARIABLE compat_stdout
  ERROR_VARIABLE compat_stderr
)
if(NOT compat_result EQUAL 0)
  message(STATUS "compat stdout:\n${compat_stdout}")
  message(STATUS "compat stderr:\n${compat_stderr}")
  message(FATAL_ERROR "compat generation failed for compat-dependent fixture")
endif()
if(NOT compat_stderr MATCHES "compat mode: fixed array capacity clamped to 1")
  message(FATAL_ERROR "compat diagnostics missing fixed-array clamp warning")
endif()
if(NOT compat_stderr MATCHES "compat mode: inclusive bound clamped to 1")
  message(FATAL_ERROR "compat diagnostics missing inclusive-bound clamp warning")
endif()
if(NOT compat_stderr MATCHES "compat mode: exclusive bound clamped to 2")
  message(FATAL_ERROR "compat diagnostics missing exclusive-bound clamp warning")
endif()

set(type_file "${ts_out}/compatdemo/compat_array_1_0.ts")
if(NOT EXISTS "${type_file}")
  message(FATAL_ERROR "missing compat-generated type module: ${type_file}")
endif()
file(READ "${type_file}" type_text)
if(type_text MATCHES "TypeScript runtime path is not yet available for this DSDL type")
  message(FATAL_ERROR "compat-generated type module unexpectedly contains runtime unsupported stub")
endif()
if(NOT type_text MATCHES "field 'fixed_bad' expects exactly 1 elements")
  message(FATAL_ERROR "compat-generated type module missing fixed-array clamp behavior")
endif()
if(NOT type_text MATCHES "field 'var_inc_bad' exceeds max length 1")
  message(FATAL_ERROR "compat-generated type module missing inclusive-array clamp behavior")
endif()
if(NOT type_text MATCHES "field 'var_exc_bad' exceeds max length 1")
  message(FATAL_ERROR "compat-generated type module missing exclusive-array clamp behavior")
endif()

message(STATUS "TypeScript compat generation gate passed")
