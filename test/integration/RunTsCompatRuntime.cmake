cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC OUT_DIR TSC_EXECUTABLE NODE_EXECUTABLE)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()
if(NOT EXISTS "${TSC_EXECUTABLE}")
  message(FATAL_ERROR "tsc executable not found: ${TSC_EXECUTABLE}")
endif()
if(NOT EXISTS "${NODE_EXECUTABLE}")
  message(FATAL_ERROR "node executable not found: ${NODE_EXECUTABLE}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

set(fixture_ns_root "${OUT_DIR}/compatdemo")
set(ts_out "${OUT_DIR}/ts")
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
      --compat-mode
      --out-dir "${ts_out}"
      --ts-module "ts_compat_runtime_smoke"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "TypeScript compat generation failed")
endif()

set(ts_index "${ts_out}/index.ts")
if(NOT EXISTS "${ts_index}")
  message(FATAL_ERROR "generated TypeScript index missing: ${ts_index}")
endif()
file(READ "${ts_index}" ts_index_content)
string(REGEX MATCH
  "export \\* as ([A-Za-z0-9_]+) from \"\\./[^\"]*compat_array_1_0\";"
  ts_type_export
  "${ts_index_content}")
if("${CMAKE_MATCH_1}" STREQUAL "")
  message(FATAL_ERROR "failed to locate compat_array_1_0 export alias in ${ts_index}")
endif()
set(ts_type_module "${CMAKE_MATCH_1}")

file(WRITE
  "${ts_out}/runtime_compat_smoke.ts"
  "import { ${ts_type_module} } from \"./index\";\n"
  "\n"
  "const inObj: ${ts_type_module}.CompatArray_1_0 = {\n"
  "  fixed_bad: [7],\n"
  "  var_inc_bad: [9],\n"
  "  var_exc_bad: [5],\n"
  "};\n"
  "const outBytes = ${ts_type_module}.serializeCompatArray_1_0(inObj);\n"
  "console.log(Array.from(outBytes).join(\" \"));\n"
  "const decoded = ${ts_type_module}.deserializeCompatArray_1_0(outBytes);\n"
  "console.log(\n"
  "  String(decoded.value.fixed_bad[0]) + \" \" +\n"
  "  String(decoded.value.var_inc_bad.length) + \" \" +\n"
  "  String(decoded.value.var_inc_bad[0]) + \" \" +\n"
  "  String(decoded.value.var_exc_bad.length) + \" \" +\n"
  "  String(decoded.value.var_exc_bad[0]) + \" \" +\n"
  "  String(decoded.consumed)\n"
  ");\n"
  "let rejected = 0;\n"
  "try {\n"
  "  ${ts_type_module}.serializeCompatArray_1_0({ fixed_bad: [1, 2], var_inc_bad: [9], var_exc_bad: [5] });\n"
  "} catch {\n"
  "  rejected += 1;\n"
  "}\n"
  "try {\n"
  "  ${ts_type_module}.serializeCompatArray_1_0({ fixed_bad: [1], var_inc_bad: [2, 3], var_exc_bad: [5] });\n"
  "} catch {\n"
  "  rejected += 1;\n"
  "}\n"
  "try {\n"
  "  ${ts_type_module}.serializeCompatArray_1_0({ fixed_bad: [1], var_inc_bad: [2], var_exc_bad: [5, 6] });\n"
  "} catch {\n"
  "  rejected += 1;\n"
  "}\n"
  "console.log(\"compat_rejections \" + String(rejected));\n"
  "if (rejected !== 3) {\n"
  "  throw new Error(\"expected 3 compat rejection checks, got \" + String(rejected));\n"
  "}\n"
)

file(WRITE
  "${ts_out}/tsconfig-runtime-compat-smoke.json"
  "{\n"
  "  \"compilerOptions\": {\n"
  "    \"target\": \"ES2022\",\n"
  "    \"module\": \"CommonJS\",\n"
  "    \"moduleResolution\": \"Node\",\n"
  "    \"strict\": true,\n"
  "    \"skipLibCheck\": true,\n"
  "    \"outDir\": \"./js\"\n"
  "  },\n"
  "  \"include\": [\"./**/*.ts\"]\n"
  "}\n"
)

execute_process(
  COMMAND "${TSC_EXECUTABLE}" -p "${ts_out}/tsconfig-runtime-compat-smoke.json" --pretty false
  WORKING_DIRECTORY "${ts_out}"
  RESULT_VARIABLE tsc_result
  OUTPUT_VARIABLE tsc_stdout
  ERROR_VARIABLE tsc_stderr
)
if(NOT tsc_result EQUAL 0)
  message(STATUS "tsc stdout:\n${tsc_stdout}")
  message(STATUS "tsc stderr:\n${tsc_stderr}")
  message(FATAL_ERROR "failed to compile TypeScript compat runtime smoke")
endif()

file(WRITE "${ts_out}/js/package.json" "{\n  \"type\": \"commonjs\"\n}\n")

execute_process(
  COMMAND "${NODE_EXECUTABLE}" "${ts_out}/js/runtime_compat_smoke.js"
  RESULT_VARIABLE node_result
  OUTPUT_VARIABLE node_stdout
  ERROR_VARIABLE node_stderr
)
if(NOT node_result EQUAL 0)
  message(STATUS "node stdout:\n${node_stdout}")
  message(STATUS "node stderr:\n${node_stderr}")
  message(FATAL_ERROR "TypeScript compat runtime smoke execution failed")
endif()

string(STRIP "${node_stdout}" actual_output)
set(expected_output "7 1 9 1 5\n7 1 9 1 5 5\ncompat_rejections 3")
if(NOT actual_output STREQUAL expected_output)
  file(WRITE "${OUT_DIR}/expected-output.txt" "${expected_output}\n")
  file(WRITE "${OUT_DIR}/actual-output.txt" "${actual_output}\n")
  message(FATAL_ERROR
    "unexpected TypeScript compat runtime output. See ${OUT_DIR}/expected-output.txt and ${OUT_DIR}/actual-output.txt.")
endif()

message(STATUS "TypeScript compat runtime smoke passed")
