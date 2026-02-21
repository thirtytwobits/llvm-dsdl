cmake_minimum_required(VERSION 3.24)

foreach(var DSDLC FIXTURES_ROOT OUT_DIR PYTHON_EXECUTABLE)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${DSDLC}")
  message(FATAL_ERROR "dsdlc executable not found: ${DSDLC}")
endif()
if(NOT EXISTS "${FIXTURES_ROOT}")
  message(FATAL_ERROR "fixtures root not found: ${FIXTURES_ROOT}")
endif()
if(NOT EXISTS "${PYTHON_EXECUTABLE}")
  message(FATAL_ERROR "python executable not found: ${PYTHON_EXECUTABLE}")
endif()

set(PY_PACKAGE "llvmdsdl_py_bench")
set(py_package_path "${PY_PACKAGE}")
string(REPLACE "." "/" py_package_path "${py_package_path}")
set(package_root "${OUT_DIR}/${py_package_path}")

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

execute_process(
  COMMAND "${DSDLC}" python
    --root-namespace-dir "${FIXTURES_ROOT}"
    --out-dir "${OUT_DIR}"
    --py-package "${PY_PACKAGE}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "Python runtime benchmark generation failed")
endif()

if(DEFINED ACCEL_MODULE AND NOT "${ACCEL_MODULE}" STREQUAL "" AND EXISTS "${ACCEL_MODULE}")
  file(COPY "${ACCEL_MODULE}" DESTINATION "${package_root}")
endif()

set(bench_script "${OUT_DIR}/python_runtime_bench.py")
file(WRITE
  "${bench_script}"
  "from __future__ import annotations\n"
  "\n"
  "import importlib\n"
  "import os\n"
  "import time\n"
  "\n"
  "def run_mode(mode: str) -> tuple[str, float]:\n"
  "    os.environ['LLVMDSDL_PY_RUNTIME_MODE'] = mode\n"
  "    for mod in list(importlib.sys.modules):\n"
  "        if mod.startswith('llvmdsdl_py_bench'):\n"
  "            del importlib.sys.modules[mod]\n"
  "    runtime_loader = importlib.import_module('llvmdsdl_py_bench._runtime_loader')\n"
  "    Helpers_1_0 = importlib.import_module('llvmdsdl_py_bench.fixtures.vendor.helpers_1_0').Helpers_1_0\n"
  "    value = Helpers_1_0(a=-7, b=1.5, c=[1, 2, 3, 4, 5])\n"
  "    start = time.perf_counter()\n"
  "    for _ in range(8000):\n"
  "        encoded = value.serialize()\n"
  "        _ = Helpers_1_0.deserialize(encoded)\n"
  "    elapsed = time.perf_counter() - start\n"
  "    return runtime_loader.BACKEND, elapsed\n"
  "\n"
  "pure_backend, pure_elapsed = run_mode('pure')\n"
  "print(f'pure backend={pure_backend} elapsed={pure_elapsed:.6f}s')\n"
  "\n"
  "try:\n"
  "    accel_backend, accel_elapsed = run_mode('accel')\n"
  "    print(f'accel backend={accel_backend} elapsed={accel_elapsed:.6f}s')\n"
  "except RuntimeError:\n"
  "    accel_backend, accel_elapsed = 'unavailable', 0.0\n"
  "    print('accel backend unavailable')\n"
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${OUT_DIR}"
    "${PYTHON_EXECUTABLE}" "${bench_script}"
  RESULT_VARIABLE bench_result
  OUTPUT_VARIABLE bench_stdout
  ERROR_VARIABLE bench_stderr
)
if(NOT bench_result EQUAL 0)
  message(STATUS "python bench stdout:\n${bench_stdout}")
  message(STATUS "python bench stderr:\n${bench_stderr}")
  message(FATAL_ERROR "Python runtime benchmark execution failed")
endif()

file(WRITE "${OUT_DIR}/python-runtime-bench.txt" "${bench_stdout}\n")
message(STATUS "Python runtime benchmark finished")
message(STATUS "${bench_stdout}")
