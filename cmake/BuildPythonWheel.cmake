cmake_minimum_required(VERSION 3.24)

foreach(var GENERATED_OUT_DIR WHEEL_OUT_DIR PYTHON_EXECUTABLE PY_PACKAGE)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${GENERATED_OUT_DIR}")
  message(FATAL_ERROR
    "Generated output directory not found: ${GENERATED_OUT_DIR}. "
    "Generate Python artifacts first (for example, run generate-uavcan-python).")
endif()

if(NOT EXISTS "${PYTHON_EXECUTABLE}")
  message(FATAL_ERROR "python executable not found: ${PYTHON_EXECUTABLE}")
endif()

if(NOT EXISTS "${GENERATED_OUT_DIR}/pyproject.toml")
  message(FATAL_ERROR
    "Missing pyproject.toml in generated output: ${GENERATED_OUT_DIR}. "
    "Wheel build requires generated packaging metadata.")
endif()

if(NOT DEFINED REQUIRE_ACCELERATOR OR "${REQUIRE_ACCELERATOR}" STREQUAL "")
  set(REQUIRE_ACCELERATOR OFF)
endif()

set(package_path "${PY_PACKAGE}")
string(REPLACE "." "/" package_path "${package_path}")
set(package_root "${GENERATED_OUT_DIR}/${package_path}")
if(NOT EXISTS "${package_root}")
  message(FATAL_ERROR
    "Generated package root not found: ${package_root}. "
    "Verify PY_PACKAGE matches the package used during generation.")
endif()

if(REQUIRE_ACCELERATOR)
  file(GLOB accel_candidates
    "${package_root}/_dsdl_runtime_accel*.so"
    "${package_root}/_dsdl_runtime_accel*.dylib"
    "${package_root}/_dsdl_runtime_accel*.pyd")
  list(LENGTH accel_candidates accel_count)
  if(accel_count EQUAL 0)
    message(FATAL_ERROR
      "Wheel build requires staged accelerator module, but none was found in ${package_root}. "
      "Run stage-uavcan-python-runtime-accelerator-required first.")
  endif()
endif()

file(REMOVE_RECURSE "${WHEEL_OUT_DIR}")
file(MAKE_DIRECTORY "${WHEEL_OUT_DIR}")

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -m pip --disable-pip-version-check
    wheel --no-deps --wheel-dir "${WHEEL_OUT_DIR}" "${GENERATED_OUT_DIR}"
  RESULT_VARIABLE wheel_result
  OUTPUT_VARIABLE wheel_stdout
  ERROR_VARIABLE wheel_stderr
)
if(NOT wheel_result EQUAL 0)
  message(STATUS "wheel build stdout:\n${wheel_stdout}")
  message(STATUS "wheel build stderr:\n${wheel_stderr}")
  message(FATAL_ERROR
    "Python wheel build failed. Ensure pip/setuptools/wheel are available for ${PYTHON_EXECUTABLE}.")
endif()

file(GLOB wheels "${WHEEL_OUT_DIR}/*.whl")
list(LENGTH wheels wheel_count)
if(wheel_count EQUAL 0)
  message(FATAL_ERROR "Wheel build completed but no .whl artifact was found in ${WHEEL_OUT_DIR}")
endif()

set(manifest "${WHEEL_OUT_DIR}/MANIFEST.txt")
file(WRITE "${manifest}"
  "Python wheel staging manifest\n"
  "generated-out-dir: ${GENERATED_OUT_DIR}\n"
  "package: ${PY_PACKAGE}\n"
  "python-executable: ${PYTHON_EXECUTABLE}\n"
  "require-accelerator: ${REQUIRE_ACCELERATOR}\n"
  "wheel-count: ${wheel_count}\n"
  "artifacts:\n")
foreach(wheel_path IN LISTS wheels)
  get_filename_component(wheel_name "${wheel_path}" NAME)
  file(APPEND "${manifest}" "  - ${wheel_name}\n")
endforeach()

message(STATUS "Built ${wheel_count} wheel artifact(s) into ${WHEEL_OUT_DIR}")
