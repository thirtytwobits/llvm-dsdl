cmake_minimum_required(VERSION 3.24)

foreach(var GENERATED_OUT_DIR PY_PACKAGE)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

if(NOT EXISTS "${GENERATED_OUT_DIR}")
  message(FATAL_ERROR
    "Generated output directory not found: ${GENERATED_OUT_DIR}. "
    "Generate Python artifacts first (for example, run generate-uavcan-python).")
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

set(has_accel FALSE)
if(DEFINED ACCEL_MODULE AND NOT "${ACCEL_MODULE}" STREQUAL "" AND EXISTS "${ACCEL_MODULE}")
  set(has_accel TRUE)
endif()

if(has_accel)
  file(COPY "${ACCEL_MODULE}" DESTINATION "${package_root}")
  get_filename_component(accel_name "${ACCEL_MODULE}" NAME)
  message(STATUS
    "Staged Python accelerator module '${accel_name}' into ${package_root}")
elseif(REQUIRE_ACCELERATOR)
  message(FATAL_ERROR
    "Python accelerator staging requested but no accelerator module is available. "
    "Configure with -DLLVMDSDL_ENABLE_PYTHON_ACCELERATOR=ON and build target "
    "'build-python-runtime-accelerator', then rerun staging.")
else()
  message(STATUS
    "Python accelerator module is unavailable; skipping staging. "
    "Set REQUIRE_ACCELERATOR=ON to require accel staging.")
endif()
