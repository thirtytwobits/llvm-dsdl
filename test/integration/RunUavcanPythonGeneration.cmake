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

if(NOT DEFINED PY_PACKAGE OR "${PY_PACKAGE}" STREQUAL "")
  set(PY_PACKAGE "uavcan_dsdl_generated_py")
endif()
set(py_package_path "${PY_PACKAGE}")
string(REPLACE "." "/" py_package_path "${py_package_path}")
set(package_root "${OUT_DIR}/${py_package_path}")

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

execute_process(
  COMMAND "${DSDLC}" python
    --root-namespace-dir "${UAVCAN_ROOT}"
    --out-dir "${OUT_DIR}"
    --py-package "${PY_PACKAGE}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_stdout
  ERROR_VARIABLE gen_stderr
)
if(NOT gen_result EQUAL 0)
  message(STATUS "dsdlc stdout:\n${gen_stdout}")
  message(STATUS "dsdlc stderr:\n${gen_stderr}")
  message(FATAL_ERROR "uavcan Python generation failed")
endif()

foreach(required
    "${package_root}/__init__.py"
    "${package_root}/_dsdl_runtime.py"
    "${package_root}/_runtime_loader.py")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing required generated file: ${required}")
  endif()
endforeach()

file(GLOB_RECURSE dsdl_files "${UAVCAN_ROOT}/*.dsdl")
list(LENGTH dsdl_files dsdl_count)

file(GLOB_RECURSE py_files "${package_root}/*.py")
set(type_py_files "")
foreach(py IN LISTS py_files)
  get_filename_component(name "${py}" NAME)
  if(NOT name STREQUAL "__init__.py" AND
     NOT name STREQUAL "_dsdl_runtime.py" AND
     NOT name STREQUAL "_runtime_loader.py")
    list(APPEND type_py_files "${py}")
  endif()
endforeach()
list(LENGTH type_py_files type_py_count)

if(NOT dsdl_count EQUAL type_py_count)
  message(FATAL_ERROR
    "Python type file count mismatch: dsdl=${dsdl_count}, generated=${type_py_count}")
endif()

set(found_dataclass FALSE)
set(found_serialize FALSE)
set(found_deserialize FALSE)
set(found_metadata_constant FALSE)
foreach(py IN LISTS type_py_files)
  file(READ "${py}" text)
  if(text MATCHES "@dataclass\\(slots=True\\)")
    set(found_dataclass TRUE)
  endif()
  if(text MATCHES "def serialize\\(self\\) -> bytes")
    set(found_serialize TRUE)
  endif()
  if(text MATCHES "def deserialize\\(cls, data: bytes \\| bytearray \\| memoryview\\)")
    set(found_deserialize TRUE)
  endif()
  if(text MATCHES "DSDL_FULL_NAME = ")
    set(found_metadata_constant TRUE)
  endif()
endforeach()

if(NOT found_dataclass)
  message(FATAL_ERROR "Generated Python files are missing dataclass declarations")
endif()
if(NOT found_serialize)
  message(FATAL_ERROR "Generated Python files are missing serialize() methods")
endif()
if(NOT found_deserialize)
  message(FATAL_ERROR "Generated Python files are missing deserialize() methods")
endif()
if(NOT found_metadata_constant)
  message(FATAL_ERROR "Generated Python files are missing DSDL metadata constants")
endif()

message(STATUS
  "uavcan Python generation check passed: ${dsdl_count} DSDL -> ${type_py_count} Python type files")
