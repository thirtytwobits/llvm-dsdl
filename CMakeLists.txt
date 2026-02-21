cmake_minimum_required(VERSION 3.24)
project(llvm-dsdl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Some distro LLVM packages export LLVMSupport with zstd::libzstd_* targets but
# do not bring zstd in transitively. Preload zstd to avoid unresolved imported
# target references during find_package(LLVM).
find_package(zstd CONFIG QUIET)
if(TARGET ZSTD::ZSTD)
  if(NOT TARGET zstd::libzstd_shared)
    add_library(zstd::libzstd_shared INTERFACE IMPORTED)
    target_link_libraries(zstd::libzstd_shared INTERFACE ZSTD::ZSTD)
  endif()
  if(NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static INTERFACE IMPORTED)
    target_link_libraries(zstd::libzstd_static INTERFACE ZSTD::ZSTD)
  endif()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

if(APPLE AND LLVM_LINKER_SUPPORTS_NO_WARN_DUPLICATE_LIBRARIES)
  add_link_options("-Wl,-no_warn_duplicate_libraries")
endif()

set(LLVMDSDL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVMDSDL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_program(LLVMDSDL_CLANG_FORMAT_EXE
  NAMES clang-format clang-format-21 clang-format-20 clang-format-19
        clang-format-18 clang-format-17 clang-format-16 clang-format-15
)

find_program(LLVMDSDL_CLANG_TIDY_EXE
  NAMES clang-tidy clang-tidy-21 clang-tidy-20 clang-tidy-19
        clang-tidy-18 clang-tidy-17 clang-tidy-16 clang-tidy-15
)

find_program(LLVMDSDL_IWYU_EXE
  NAMES include-what-you-use iwyu include-what-you-use-21
        include-what-you-use-20 include-what-you-use-19
        include-what-you-use-18 include-what-you-use-17
        include-what-you-use-16 include-what-you-use-15
)

option(LLVMDSDL_ENABLE_LLVM_COVERAGE
       "Enable LLVM source-based coverage instrumentation and reporting targets"
       OFF)

option(LLVMDSDL_ENABLE_PYTHON_ACCELERATOR
       "Build optional CPython runtime accelerator module for generated Python code"
       OFF)

set(LLVMDSDL_COVERAGE_CTEST_ARGS
    "--output-on-failure"
    CACHE STRING
    "Additional arguments forwarded to ctest by coverage-run (space-separated).")

if(LLVMDSDL_ENABLE_LLVM_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    message(FATAL_ERROR
      "LLVM source coverage requires Clang/AppleClang. "
      "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
  add_link_options(-fprofile-instr-generate -fcoverage-mapping)

  find_program(LLVMDSDL_LLVM_PROFDATA_EXE
    NAMES llvm-profdata llvm-profdata-21 llvm-profdata-20 llvm-profdata-19
          llvm-profdata-18 llvm-profdata-17 llvm-profdata-16
    HINTS ${LLVM_TOOLS_BINARY_DIR} /opt/homebrew/opt/llvm/bin
  )

  find_program(LLVMDSDL_LLVM_COV_EXE
    NAMES llvm-cov llvm-cov-21 llvm-cov-20 llvm-cov-19
          llvm-cov-18 llvm-cov-17 llvm-cov-16
    HINTS ${LLVM_TOOLS_BINARY_DIR} /opt/homebrew/opt/llvm/bin
  )

  if(NOT LLVMDSDL_LLVM_PROFDATA_EXE OR NOT LLVMDSDL_LLVM_COV_EXE)
    message(FATAL_ERROR
      "LLVM source coverage is enabled but llvm-profdata/llvm-cov could not be found. "
      "Set LLVM_TOOLS_BINARY_DIR or ensure tools are on PATH.")
  endif()

  message(STATUS "LLVM source coverage enabled")
  message(STATUS "Using llvm-profdata: ${LLVMDSDL_LLVM_PROFDATA_EXE}")
  message(STATUS "Using llvm-cov: ${LLVMDSDL_LLVM_COV_EXE}")
endif()

if(NOT LLVMDSDL_CLANG_FORMAT_EXE)
  message(WARNING
    "clang-format not found; check-format/format-source targets will fail "
    "until it is installed.")
endif()

if(NOT LLVMDSDL_CLANG_TIDY_EXE)
  message(WARNING
    "clang-tidy not found; check-clang-tidy target will fail "
    "until it is installed.")
endif()

if(NOT LLVMDSDL_IWYU_EXE)
  message(WARNING
    "include-what-you-use not found; check-iwyu target will fail "
    "until it is installed.")
endif()

add_custom_target(check-clang-format
  COMMAND
    ${CMAKE_COMMAND}
      -DCLANG_FORMAT=${LLVMDSDL_CLANG_FORMAT_EXE}
      -DLLVMDSDL_SOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
      -P ${LLVMDSDL_SOURCE_DIR}/cmake/CheckClangFormat.cmake
  USES_TERMINAL
  VERBATIM
  COMMENT "Checking source formatting with clang-format"
)

add_custom_target(check-format
  DEPENDS check-clang-format
)

add_custom_target(format-source
  COMMAND
    ${CMAKE_COMMAND}
      -DCLANG_FORMAT=${LLVMDSDL_CLANG_FORMAT_EXE}
      -DLLVMDSDL_SOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
      -P ${LLVMDSDL_SOURCE_DIR}/cmake/RunClangFormat.cmake
  USES_TERMINAL
  VERBATIM
  COMMENT "Reformatting project source with clang-format"
)

add_custom_target(check-clang-tidy
  COMMAND
    ${CMAKE_COMMAND}
      -DCLANG_TIDY=${LLVMDSDL_CLANG_TIDY_EXE}
      -DLLVMDSDL_SOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
      -DLLVMDSDL_BINARY_DIR=${LLVMDSDL_BINARY_DIR}
      -P ${LLVMDSDL_SOURCE_DIR}/cmake/CheckClangTidy.cmake
  USES_TERMINAL
  VERBATIM
  COMMENT "Checking source diagnostics with clang-tidy"
)

add_custom_target(check-iwyu
  COMMAND
    ${CMAKE_COMMAND}
      -DINCLUDE_WHAT_YOU_USE=${LLVMDSDL_IWYU_EXE}
      -DLLVMDSDL_SOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
      -DLLVMDSDL_BINARY_DIR=${LLVMDSDL_BINARY_DIR}
      -P ${LLVMDSDL_SOURCE_DIR}/cmake/CheckIncludeWhatYouUse.cmake
  USES_TERMINAL
  VERBATIM
  COMMENT "Checking source includes with include-what-you-use"
)

add_custom_target(check-include-what-you-use
  DEPENDS check-iwyu
)

add_subdirectory(include/llvmdsdl)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(examples/cyphal_ping_pong)
add_subdirectory(examples/cyphal_yakut_register_demo)

if(LLVMDSDL_ENABLE_PYTHON_ACCELERATOR)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

  add_library(llvmdsdl-python-runtime-accelerator MODULE
    runtime/python_accel/dsdl_runtime_accel.c
  )
  target_include_directories(llvmdsdl-python-runtime-accelerator
    PRIVATE
      ${LLVMDSDL_SOURCE_DIR}/runtime
  )
  target_link_libraries(llvmdsdl-python-runtime-accelerator
    PRIVATE
      Python3::Module
  )
  set_target_properties(llvmdsdl-python-runtime-accelerator
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "_dsdl_runtime_accel"
      LIBRARY_OUTPUT_DIRECTORY "${LLVMDSDL_BINARY_DIR}/python-accel"
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${LLVMDSDL_BINARY_DIR}/python-accel/Debug"
      LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LLVMDSDL_BINARY_DIR}/python-accel/RelWithDebInfo"
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LLVMDSDL_BINARY_DIR}/python-accel/Release"
      LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${LLVMDSDL_BINARY_DIR}/python-accel/MinSizeRel"
  )

  add_custom_target(build-python-runtime-accelerator
    DEPENDS llvmdsdl-python-runtime-accelerator
  )
endif()

set(LLVMDSDL_SELF_CONTAINED_TOOL_BUNDLE_DEFAULT OFF)
if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LLVMDSDL_SELF_CONTAINED_TOOL_BUNDLE_DEFAULT ON)
endif()

option(LLVMDSDL_ENABLE_SELF_CONTAINED_TOOL_BUNDLE
       "Enable self-contained tool bundle target (copies dsdlc/dsdl-opt with runtime deps)"
       ${LLVMDSDL_SELF_CONTAINED_TOOL_BUNDLE_DEFAULT})

if(LLVMDSDL_ENABLE_SELF_CONTAINED_TOOL_BUNDLE)
  set(LLVMDSDL_SELF_CONTAINED_TOOLS_DIR
      "${LLVMDSDL_BINARY_DIR}/self-contained-tools")
  add_custom_target(bundle-tools-self-contained
    COMMAND
      ${CMAKE_COMMAND}
        -DTOOL_DSDLC=$<TARGET_FILE:dsdlc>
        -DTOOL_DSDLOPT=$<TARGET_FILE:dsdl-opt>
        -DOUTPUT_DIR=${LLVMDSDL_SELF_CONTAINED_TOOLS_DIR}
        -DLLVMDSDL_SOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
        -P ${LLVMDSDL_SOURCE_DIR}/cmake/BundleSelfContainedTools.cmake
    DEPENDS dsdlc dsdl-opt
    USES_TERMINAL
    VERBATIM
    COMMENT
      "Bundling self-contained tools into ${LLVMDSDL_SELF_CONTAINED_TOOLS_DIR}"
  )
endif()

set(LLVMDSDL_UAVCAN_ROOT "")
if(EXISTS "${LLVMDSDL_SOURCE_DIR}/public_regulated_dsdl/uavcan")
  set(LLVMDSDL_UAVCAN_ROOT
      "${LLVMDSDL_SOURCE_DIR}/public_regulated_dsdl/uavcan")
elseif(EXISTS "${LLVMDSDL_SOURCE_DIR}/submodules/public_regulated_data_types/uavcan")
  set(LLVMDSDL_UAVCAN_ROOT
      "${LLVMDSDL_SOURCE_DIR}/submodules/public_regulated_data_types/uavcan")
endif()

if(LLVMDSDL_UAVCAN_ROOT)
  set(LLVMDSDL_GENERATED_UAVCAN_DIR
      "${LLVMDSDL_BINARY_DIR}/generated/uavcan")

  function(llvmdsdl_add_uavcan_codegen_target target_name out_subdir)
    set(out_dir "${LLVMDSDL_GENERATED_UAVCAN_DIR}/${out_subdir}")
    add_custom_target(${target_name}
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${out_dir}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
      COMMAND
        $<TARGET_FILE:dsdlc>
          ${ARGN}
          --root-namespace-dir "${LLVMDSDL_UAVCAN_ROOT}"
          --out-dir "${out_dir}"
      DEPENDS dsdlc
      USES_TERMINAL
      VERBATIM
      COMMENT "Generating uavcan DSDL artifacts: ${target_name}"
    )
  endfunction()

  function(llvmdsdl_add_uavcan_c_impl_target target_name out_subdir)
    set(out_dir "${LLVMDSDL_GENERATED_UAVCAN_DIR}/${out_subdir}")
    add_custom_target(${target_name}
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${out_dir}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
      COMMAND
        $<TARGET_FILE:dsdlc>
          c
          ${ARGN}
          --root-namespace-dir "${LLVMDSDL_UAVCAN_ROOT}"
          --out-dir "${out_dir}"
      COMMAND
        ${CMAKE_COMMAND}
          -DC_COMPILER=${CMAKE_C_COMPILER}
          -DSOURCE_ROOT=${out_dir}
          -DINCLUDE_ROOT=${out_dir}
          -P "${LLVMDSDL_SOURCE_DIR}/cmake/CompileGeneratedCSources.cmake"
      DEPENDS dsdlc
      USES_TERMINAL
      VERBATIM
      COMMENT "Generating and compiling uavcan C impl translation units: ${target_name}"
    )
  endfunction()

  llvmdsdl_add_uavcan_c_impl_target(generate-uavcan-c
                                    c)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-cpp-std
                                     cpp-std
                                     cpp --cpp-profile std)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-cpp-pmr
                                     cpp-pmr
                                     cpp --cpp-profile pmr)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-cpp-both
                                     cpp-both
                                     cpp --cpp-profile both)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-rust-std
                                     rust-std
                                     rust --rust-profile std --rust-crate-name
                                     uavcan_dsdl_generated)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-rust-no-std-alloc
                                     rust-no-std-alloc
                                     rust --rust-profile no-std-alloc --rust-crate-name
                                     uavcan_dsdl_generated_no_std)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-rust-runtime-fast
                                     rust-runtime-fast
                                     rust --rust-runtime-specialization fast --rust-profile std --rust-crate-name
                                     uavcan_dsdl_generated_fast)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-rust-no-std-runtime-fast
                                     rust-no-std-runtime-fast
                                     rust --rust-profile no-std-alloc --rust-runtime-specialization fast --rust-crate-name
                                     uavcan_dsdl_generated_no_std_fast)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-go
                                     go
                                     go --go-module uavcan_dsdl_generated)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-ts
                                     ts
                                     ts --ts-module uavcan_dsdl_generated_ts)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-ts-runtime-fast
                                     ts-runtime-fast
                                     ts --ts-module uavcan_dsdl_generated_ts_fast --ts-runtime-specialization fast)
  llvmdsdl_add_uavcan_codegen_target(generate-uavcan-python
                                     python
                                     python --py-package uavcan_dsdl_generated_py)

  add_custom_target(generate-uavcan-all)
  add_dependencies(generate-uavcan-all
                   generate-uavcan-c
                   generate-uavcan-cpp-both
                   generate-uavcan-cpp-std
                   generate-uavcan-cpp-pmr
                   generate-uavcan-rust-std
                   generate-uavcan-rust-no-std-alloc
                   generate-uavcan-rust-runtime-fast
                   generate-uavcan-rust-no-std-runtime-fast
                   generate-uavcan-go
                   generate-uavcan-ts
                   generate-uavcan-ts-runtime-fast
                   generate-uavcan-python)

  add_custom_target(generate-demo-2026-02-16
    COMMAND
      ${CMAKE_COMMAND}
        -DDSDLC=$<TARGET_FILE:dsdlc>
        -DDSDLOPT=$<TARGET_FILE:dsdl-opt>
        -DUAVCAN_ROOT=${LLVMDSDL_UAVCAN_ROOT}
        -DOUT_DIR=${LLVMDSDL_BINARY_DIR}/demo-2026-02-16
        -DBINARY_DIR=${LLVMDSDL_BINARY_DIR}
        -DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}
        -DC_COMPILER=${CMAKE_C_COMPILER}
        -DCOMPILE_C_SCRIPT=${LLVMDSDL_SOURCE_DIR}/cmake/CompileGeneratedCSources.cmake
        -DSOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
        -P ${LLVMDSDL_SOURCE_DIR}/cmake/GenerateDemoArtifacts.cmake
    DEPENDS dsdlc dsdl-opt
    USES_TERMINAL
    VERBATIM
    COMMENT "Generating demo artifact bundle for 2026-02-16"
  )
else()
  message(WARNING
    "uavcan DSDL root not found. Expected either "
    "${LLVMDSDL_SOURCE_DIR}/public_regulated_dsdl/uavcan or "
    "${LLVMDSDL_SOURCE_DIR}/submodules/public_regulated_data_types/uavcan. "
    "Skipping generation targets.")
endif()

enable_testing()
add_subdirectory(test)

if(LLVMDSDL_ENABLE_LLVM_COVERAGE)
  separate_arguments(LLVMDSDL_COVERAGE_CTEST_ARGS_LIST
                     NATIVE_COMMAND
                     "${LLVMDSDL_COVERAGE_CTEST_ARGS}")

  set(LLVMDSDL_COVERAGE_CONFIG_SUBDIR "")
  set(LLVMDSDL_COVERAGE_CTEST_CONFIG_ARGS "")
  if(CMAKE_CONFIGURATION_TYPES)
    set(LLVMDSDL_COVERAGE_CONFIG_SUBDIR "/$<CONFIG>")
    list(APPEND LLVMDSDL_COVERAGE_CTEST_CONFIG_ARGS --build-config $<CONFIG>)
  elseif(CMAKE_BUILD_TYPE)
    set(LLVMDSDL_COVERAGE_CONFIG_SUBDIR "/${CMAKE_BUILD_TYPE}")
  endif()

  set(LLVMDSDL_COVERAGE_DIR
      "${LLVMDSDL_BINARY_DIR}/coverage${LLVMDSDL_COVERAGE_CONFIG_SUBDIR}")
  set(LLVMDSDL_COVERAGE_PROFILES_DIR "${LLVMDSDL_COVERAGE_DIR}/profiles")
  set(LLVMDSDL_COVERAGE_PROFILE_PATTERN
      "${LLVMDSDL_COVERAGE_PROFILES_DIR}/default-%p-%m.profraw")
  set(LLVMDSDL_COVERAGE_PROFILE_DATA
      "${LLVMDSDL_COVERAGE_DIR}/coverage.profdata")
  set(LLVMDSDL_COVERAGE_SUMMARY
      "${LLVMDSDL_COVERAGE_DIR}/summary.txt")
  set(LLVMDSDL_COVERAGE_LCOV
      "${LLVMDSDL_COVERAGE_DIR}/coverage.lcov")
  set(LLVMDSDL_COVERAGE_HTML_DIR
      "${LLVMDSDL_COVERAGE_DIR}/html")

  add_custom_target(coverage-clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_COVERAGE_DIR}"
    USES_TERMINAL
    VERBATIM
    COMMENT "Removing coverage artifacts from ${LLVMDSDL_COVERAGE_DIR}"
  )

  add_custom_target(coverage-run
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_COVERAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_COVERAGE_PROFILES_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E env
        LLVM_PROFILE_FILE=${LLVMDSDL_COVERAGE_PROFILE_PATTERN}
        ${CMAKE_CTEST_COMMAND}
        ${LLVMDSDL_COVERAGE_CTEST_ARGS_LIST}
        ${LLVMDSDL_COVERAGE_CTEST_CONFIG_ARGS}
    WORKING_DIRECTORY ${LLVMDSDL_BINARY_DIR}
    DEPENDS llvmdsdl-unit-tests dsdlc dsdl-opt
    USES_TERMINAL
    VERBATIM
    COMMENT "Running tests with LLVM source coverage instrumentation"
  )

  add_custom_target(coverage-report
    COMMAND
      ${CMAKE_COMMAND}
        -DLLVM_PROFDATA=${LLVMDSDL_LLVM_PROFDATA_EXE}
        -DLLVM_COV=${LLVMDSDL_LLVM_COV_EXE}
        -DPROFILE_DIR=${LLVMDSDL_COVERAGE_PROFILES_DIR}
        -DPROFILE_DATA=${LLVMDSDL_COVERAGE_PROFILE_DATA}
        -DREPORT_TXT=${LLVMDSDL_COVERAGE_SUMMARY}
        -DREPORT_LCOV=${LLVMDSDL_COVERAGE_LCOV}
        -DREPORT_HTML_DIR=${LLVMDSDL_COVERAGE_HTML_DIR}
        -DSOURCE_DIR=${LLVMDSDL_SOURCE_DIR}
        -DBINARY_DIR=${LLVMDSDL_BINARY_DIR}
        "-DOBJECTS=$<TARGET_FILE:dsdlc>;$<TARGET_FILE:dsdl-opt>;$<TARGET_FILE:llvmdsdl-unit-tests>"
        "-DIGNORE_FILENAME_REGEX=.*/(build|submodules|\\.venv)/.*"
        -P ${LLVMDSDL_SOURCE_DIR}/cmake/GenerateCoverageReport.cmake
    DEPENDS coverage-run
    USES_TERMINAL
    VERBATIM
    COMMENT "Generating LLVM source coverage reports (text, lcov, html)"
  )

  add_custom_target(coverage
    DEPENDS coverage-report
  )
endif()
