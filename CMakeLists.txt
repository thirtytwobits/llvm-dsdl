cmake_minimum_required(VERSION 3.24)
project(llvm-dsdl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(LLVMDSDL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVMDSDL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(include/llvmdsdl)
add_subdirectory(lib)
add_subdirectory(tools)

set(LLVMDSDL_UAVCAN_ROOT "")
if(EXISTS "${LLVMDSDL_SOURCE_DIR}/public_regulated_dsdl/uavcan")
  set(LLVMDSDL_UAVCAN_ROOT
      "${LLVMDSDL_SOURCE_DIR}/public_regulated_dsdl/uavcan")
elseif(EXISTS "${LLVMDSDL_SOURCE_DIR}/public_regulated_data_types/uavcan")
  set(LLVMDSDL_UAVCAN_ROOT
      "${LLVMDSDL_SOURCE_DIR}/public_regulated_data_types/uavcan")
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
          --strict
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
          --strict
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

  add_custom_target(generate-uavcan-all)
  add_dependencies(generate-uavcan-all
                   generate-uavcan-c
                   generate-uavcan-cpp-both
                   generate-uavcan-cpp-std
                   generate-uavcan-cpp-pmr
                   generate-uavcan-rust-std)

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
    "${LLVMDSDL_SOURCE_DIR}/public_regulated_data_types/uavcan. "
    "Skipping generation targets.")
endif()

enable_testing()
add_subdirectory(test)
