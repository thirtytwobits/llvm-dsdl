set(LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT "")
if(EXISTS "${CMAKE_SOURCE_DIR}/public_regulated_dsdl/uavcan")
  set(LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT
      "${CMAKE_SOURCE_DIR}/public_regulated_dsdl/uavcan")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/public_regulated_data_types/uavcan")
  set(LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT
      "${CMAKE_SOURCE_DIR}/public_regulated_data_types/uavcan")
endif()

if(NOT LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT)
  message(WARNING
    "Skipping cyphal yakut register demo target: uavcan regulated DSDL root not found.")
  return()
endif()

get_filename_component(LLVMDSDL_YAKUT_DEMO_CYPHAL_PATH
  "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}"
  DIRECTORY)

set(LLVMDSDL_YAKUT_DEMO_UDPARD_C
    "${CMAKE_SOURCE_DIR}/libudpard/libudpard/udpard.c")
set(LLVMDSDL_YAKUT_DEMO_UDPARD_INCLUDE
    "${CMAKE_SOURCE_DIR}/libudpard/libudpard")

if(NOT EXISTS "${LLVMDSDL_YAKUT_DEMO_UDPARD_C}")
  message(WARNING
    "Skipping cyphal yakut register demo target: libudpard submodule source "
    "not found at ${LLVMDSDL_YAKUT_DEMO_UDPARD_C}")
  return()
endif()

set(LLVMDSDL_YAKUT_DEMO_GEN_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated-go")
set(LLVMDSDL_YAKUT_DEMO_LOG_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/run-logs")

set(LLVMDSDL_YAKUT_DEMO_NODE_ID
    "42"
    CACHE STRING
    "Node ID used by the run-yakut-register-demo target")
set(LLVMDSDL_YAKUT_DEMO_YAKUT_NODE_ID
    "100"
    CACHE STRING
    "Yakut local node-ID used by the run-yakut-register-demo target")
set(LLVMDSDL_YAKUT_DEMO_IFACE
    "127.0.0.1"
    CACHE STRING
    "IPv4 interface used by the run-yakut-register-demo target")
set(LLVMDSDL_YAKUT_DEMO_DURATION_SECONDS
    "4"
    CACHE STRING
    "Duration in seconds for the run-yakut-register-demo target")
set(LLVMDSDL_YAKUT_DEMO_HEARTBEAT_COUNT
    "3"
    CACHE STRING
    "Heartbeat samples required by run-yakut-register-heartbeat-demo")
set(LLVMDSDL_YAKUT_DEMO_HEARTBEAT_TIMEOUT_SECONDS
    "12"
    CACHE STRING
    "Timeout in seconds for heartbeat validation in run-yakut-register-heartbeat-demo")

set(LLVMDSDL_YAKUT_DEMO_YAKUT_BIN "")
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/yakut")
  set(LLVMDSDL_YAKUT_DEMO_YAKUT_BIN
      "${CMAKE_SOURCE_DIR}/.venv/bin/yakut")
endif()
set(LLVMDSDL_YAKUT_DEMO_YAKUT_BIN
    "${LLVMDSDL_YAKUT_DEMO_YAKUT_BIN}"
    CACHE FILEPATH
    "Path to yakut executable used by run-yakut-register-demo (empty means PATH lookup)")

set(LLVMDSDL_YAKUT_DEMO_GENERATED_FILES
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/uavcan/node/Heartbeat_1_0.hpp"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/uavcan/register/Access_1_0.hpp"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/uavcan/register/List_1_0.hpp"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/uavcan/register/Name_1_0.hpp"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/uavcan/register/Value_1_0.hpp"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/dsdl_runtime.h"
    "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}/dsdl_runtime.hpp")

add_custom_command(
  OUTPUT
    ${LLVMDSDL_YAKUT_DEMO_GENERATED_FILES}
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}"
  COMMAND
    $<TARGET_FILE:dsdlc>
      cpp
      --root-namespace-dir "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}"
      --cpp-profile std
      --out-dir "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}"
  DEPENDS
    dsdlc
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/node/7509.Heartbeat.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/384.Access.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/385.List.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/Name.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/Value.1.0.dsdl"
  VERBATIM
  COMMENT "Generating uavcan C++ DSDL artifacts for yakut register demo")

add_custom_target(generate-cyphal-yakut-register-demo-cpp
  DEPENDS ${LLVMDSDL_YAKUT_DEMO_GENERATED_FILES})

add_executable(cyphal-yakut-register-node
  src/native_register_node.cpp
  "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net/udp_posix.c"
  "${LLVMDSDL_YAKUT_DEMO_UDPARD_C}")

add_dependencies(cyphal-yakut-register-node
  generate-cyphal-yakut-register-demo-cpp)

target_include_directories(cyphal-yakut-register-node PRIVATE
  "${LLVMDSDL_YAKUT_DEMO_GEN_DIR}"
  "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net"
  "${LLVMDSDL_YAKUT_DEMO_UDPARD_INCLUDE}")

set_target_properties(cyphal-yakut-register-node PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)

set(LLVMDSDL_YAKUT_DEMO_GO_GENERATED_FILES
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/go.mod"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/uavcan/node/heartbeat_1_0.go"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/uavcan/register/access_1_0.go"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/uavcan/register/list_1_0.go"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/uavcan/register/name_1_0.go"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/uavcan/register/value_1_0.go"
    "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}/dsdlruntime/dsdl_runtime.go")

add_custom_command(
  OUTPUT
    ${LLVMDSDL_YAKUT_DEMO_GO_GENERATED_FILES}
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}"
  COMMAND
    $<TARGET_FILE:dsdlc>
      go
      --root-namespace-dir "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}"
      --go-module cyphal_yakut_demo_generated
      --out-dir "${LLVMDSDL_YAKUT_DEMO_GEN_GO_DIR}"
  DEPENDS
    dsdlc
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/node/7509.Heartbeat.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/384.Access.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/385.List.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/Name.1.0.dsdl"
    "${LLVMDSDL_YAKUT_DEMO_UAVCAN_ROOT}/register/Value.1.0.dsdl"
  VERBATIM
  COMMENT "Generating uavcan Go DSDL artifacts for yakut register demo")

add_custom_target(generate-cyphal-yakut-register-demo-go
  DEPENDS ${LLVMDSDL_YAKUT_DEMO_GO_GENERATED_FILES})

set(LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/go-node")
set(LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY
    "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/cyphal-yakut-register-go-node-bin")
set(LLVMDSDL_YAKUT_DEMO_GO_NODE_GOMOD
    "${CMAKE_CURRENT_BINARY_DIR}/go-node.mod")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/go.mod.in"
  "${LLVMDSDL_YAKUT_DEMO_GO_NODE_GOMOD}"
  @ONLY)

find_program(LLVMDSDL_YAKUT_DEMO_GO_EXECUTABLE go QUIET)
if(LLVMDSDL_YAKUT_DEMO_GO_EXECUTABLE)
  add_custom_command(
    OUTPUT
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/main.go"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/main.go"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/transport_shim.h"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/transport_shim.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/transport_shim.c"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/transport_shim.c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net/udp_posix.h"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/udp_posix.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net/udp_posix.c"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/udp_posix.c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/udpard.h"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/udpard.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/udpard.c"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/udpard.c"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/_udpard_cavl.h"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/_udpard_cavl.h"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_GOMOD}"
      "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/go.mod"
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}"
        ${CMAKE_COMMAND} -E env
          CC=${CMAKE_C_COMPILER}
          CGO_ENABLED=1
          GOCACHE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/.gocache
          GOMODCACHE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/.gomodcache
          ${LLVMDSDL_YAKUT_DEMO_GO_EXECUTABLE} mod tidy
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}"
        ${CMAKE_COMMAND} -E env
          CC=${CMAKE_C_COMPILER}
          CGO_ENABLED=1
          GOCACHE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/.gocache
          GOMODCACHE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_DIR}/.gomodcache
          ${LLVMDSDL_YAKUT_DEMO_GO_EXECUTABLE} build -o "${LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY}" .
    DEPENDS
      generate-cyphal-yakut-register-demo-go
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/main.go"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/transport_shim.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/transport_shim.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/go_register_node/go.mod.in"
      "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net/udp_posix.h"
      "${CMAKE_SOURCE_DIR}/examples/cyphal_ping_pong/src/net/udp_posix.c"
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/udpard.h"
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/udpard.c"
      "${CMAKE_SOURCE_DIR}/libudpard/libudpard/_udpard_cavl.h"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM
    COMMENT "Building Go yakut register demo node")

  add_custom_target(cyphal-yakut-register-go-node
    DEPENDS "${LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY}")
else()
  message(STATUS
    "Skipping Go yakut register demo node targets: go executable unavailable")
endif()

add_custom_target(run-yakut-register-demo-native
  COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_LOG_DIR}"
  COMMAND
    ${CMAKE_COMMAND} -E env
      NODE_EXE=$<TARGET_FILE:cyphal-yakut-register-node>
      NODE_BACKEND=native
      LOG_DIR=${LLVMDSDL_YAKUT_DEMO_LOG_DIR}
      CYPHAL_PATH_ROOT=${LLVMDSDL_YAKUT_DEMO_CYPHAL_PATH}
      NODE_ID=${LLVMDSDL_YAKUT_DEMO_NODE_ID}
      YAKUT_NODE_ID=${LLVMDSDL_YAKUT_DEMO_YAKUT_NODE_ID}
      IFACE=${LLVMDSDL_YAKUT_DEMO_IFACE}
      DURATION_SECONDS=${LLVMDSDL_YAKUT_DEMO_DURATION_SECONDS}
      YAKUT_BIN=${LLVMDSDL_YAKUT_DEMO_YAKUT_BIN}
      /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/run_yakut_register_demo.sh"
  DEPENDS cyphal-yakut-register-node
  USES_TERMINAL
  VERBATIM
  COMMENT "Running yakut register demo against native libudpard node")

add_custom_target(run-yakut-register-heartbeat-demo-native
  COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_LOG_DIR}"
  COMMAND
    ${CMAKE_COMMAND} -E env
      NODE_EXE=$<TARGET_FILE:cyphal-yakut-register-node>
      NODE_BACKEND=native
      LOG_DIR=${LLVMDSDL_YAKUT_DEMO_LOG_DIR}
      CYPHAL_PATH_ROOT=${LLVMDSDL_YAKUT_DEMO_CYPHAL_PATH}
      NODE_ID=${LLVMDSDL_YAKUT_DEMO_NODE_ID}
      YAKUT_NODE_ID=${LLVMDSDL_YAKUT_DEMO_YAKUT_NODE_ID}
      IFACE=${LLVMDSDL_YAKUT_DEMO_IFACE}
      DURATION_SECONDS=${LLVMDSDL_YAKUT_DEMO_DURATION_SECONDS}
      YAKUT_BIN=${LLVMDSDL_YAKUT_DEMO_YAKUT_BIN}
      VERIFY_HEARTBEAT=1
      HEARTBEAT_COUNT=${LLVMDSDL_YAKUT_DEMO_HEARTBEAT_COUNT}
      HEARTBEAT_TIMEOUT_SECONDS=${LLVMDSDL_YAKUT_DEMO_HEARTBEAT_TIMEOUT_SECONDS}
      /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/run_yakut_register_demo.sh"
  DEPENDS cyphal-yakut-register-node
  USES_TERMINAL
  VERBATIM
  COMMENT "Running yakut register demo with heartbeat subscribe verification")

if(TARGET cyphal-yakut-register-go-node)
  add_custom_target(run-yakut-register-demo-go
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_LOG_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E env
        NODE_EXE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY}
        NODE_BACKEND=go
        LOG_DIR=${LLVMDSDL_YAKUT_DEMO_LOG_DIR}
        CYPHAL_PATH_ROOT=${LLVMDSDL_YAKUT_DEMO_CYPHAL_PATH}
        NODE_ID=${LLVMDSDL_YAKUT_DEMO_NODE_ID}
        YAKUT_NODE_ID=${LLVMDSDL_YAKUT_DEMO_YAKUT_NODE_ID}
        IFACE=${LLVMDSDL_YAKUT_DEMO_IFACE}
        DURATION_SECONDS=${LLVMDSDL_YAKUT_DEMO_DURATION_SECONDS}
        YAKUT_BIN=${LLVMDSDL_YAKUT_DEMO_YAKUT_BIN}
        /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/run_yakut_register_demo.sh"
    DEPENDS cyphal-yakut-register-go-node
    USES_TERMINAL
    VERBATIM
    COMMENT "Running yakut register demo against Go libudpard node")

  add_custom_target(run-yakut-register-heartbeat-demo-go
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LLVMDSDL_YAKUT_DEMO_LOG_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E env
        NODE_EXE=${LLVMDSDL_YAKUT_DEMO_GO_NODE_BINARY}
        NODE_BACKEND=go
        LOG_DIR=${LLVMDSDL_YAKUT_DEMO_LOG_DIR}
        CYPHAL_PATH_ROOT=${LLVMDSDL_YAKUT_DEMO_CYPHAL_PATH}
        NODE_ID=${LLVMDSDL_YAKUT_DEMO_NODE_ID}
        YAKUT_NODE_ID=${LLVMDSDL_YAKUT_DEMO_YAKUT_NODE_ID}
        IFACE=${LLVMDSDL_YAKUT_DEMO_IFACE}
        DURATION_SECONDS=${LLVMDSDL_YAKUT_DEMO_DURATION_SECONDS}
        YAKUT_BIN=${LLVMDSDL_YAKUT_DEMO_YAKUT_BIN}
        VERIFY_HEARTBEAT=1
        HEARTBEAT_COUNT=${LLVMDSDL_YAKUT_DEMO_HEARTBEAT_COUNT}
        HEARTBEAT_TIMEOUT_SECONDS=${LLVMDSDL_YAKUT_DEMO_HEARTBEAT_TIMEOUT_SECONDS}
        /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/run_yakut_register_demo.sh"
    DEPENDS cyphal-yakut-register-go-node
    USES_TERMINAL
    VERBATIM
    COMMENT "Running yakut register demo with heartbeat verification against Go node")

  add_custom_target(run-yakut-register-demo-all)
  add_dependencies(run-yakut-register-demo-all
    run-yakut-register-demo-native
    run-yakut-register-demo-go)

  add_custom_target(run-yakut-register-heartbeat-demo-all)
  add_dependencies(run-yakut-register-heartbeat-demo-all
    run-yakut-register-heartbeat-demo-native
    run-yakut-register-heartbeat-demo-go)
endif()

add_custom_target(run-yakut-register-demo)
add_dependencies(run-yakut-register-demo run-yakut-register-demo-native)

add_custom_target(run-yakut-register-heartbeat-demo)
add_dependencies(run-yakut-register-heartbeat-demo run-yakut-register-heartbeat-demo-native)
