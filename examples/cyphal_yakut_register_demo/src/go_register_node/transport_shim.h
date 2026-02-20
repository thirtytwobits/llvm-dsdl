//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// C transport shim used by the Go Yakut register demo node.
///
/// This shim wraps libudpard and the demo POSIX UDP adapter to provide a
/// compact C API that can be called from cgo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVMDSDL_EXAMPLES_CYPHAL_YAKUT_REGISTER_DEMO_GO_TRANSPORT_SHIM_H
#define LLVMDSDL_EXAMPLES_CYPHAL_YAKUT_REGISTER_DEMO_GO_TRANSPORT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GoDemoNode GoDemoNode;

    typedef struct
    {
        uint16_t service_id;
        uint16_t source_node_id;
        uint8_t  transfer_id;
        uint8_t* payload;
        size_t   payload_size;
    } GoDemoRpcTransfer;

    GoDemoNode* go_demo_node_create(void);

    void go_demo_node_destroy(GoDemoNode* node);

    uint16_t go_demo_node_id_max(void);

    uint8_t go_demo_priority_nominal(void);

    int go_demo_parse_iface_address(const char* address, uint32_t* out_iface_address);

    int go_demo_node_init(GoDemoNode* node,
                          uint16_t    node_id,
                          const char* iface_address,
                          uint16_t    register_access_service_id,
                          size_t      register_access_request_extent,
                          uint16_t    register_list_service_id,
                          size_t      register_list_request_extent,
                          uint32_t    tx_queue_capacity,
                          size_t      rx_datagram_capacity);

    int go_demo_node_get_rpc_endpoint(const GoDemoNode* node, uint32_t* out_ip_address, uint16_t* out_udp_port);

    int go_demo_node_publish(GoDemoNode*     node,
                             uint16_t        subject_id,
                             uint8_t         transfer_id,
                             const uint8_t*  payload,
                             size_t          payload_size,
                             uint64_t        deadline_usec,
                             uint8_t         priority);

    int go_demo_node_respond(GoDemoNode*     node,
                             uint16_t        service_id,
                             uint16_t        destination_node_id,
                             uint8_t         transfer_id,
                             const uint8_t*  payload,
                             size_t          payload_size,
                             uint64_t        deadline_usec,
                             uint8_t         priority);

    int go_demo_node_poll_rpc(GoDemoNode* node, uint64_t timeout_usec, GoDemoRpcTransfer* out_transfer);

    void go_demo_node_release_transfer(GoDemoRpcTransfer* transfer);

    void go_demo_node_pump_tx(GoDemoNode* node);

    uint64_t go_demo_now_usec(void);

    int go_demo_node_shutdown(GoDemoNode* node);

#ifdef __cplusplus
}
#endif

#endif  // LLVMDSDL_EXAMPLES_CYPHAL_YAKUT_REGISTER_DEMO_GO_TRANSPORT_SHIM_H
