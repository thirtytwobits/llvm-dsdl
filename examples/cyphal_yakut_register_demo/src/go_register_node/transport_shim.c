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
/// This implementation owns libudpard transport state and exposes simple C
/// entrypoints for init, publish, receive, and response operations.
///
//===----------------------------------------------------------------------===//

#include "transport_shim.h"

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "udp_posix.h"
#include "udpard.h"

struct GoDemoNode
{
    bool                    running;
    UdpardNodeID            node_id;
    uint32_t                local_iface_address;
    size_t                  rx_datagram_capacity;
    UdpardPortID            register_access_service_id;
    UdpardPortID            register_list_service_id;

    struct UdpardTxMemoryResources tx_memory;
    struct UdpardRxMemoryResources rx_memory;

    struct UdpardTx              tx;
    struct UdpardRxRPCDispatcher rpc_dispatcher;
    struct UdpardRxRPCPort       register_access_port;
    struct UdpardRxRPCPort       register_list_port;
    struct UdpardUDPIPEndpoint   rpc_endpoint;

    UDPTxHandle tx_socket;
    UDPRxHandle rx_socket;
};

static void* heap_allocate(void* const user_reference, const size_t size)
{
    (void) user_reference;
    return malloc(size);
}

static void heap_deallocate(void* const user_reference, const size_t size, void* const pointer)
{
    (void) user_reference;
    (void) size;
    free(pointer);
}

static void init_node_defaults(GoDemoNode* const node)
{
    memset(node, 0, sizeof(*node));
    node->tx_socket.fd = -1;
    node->rx_socket.fd = -1;
}

static void drain_tx_queue(GoDemoNode* const node)
{
    if (node == NULL)
    {
        return;
    }
    for (;;)
    {
        struct UdpardTxItem* const item = udpardTxPeek(&node->tx);
        if (item == NULL)
        {
            break;
        }
        udpardTxFree(node->tx.memory, udpardTxPop(&node->tx, item));
    }
}

GoDemoNode* go_demo_node_create(void)
{
    GoDemoNode* const node = (GoDemoNode*) malloc(sizeof(GoDemoNode));
    if (node == NULL)
    {
        return NULL;
    }
    init_node_defaults(node);
    return node;
}

void go_demo_node_destroy(GoDemoNode* const node)
{
    if (node == NULL)
    {
        return;
    }
    (void) go_demo_node_shutdown(node);
    free(node);
}

uint16_t go_demo_node_id_max(void)
{
    return (uint16_t) UDPARD_NODE_ID_MAX;
}

uint8_t go_demo_priority_nominal(void)
{
    return (uint8_t) UdpardPriorityNominal;
}

int go_demo_parse_iface_address(const char* const address, uint32_t* const out_iface_address)
{
    if ((address == NULL) || (out_iface_address == NULL))
    {
        return -EINVAL;
    }
    const uint32_t parsed = udpParseIfaceAddress(address);
    if (parsed == 0U)
    {
        return -EINVAL;
    }
    *out_iface_address = parsed;
    return 0;
}

int go_demo_node_init(GoDemoNode* const node,
                      const uint16_t    node_id,
                      const char* const iface_address,
                      const uint16_t    register_access_service_id,
                      const size_t      register_access_request_extent,
                      const uint16_t    register_list_service_id,
                      const size_t      register_list_request_extent,
                      const uint32_t    tx_queue_capacity,
                      const size_t      rx_datagram_capacity)
{
    if ((node == NULL) || (iface_address == NULL) || (tx_queue_capacity == 0U) ||
        (register_access_request_extent == 0U) || (register_list_request_extent == 0U))
    {
        return -EINVAL;
    }
    if (node->running)
    {
        return -EALREADY;
    }

    init_node_defaults(node);

    node->node_id = (UdpardNodeID) node_id;
    node->register_access_service_id = (UdpardPortID) register_access_service_id;
    node->register_list_service_id   = (UdpardPortID) register_list_service_id;
    node->rx_datagram_capacity       = (rx_datagram_capacity > 0U) ? rx_datagram_capacity : 2048U;

    const int parse_rc = go_demo_parse_iface_address(iface_address, &node->local_iface_address);
    if (parse_rc < 0)
    {
        return parse_rc;
    }

    const struct UdpardMemoryResource resource = {
        .user_reference = NULL,
        .deallocate     = heap_deallocate,
        .allocate       = heap_allocate,
    };
    node->tx_memory = (struct UdpardTxMemoryResources){
        .fragment = resource,
        .payload  = resource,
    };
    node->rx_memory = (struct UdpardRxMemoryResources){
        .session  = resource,
        .fragment = resource,
        .payload  = {
            .user_reference = NULL,
            .deallocate     = heap_deallocate,
        },
    };

    const int_fast8_t tx_init_rc = udpardTxInit(&node->tx, &node->node_id, tx_queue_capacity, node->tx_memory);
    if (tx_init_rc < 0)
    {
        return (int) tx_init_rc;
    }

    const int_fast8_t dispatcher_init_rc = udpardRxRPCDispatcherInit(&node->rpc_dispatcher, node->rx_memory);
    if (dispatcher_init_rc < 0)
    {
        return (int) dispatcher_init_rc;
    }

    const int_fast8_t dispatcher_start_rc =
        udpardRxRPCDispatcherStart(&node->rpc_dispatcher, node->node_id, &node->rpc_endpoint);
    if (dispatcher_start_rc < 0)
    {
        return (int) dispatcher_start_rc;
    }

    const int_fast8_t access_listen_rc = udpardRxRPCDispatcherListen(&node->rpc_dispatcher,
                                                                      &node->register_access_port,
                                                                      node->register_access_service_id,
                                                                      true,
                                                                      register_access_request_extent);
    if (access_listen_rc < 0)
    {
        return (int) access_listen_rc;
    }

    const int_fast8_t list_listen_rc = udpardRxRPCDispatcherListen(&node->rpc_dispatcher,
                                                                    &node->register_list_port,
                                                                    node->register_list_service_id,
                                                                    true,
                                                                    register_list_request_extent);
    if (list_listen_rc < 0)
    {
        return (int) list_listen_rc;
    }

    const int16_t tx_socket_rc = udpTxInit(&node->tx_socket, node->local_iface_address);
    if (tx_socket_rc < 0)
    {
        return (int) tx_socket_rc;
    }

    const int16_t rx_socket_rc =
        udpRxInit(&node->rx_socket, node->local_iface_address, node->rpc_endpoint.ip_address, node->rpc_endpoint.udp_port);
    if (rx_socket_rc < 0)
    {
        return (int) rx_socket_rc;
    }

    node->running = true;
    return 0;
}

int go_demo_node_get_rpc_endpoint(const GoDemoNode* const node, uint32_t* const out_ip_address, uint16_t* const out_udp_port)
{
    if ((node == NULL) || (out_ip_address == NULL) || (out_udp_port == NULL) || (!node->running))
    {
        return -EINVAL;
    }
    *out_ip_address = node->rpc_endpoint.ip_address;
    *out_udp_port   = node->rpc_endpoint.udp_port;
    return 0;
}

static UdpardMicrosecond now_usec(void)
{
    struct timespec ts = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        return 0U;
    }
    return ((UdpardMicrosecond) ts.tv_sec * 1000000ULL) + ((UdpardMicrosecond) ts.tv_nsec / 1000ULL);
}

uint64_t go_demo_now_usec(void)
{
    return (uint64_t) now_usec();
}

void go_demo_node_pump_tx(GoDemoNode* const node)
{
    if ((node == NULL) || (!node->running))
    {
        return;
    }

    const UdpardMicrosecond now = now_usec();
    for (;;)
    {
        struct UdpardTxItem* const item = udpardTxPeek(&node->tx);
        if (item == NULL)
        {
            break;
        }

        if (item->deadline_usec <= now)
        {
            udpardTxFree(node->tx.memory, udpardTxPop(&node->tx, item));
            continue;
        }

        const int16_t send_rc = udpTxSend(&node->tx_socket,
                                          item->destination.ip_address,
                                          item->destination.udp_port,
                                          item->dscp,
                                          item->datagram_payload.size,
                                          item->datagram_payload.data);
        if (send_rc == 1)
        {
            udpardTxFree(node->tx.memory, udpardTxPop(&node->tx, item));
            continue;
        }
        if (send_rc == 0)
        {
            break;
        }

        udpardTxFree(node->tx.memory, udpardTxPop(&node->tx, item));
    }
}

int go_demo_node_publish(GoDemoNode* const    node,
                         const uint16_t       subject_id,
                         const uint8_t        transfer_id,
                         const uint8_t* const payload,
                         const size_t         payload_size,
                         const uint64_t       deadline_usec,
                         const uint8_t        priority)
{
    if ((node == NULL) || (!node->running) || ((payload == NULL) && (payload_size > 0U)))
    {
        return -EINVAL;
    }
    const int32_t rc = udpardTxPublish(&node->tx,
                                       (UdpardMicrosecond) deadline_usec,
                                       (enum UdpardPriority) priority,
                                       (UdpardPortID) subject_id,
                                       (UdpardTransferID) transfer_id,
                                       (struct UdpardPayload){
                                           .size = payload_size,
                                           .data = payload,
                                       },
                                       NULL);
    return (int) rc;
}

int go_demo_node_respond(GoDemoNode* const    node,
                         const uint16_t       service_id,
                         const uint16_t       destination_node_id,
                         const uint8_t        transfer_id,
                         const uint8_t* const payload,
                         const size_t         payload_size,
                         const uint64_t       deadline_usec,
                         const uint8_t        priority)
{
    if ((node == NULL) || (!node->running) || ((payload == NULL) && (payload_size > 0U)))
    {
        return -EINVAL;
    }
    const int32_t rc = udpardTxRespond(&node->tx,
                                       (UdpardMicrosecond) deadline_usec,
                                       (enum UdpardPriority) priority,
                                       (UdpardPortID) service_id,
                                       (UdpardNodeID) destination_node_id,
                                       (UdpardTransferID) transfer_id,
                                       (struct UdpardPayload){
                                           .size = payload_size,
                                           .data = payload,
                                       },
                                       NULL);
    return (int) rc;
}

static int collect_dispatch(GoDemoNode* const      node,
                            struct UdpardRxRPCPort* const out_port,
                            const struct UdpardRxRPCTransfer transfer,
                            GoDemoRpcTransfer* const out_transfer)
{
    const bool is_access = out_port == &node->register_access_port;
    const bool is_list   = out_port == &node->register_list_port;
    if ((!is_access) && (!is_list))
    {
        udpardRxFragmentFree(transfer.base.payload, node->rx_memory.fragment, node->rx_memory.payload);
        return 0;
    }

    uint8_t* payload_copy = NULL;
    if (transfer.base.payload_size > 0U)
    {
        payload_copy = (uint8_t*) malloc(transfer.base.payload_size);
        if (payload_copy == NULL)
        {
            udpardRxFragmentFree(transfer.base.payload, node->rx_memory.fragment, node->rx_memory.payload);
            return -ENOMEM;
        }
        (void) udpardGather(transfer.base.payload, transfer.base.payload_size, payload_copy);
    }

    out_transfer->service_id     = (uint16_t) (is_access ? node->register_access_service_id : node->register_list_service_id);
    out_transfer->source_node_id = (uint16_t) transfer.base.source_node_id;
    out_transfer->transfer_id    = (uint8_t) transfer.base.transfer_id;
    out_transfer->payload        = payload_copy;
    out_transfer->payload_size   = transfer.base.payload_size;

    udpardRxFragmentFree(transfer.base.payload, node->rx_memory.fragment, node->rx_memory.payload);
    return 1;
}

int go_demo_node_poll_rpc(GoDemoNode* const node, const uint64_t timeout_usec, GoDemoRpcTransfer* const out_transfer)
{
    if ((node == NULL) || (!node->running) || (out_transfer == NULL))
    {
        return -EINVAL;
    }

    memset(out_transfer, 0, sizeof(*out_transfer));
    go_demo_node_pump_tx(node);

    UDPTxAwaitable tx_await = {
        .handle         = &node->tx_socket,
        .ready          = false,
        .user_reference = NULL,
    };
    UDPRxAwaitable rx_await = {
        .handle         = &node->rx_socket,
        .ready          = false,
        .user_reference = NULL,
    };
    const int16_t wait_rc = udpWait(timeout_usec, 1U, &tx_await, 1U, &rx_await);
    if (wait_rc < 0)
    {
        return (int) wait_rc;
    }

    if (tx_await.ready)
    {
        go_demo_node_pump_tx(node);
    }
    if (!rx_await.ready)
    {
        return 0;
    }

    for (;;)
    {
        void* const datagram_buffer = malloc(node->rx_datagram_capacity);
        if (datagram_buffer == NULL)
        {
            return -ENOMEM;
        }

        size_t        payload_size = node->rx_datagram_capacity;
        const int16_t rx_rc        = udpRxReceive(&node->rx_socket, &payload_size, datagram_buffer);
        if (rx_rc == 0)
        {
            free(datagram_buffer);
            return 0;
        }
        if (rx_rc < 0)
        {
            free(datagram_buffer);
            if ((rx_rc == -EAGAIN) || (rx_rc == -EWOULDBLOCK))
            {
                return 0;
            }
            return (int) rx_rc;
        }

        struct UdpardRxRPCTransfer transfer     = {0};
        struct UdpardRxRPCPort*    matched_port = NULL;
        const int_fast8_t   dispatch_rc  = udpardRxRPCDispatcherReceive(&node->rpc_dispatcher,
                                                                      now_usec(),
                                                                        (struct UdpardMutablePayload){
                                                                            .size = payload_size,
                                                                            .data = datagram_buffer,
                                                                        },
                                                                        0U,
                                                                        &matched_port,
                                                                        &transfer);
        if (dispatch_rc < 0)
        {
            continue;
        }
        if (dispatch_rc == 0)
        {
            continue;
        }

        return collect_dispatch(node, matched_port, transfer, out_transfer);
    }
}

void go_demo_node_release_transfer(GoDemoRpcTransfer* const transfer)
{
    if (transfer == NULL)
    {
        return;
    }
    if (transfer->payload != NULL)
    {
        free(transfer->payload);
    }
    memset(transfer, 0, sizeof(*transfer));
}

int go_demo_node_shutdown(GoDemoNode* const node)
{
    if (node == NULL)
    {
        return -EINVAL;
    }
    if (!node->running)
    {
        return 0;
    }

    udpardRxRPCDispatcherCancel(&node->rpc_dispatcher, node->register_access_service_id, true);
    udpardRxRPCDispatcherCancel(&node->rpc_dispatcher, node->register_list_service_id, true);
    udpRxClose(&node->rx_socket);
    udpTxClose(&node->tx_socket);
    drain_tx_queue(node);

    node->running = false;
    return 0;
}
