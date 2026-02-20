//===----------------------------------------------------------------------===//
//
// Part of the OpenCyphal project, under the MIT licence
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
///
/// @file
/// POSIX UDP networking shim for libudpard demo applications.
///
/// This implementation uses non-blocking Berkeley sockets with multicast
/// membership and poll-based readiness waiting. It is designed for Linux/macOS.
///
//===----------------------------------------------------------------------===//

#include "udp_posix.h"

#ifndef _DEFAULT_SOURCE
#    define _DEFAULT_SOURCE
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define OVERRIDE_TTL 16
#define DSCP_MAX 63

static bool isMulticast(const uint32_t address)
{
    return (address & 0xF0000000UL) == 0xE0000000UL;
}

static int closeAndInvalidate(const int fd)
{
    if (fd >= 0)
    {
        return close(fd);
    }
    return 0;
}

static bool configureSocketNonBlocking(const int fd)
{
    return (fd >= 0) && (fcntl(fd, F_SETFL, O_NONBLOCK) == 0);
}

int16_t udpTxInit(UDPTxHandle* const self, const uint32_t local_iface_address)
{
    if ((self == NULL) || (local_iface_address == 0U))
    {
        return -EINVAL;
    }
    self->fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (self->fd < 0)
    {
        return (int16_t) -errno;
    }

    const int                ttl            = OVERRIDE_TTL;
    const uint32_t           local_iface_be = htonl(local_iface_address);
    const struct sockaddr_in bind_addr      = {
             .sin_family = AF_INET,
             .sin_port   = 0U,
             .sin_addr   = {.s_addr = local_iface_be},
    };

    bool ok = configureSocketNonBlocking(self->fd);
    ok      = ok && (bind(self->fd, (const struct sockaddr*) &bind_addr, sizeof(bind_addr)) == 0);
    ok      = ok && (setsockopt(self->fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) == 0);
    ok      = ok && (setsockopt(self->fd, IPPROTO_IP, IP_MULTICAST_IF, &local_iface_be, sizeof(local_iface_be)) == 0);
    if (!ok)
    {
        const int err = errno;
        (void) closeAndInvalidate(self->fd);
        self->fd = -1;
        return (int16_t) -err;
    }
    return 0;
}

int16_t udpTxSend(UDPTxHandle* const self,
                  const uint32_t     remote_address,
                  const uint16_t     remote_port,
                  const uint8_t      dscp,
                  const size_t       payload_size,
                  const void* const  payload)
{
    if ((self == NULL) || (self->fd < 0) || (remote_address == 0U) || (remote_port == 0U) || (payload == NULL) ||
        (dscp > DSCP_MAX))
    {
        return -EINVAL;
    }

    const int dscp_bits = ((int) dscp) << 2;
    (void) setsockopt(self->fd, IPPROTO_IP, IP_TOS, &dscp_bits, sizeof(dscp_bits));

    const struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port   = htons(remote_port),
        .sin_addr   = {.s_addr = htonl(remote_address)},
    };
    const ssize_t send_result =
        sendto(self->fd, payload, payload_size, MSG_DONTWAIT, (const struct sockaddr*) &dst, sizeof(dst));

    if (send_result == (ssize_t) payload_size)
    {
        return 1;
    }
    if ((errno == EAGAIN) || (errno == EWOULDBLOCK))
    {
        return 0;
    }
    return (int16_t) -errno;
}

void udpTxClose(UDPTxHandle* const self)
{
    if (self == NULL)
    {
        return;
    }
    (void) closeAndInvalidate(self->fd);
    self->fd = -1;
}

static bool bindSocketWithFallback(const int fd, const uint32_t multicast_group, const uint16_t remote_port)
{
    const struct sockaddr_in bind_multicast = {
        .sin_family = AF_INET,
        .sin_port   = htons(remote_port),
        .sin_addr   = {.s_addr = htonl(multicast_group)},
    };
    if (bind(fd, (const struct sockaddr*) &bind_multicast, sizeof(bind_multicast)) == 0)
    {
        return true;
    }
    const struct sockaddr_in bind_any = {
        .sin_family = AF_INET,
        .sin_port   = htons(remote_port),
        .sin_addr   = {.s_addr = htonl(INADDR_ANY)},
    };
    return bind(fd, (const struct sockaddr*) &bind_any, sizeof(bind_any)) == 0;
}

int16_t udpRxInit(UDPRxHandle* const self,
                  const uint32_t     local_iface_address,
                  const uint32_t     multicast_group,
                  const uint16_t     remote_port)
{
    if ((self == NULL) || (local_iface_address == 0U) || (remote_port == 0U) || !isMulticast(multicast_group))
    {
        return -EINVAL;
    }

    self->fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (self->fd < 0)
    {
        return (int16_t) -errno;
    }

    const int reuse = 1;
    bool      ok    = configureSocketNonBlocking(self->fd);
    ok              = ok && (setsockopt(self->fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == 0);
#ifdef SO_REUSEPORT
    ok = ok && (setsockopt(self->fd, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) == 0);
#endif
    ok = ok && bindSocketWithFallback(self->fd, multicast_group, remote_port);
    if (ok)
    {
        const struct ip_mreq membership = {
            .imr_multiaddr = {.s_addr = htonl(multicast_group)},
            .imr_interface = {.s_addr = htonl(local_iface_address)},
        };
        ok = setsockopt(self->fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership, sizeof(membership)) == 0;
    }
    if (!ok)
    {
        const int err = errno;
        (void) closeAndInvalidate(self->fd);
        self->fd = -1;
        return (int16_t) -err;
    }
    return 0;
}

int16_t udpRxReceive(UDPRxHandle* const self, size_t* const inout_payload_size, void* const out_payload)
{
    if ((self == NULL) || (self->fd < 0) || (inout_payload_size == NULL) || (out_payload == NULL))
    {
        return -EINVAL;
    }

    const ssize_t recv_result = recv(self->fd, out_payload, *inout_payload_size, MSG_DONTWAIT);
    if (recv_result >= 0)
    {
        *inout_payload_size = (size_t) recv_result;
        return 1;
    }
    if ((errno == EAGAIN) || (errno == EWOULDBLOCK))
    {
        return 0;
    }
    return (int16_t) -errno;
}

void udpRxClose(UDPRxHandle* const self)
{
    if (self == NULL)
    {
        return;
    }
    (void) closeAndInvalidate(self->fd);
    self->fd = -1;
}

int16_t udpWait(const uint64_t        timeout_usec,
                const size_t          tx_count,
                UDPTxAwaitable* const tx,
                const size_t          rx_count,
                UDPRxAwaitable* const rx)
{
    const size_t total_count = tx_count + rx_count;
    if ((total_count == 0U) || (total_count > (size_t) INT32_MAX) || (tx == NULL) || (rx == NULL))
    {
        return -EINVAL;
    }

    struct pollfd* const fds = (struct pollfd*) calloc(total_count, sizeof(struct pollfd));
    if (fds == NULL)
    {
        return -ENOMEM;
    }

    size_t idx = 0;
    for (; idx < tx_count; ++idx)
    {
        fds[idx].fd     = tx[idx].handle->fd;
        fds[idx].events = POLLOUT;
        tx[idx].ready   = false;
    }
    for (; idx < total_count; ++idx)
    {
        fds[idx].fd              = rx[idx - tx_count].handle->fd;
        fds[idx].events          = POLLIN;
        rx[idx - tx_count].ready = false;
    }

    const uint64_t timeout_ms       = timeout_usec / 1000U;
    const int      timeout_ms_bound = (timeout_ms > (uint64_t) INT_MAX) ? INT_MAX : (int) timeout_ms;
    const int      poll_result      = poll(fds, (nfds_t) total_count, timeout_ms_bound);
    if (poll_result < 0)
    {
        const int err = errno;
        free(fds);
        return (int16_t) -err;
    }

    for (idx = 0; idx < tx_count; ++idx)
    {
        tx[idx].ready = (fds[idx].revents & POLLOUT) != 0;
    }
    for (; idx < total_count; ++idx)
    {
        rx[idx - tx_count].ready = (fds[idx].revents & POLLIN) != 0;
    }

    free(fds);
    return 0;
}

uint32_t udpParseIfaceAddress(const char* const address)
{
    if (address == NULL)
    {
        return 0U;
    }
    struct in_addr addr = {0};
    if (inet_pton(AF_INET, address, &addr) != 1)
    {
        return 0U;
    }
    return ntohl(addr.s_addr);
}
