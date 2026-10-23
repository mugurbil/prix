#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Peer-to-peer TCP communicator, optionally over mutually authenticated TLS.

Every process (MPC parties, the TTP dealer if used, and Fission evaluators) has an address
in `peers`, indexed by rank. Each pair of processes shares one direct TCP connection; there
is no coordinator in the data path. Messages use PriX's pickle-free wire format
(`prix.communicator.wire`).

All sockets are non-blocking and every blocking wait keeps flushing pending outgoing data,
so two parties that send large messages to each other at the same time cannot deadlock.
"""

import collections
import functools
import hashlib
import logging
import os
import selectors
import socket
import ssl
import struct
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import wire
from .communicator import _logging
from .distributed_communicator import DistributedCommunicator

_MAGIC = b"PRIX\x01"
_LEN = struct.Struct("<Q")
_HELLO = struct.Struct("<5sI32s")  # magic, rank, session digest
_MAX_MESSAGE = 1 << 40


def parse_peers(peers):
    """Parses ["host:port", ...] (or a comma-separated string) into [(host, port), ...]."""
    if isinstance(peers, str):
        peers = [p for p in peers.split(",") if p.strip()]
    out = []
    for p in peers:
        if isinstance(p, (tuple, list)):
            host, port = p
        else:
            host, _, port = str(p).strip().rpartition(":")
            host = host.strip("[]")
        out.append((host or "127.0.0.1", int(port)))
    return out


def _tls_contexts(tls):
    """Server and client SSL contexts for mutual TLS with a shared CA."""
    certfile, keyfile, cafile = tls["certfile"], tls.get("keyfile"), tls["cafile"]
    server = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=cafile)
    server.load_cert_chain(certfile, keyfile)
    server.verify_mode = ssl.CERT_REQUIRED
    client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
    client.load_cert_chain(certfile, keyfile)
    # Peers are pinned by the shared CA; hostnames are often bare IPs, so hostname checking
    # is opt-in.
    client.check_hostname = bool(tls.get("check_hostname", False))
    for ctx in (server, client):
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return server, client


class _Peer:
    """One connection: outgoing buffer queue and incremental frame parser."""

    def __init__(self, rank, sock):
        self.rank = rank
        self.sock = sock
        self.out = collections.deque()
        self.ready = collections.deque()
        self.eof = False
        self._hdr = bytearray(_LEN.size)
        self._hdr_got = 0
        self._body = None
        self._body_got = 0

    def wants_write(self):
        return bool(self.out)

    def flush_some(self):
        """Sends as much pending data as the socket accepts without blocking."""
        while self.out:
            buf = self.out[0]
            try:
                n = self.sock.send(buf)
            except (BlockingIOError, ssl.SSLWantWriteError, ssl.SSLWantReadError):
                return
            if n == len(buf):
                self.out.popleft()
            else:
                self.out[0] = buf[n:]

    def read_some(self):
        """Reads whatever is available without blocking and parses complete frames."""
        while not self.eof:
            try:
                if self._body is None:
                    n = self.sock.recv_into(memoryview(self._hdr)[self._hdr_got :])
                else:
                    n = self.sock.recv_into(memoryview(self._body)[self._body_got :])
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                return
            except (ConnectionResetError, BrokenPipeError, ssl.SSLEOFError, ssl.SSLZeroReturnError):
                n = 0
            if n == 0:
                self.eof = True
                return
            if self._body is None:
                self._hdr_got += n
                if self._hdr_got == len(self._hdr):
                    length = _LEN.unpack(self._hdr)[0]
                    if length > _MAX_MESSAGE:
                        raise ConnectionError(f"PriX TCP: message from rank {self.rank} is too large")
                    self._hdr_got = 0
                    self._body = bytearray(length)
                    self._body_got = 0
                    if length == 0:
                        self.ready.append(self._body)
                        self._body = None
            else:
                self._body_got += n
                if self._body_got == len(self._body):
                    self.ready.append(self._body)
                    self._body = None

    def pending_tls(self):
        return isinstance(self.sock, ssl.SSLSocket) and self.sock.pending() > 0


class TCPCommunicator(DistributedCommunicator):
    """
    Direct TCP (optionally TLS) communicator between all PriX processes.

    Create it through `prix.init(rank, world_size, transport="tcp", peers=[...], tls=...)`.
    """

    _instance = None
    _TIMEOUT_S = 600.0
    _CONNECT_TIMEOUT_S = 120.0

    @classmethod
    def initialize(
        cls,
        rank,
        world_size,
        evaluator_size=0,
        init_ttp=False,
        peers=None,
        tls=None,
        session=None,
        timeout=None,
    ):
        if cls._instance is not None:
            return
        cls._instance = cls(
            rank, world_size, evaluator_size=evaluator_size, init_ttp=init_ttp,
            peers=peers, tls=tls, session=session, timeout=timeout,
        )

    def __init__(self, rank, world_size, evaluator_size=0, init_ttp=False, peers=None,
                 tls=None, session=None, timeout=None):
        super().__init__(rank, world_size, evaluator_size=evaluator_size, init_ttp=init_ttp)
        peers = peers if peers is not None else os.environ.get("PRIX_PEERS")
        if not peers:
            raise ValueError("transport='tcp' needs `peers`: one 'host:port' per process (or PRIX_PEERS)")
        self.addresses = parse_peers(peers)
        total = world_size + self.ttp_offset + evaluator_size
        if len(self.addresses) != total:
            raise ValueError(
                f"transport='tcp' needs {total} peer addresses (parties, TTP, evaluators); "
                f"got {len(self.addresses)}"
            )
        if timeout is not None:
            self._TIMEOUT_S = float(timeout)
        session = session if session is not None else os.environ.get("PRIX_SESSION", "")
        # Binds connections to this run: peers from another run on the same ports are rejected.
        self._session = hashlib.sha256(
            (str(session) + "|" + ",".join(f"{h}:{p}" for h, p in self.addresses)).encode()
        ).digest()
        self.tls = tls
        if tls is None and any(h not in ("127.0.0.1", "localhost", "::1") for h, _ in self.addresses):
            logging.warning(
                "PriX TCP transport without TLS between hosts: traffic (including PRNG seeds, "
                "dealer messages and output shares) is neither encrypted nor authenticated. "
                "Pass tls={'certfile', 'keyfile', 'cafile'} for anything beyond experiments."
            )
        self._selector = selectors.DefaultSelector()
        self._peers = {}
        self._connect_all(total)

    # ------------------------------------------------------------------ setup
    def _connect_all(self, total):
        server_ctx, client_ctx = _tls_contexts(self.tls) if self.tls else (None, None)
        host, port = self.addresses[self.rank]
        bind_host = os.environ.get("PRIX_BIND_HOST", host)
        listener = socket.create_server((bind_host, port), reuse_port=False, backlog=total)
        listener.settimeout(1.0)
        deadline = time.monotonic() + self._CONNECT_TIMEOUT_S
        expected_accepts = set(range(self.rank + 1, total))
        to_connect = list(range(self.rank))
        try:
            # Connect to lower ranks, accept from higher ranks; interleave so that no
            # ordering of process start-up can deadlock.
            while to_connect or expected_accepts:
                if time.monotonic() > deadline:
                    missing = sorted(set(to_connect) | expected_accepts)
                    raise TimeoutError(f"PriX TCP: rank {self.rank} could not reach ranks {missing}")
                if to_connect:
                    dst = to_connect[0]
                    sock = self._try_connect(dst, client_ctx)
                    if sock is not None:
                        self._add_peer(dst, sock)
                        to_connect.pop(0)
                        continue
                if expected_accepts:
                    try:
                        conn, _ = listener.accept()
                    except socket.timeout:
                        continue
                    src = self._accept_hello(conn, server_ctx, expected_accepts)
                    if src is not None:
                        expected_accepts.discard(src)
                elif to_connect:
                    time.sleep(0.05)
        finally:
            listener.close()

    def _try_connect(self, dst, client_ctx):
        try:
            sock = socket.create_connection(self.addresses[dst], timeout=2.0)
        except OSError:
            return None
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if client_ctx is not None:
                sock = client_ctx.wrap_socket(sock, server_hostname=self.addresses[dst][0])
            sock.settimeout(self._CONNECT_TIMEOUT_S)
            sock.sendall(_HELLO.pack(_MAGIC, self.rank, self._session))
            ack = self._recv_exact(sock, 1)
            if ack != b"\x01":
                raise ConnectionError(f"PriX TCP: rank {dst} rejected the connection")
            return sock
        except (OSError, ssl.SSLError):
            sock.close()
            raise

    def _accept_hello(self, conn, server_ctx, expected):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(10.0)
            if server_ctx is not None:
                conn = server_ctx.wrap_socket(conn, server_side=True)
            magic, src, session = _HELLO.unpack(self._recv_exact(conn, _HELLO.size))
            if magic != _MAGIC or session != self._session or src not in expected:
                logging.warning(f"PriX TCP: rank {self.rank} rejected a connection (claimed rank {src})")
                conn.sendall(b"\x00")
                conn.close()
                return None
            conn.sendall(b"\x01")
            self._add_peer(src, conn)
            return src
        except (OSError, ssl.SSLError, struct.error) as e:
            logging.warning(f"PriX TCP: rank {self.rank} dropped a bad incoming connection: {e}")
            conn.close()
            return None

    @staticmethod
    def _recv_exact(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PriX TCP: connection closed during handshake")
            buf += chunk
        return bytes(buf)

    def _add_peer(self, rank, sock):
        sock.setblocking(False)
        for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            try:
                sock.setsockopt(socket.SOL_SOCKET, opt, 8 << 20)
            except OSError:
                pass
        peer = _Peer(rank, sock)
        self._peers[rank] = peer
        self._selector.register(sock, selectors.EVENT_READ, peer)

    # --------------------------------------------------------------- transport
    def _progress(self, until, timeout):
        """Moves data in both directions until `until()` is true."""
        deadline = time.monotonic() + timeout
        while True:
            for peer in self._peers.values():
                peer.flush_some()
                peer.read_some()
            if until():
                return
            for peer in self._peers.values():
                events = selectors.EVENT_READ | (selectors.EVENT_WRITE if peer.wants_write() else 0)
                self._selector.modify(peer.sock, events, peer)
            if any(p.pending_tls() for p in self._peers.values()):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"PriX TCP: rank {self.rank} timed out waiting for peers")
            self._selector.select(min(remaining, 1.0))

    def _kv_put(self, src, dst, payload):
        parts = wire.encode(payload)
        peer = self._peer(dst)
        peer.out.append(_LEN.pack(sum(len(p) for p in parts)))
        peer.out.extend(p if isinstance(p, memoryview) else memoryview(p) for p in parts)
        peer.flush_some()

    def _kv_get(self, src, dst):
        peer = self._peer(src)

        def ready():
            if peer.ready:
                return True
            if peer.eof:
                raise ConnectionError(f"PriX TCP: rank {src} closed the connection")
            return False

        self._progress(ready, self._TIMEOUT_S)
        return wire.decode(peer.ready.popleft())

    def _peer(self, rank):
        try:
            return self._peers[rank]
        except KeyError:
            raise ValueError(f"PriX TCP: rank {self.rank} has no connection to rank {rank}") from None

    # Used by the Fission fast path (approximations.permute_reveal_evaluate_share).
    def _put_to_mailbox(self, dst, payload):
        self._kv_put(self.rank, dst, payload)

    def _get_from_mailbox(self, src=None):
        if src is None:
            raise ValueError("TCPCommunicator requires an explicit source rank")
        return src, self._kv_get(src, self.rank)

    def flush(self):
        """Blocks until all queued outgoing data has been handed to the kernel."""
        self._progress(lambda: not any(p.out for p in self._peers.values()), self._TIMEOUT_S)

    # ------------------------------------------------------------ collectives
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _default_device():
        return jax.devices()[0]

    @classmethod
    def _get_device(cls, array):
        if hasattr(array, "share"):
            array = array.share
        if isinstance(array, jnp.ndarray):
            devs = array.devices()
            if len(devs) == 1:
                dev = next(iter(devs))
                if dev != cls._default_device():
                    return dev
        return None

    @staticmethod
    def _np_reduce(arrays, op):
        if op == "sum":
            return functools.reduce(np.add, arrays)
        if op in ("xor", "bxor"):
            return functools.reduce(np.bitwise_xor, arrays)
        raise NotImplementedError(f"Unsupported reduction op: {op}")

    def _device_put(self, arr, device):
        if isinstance(arr, np.ndarray):
            # device_put is about 2x faster than jnp.asarray for large host arrays.
            return jax.device_put(arr, device)
        return arr

    def _host_payload(self, array, batched):
        if batched:
            raw = [a.share if hasattr(a, "share") else a for a in array]
            return [np.asarray(x) for x in jax.device_get(raw)]
        return self._to_transport(array)

    @_logging
    def all_reduce(self, array, op="sum", batched=False, async_op=False):
        payload = self._host_payload(array, batched)
        for dst in range(self.world_size):
            if dst != self.rank:
                self._kv_put(self.rank, dst, payload)
        received = [payload if r == self.rank else self._kv_get(r, self.rank) for r in range(self.world_size)]
        if batched:
            dev = self._get_device(array[0]) if len(array) else None
            return [
                self._device_put(self._np_reduce([received[r][k] for r in range(self.world_size)], op), dev)
                for k in range(len(payload))
            ]
        return self._device_put(self._np_reduce(received, op), self._get_device(array))

    @_logging
    def reduce(self, array, dst, op="sum", batched=False, async_op=False):
        payload = self._host_payload(array, batched)
        if self.rank != dst:
            self._kv_put(self.rank, dst, payload)
            return None
        received = [payload if r == self.rank else self._kv_get(r, self.rank) for r in range(self.world_size)]
        if batched:
            dev = self._get_device(array[0]) if len(array) else None
            return [
                self._device_put(self._np_reduce([received[r][k] for r in range(self.world_size)], op), dev)
                for k in range(len(payload))
            ]
        return self._device_put(self._np_reduce(received, op), self._get_device(array))

    @_logging
    def recv(self, array=None, src=None):
        if src is None:
            raise ValueError("TCPCommunicator.recv requires an explicit `src` rank.")
        return self._device_put(self._kv_get(src, self.rank), self._get_device(array))

    # ---------------------------------------------------------------- teardown
    @classmethod
    def shutdown(cls):
        inst = cls._instance
        cls._instance = None
        if inst is not None:
            inst.close()

    def close(self, linger=5.0):
        """Flushes pending data, then closes every connection gracefully."""
        try:
            self.flush()
        except (OSError, TimeoutError, ConnectionError) as e:
            logging.warning(f"PriX TCP: rank {self.rank} could not flush on shutdown: {e}")
        for peer in self._peers.values():
            try:
                peer.sock.shutdown(socket.SHUT_WR)
            except (OSError, ssl.SSLError, ValueError):
                pass
        # Wait (briefly) for peers to finish too, so unread data never triggers a reset that
        # could discard a message the peer has not yet read.
        deadline = time.monotonic() + linger
        while time.monotonic() < deadline and not all(p.eof for p in self._peers.values()):
            for peer in self._peers.values():
                try:
                    peer.read_some()
                except (OSError, ConnectionError, ssl.SSLError):
                    peer.eof = True
            self._selector.select(0.05)
        for peer in self._peers.values():
            try:
                self._selector.unregister(peer.sock)
            except (KeyError, ValueError):
                pass
            peer.sock.close()
        self._selector.close()
        self._peers = {}
