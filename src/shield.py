"""Partitioned execution: run chosen modules somewhere else, transparently.

Step 1 of the FPGA ladder. The point is NOT speed -- it is to prove the cut is
numerically invisible before any hardware exists. A partition that changes the
output by one ULP is a partition whose later FPGA kernel cannot be validated,
because you no longer have an oracle.

Three transports share one stub, so the model code never changes:

  LocalTransport    calls the real module in-process. Isolates ONE question:
                    did I cut the graph in the right place?
  SocketTransport   same cut, bytes on a socket. Isolates serialization:
                    dtype, contiguity, shape, partial recv.
  (later)           /dev/xdma0_* file descriptors -- same interface again.

Each rung fails for exactly one reason, which is the entire reason there are
three of them rather than one.
"""
import socket
import struct

import torch
from torch import nn

# ---------------------------------------------------------------------------
# wire format: everything is explicit, because a silent dtype or layout change
# is precisely the bug this module exists to rule out.
# ---------------------------------------------------------------------------

_DTYPES = {0: torch.float32, 1: torch.float16, 2: torch.int8}
_CODES = {v: k for k, v in _DTYPES.items()}


def _sendall(sock, buf):
    sock.sendall(buf)


def _recvall(sock, n):
    """socket.recv() may return short reads; a tensor payload is megabytes."""
    out = bytearray(n)
    view, got = memoryview(out), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError(f"peer closed after {got} of {n} bytes")
        got += k
    return bytes(out)


def pack(tag, t):
    """(tag, tensor) -> bytes. .contiguous() matters: a transposed view's
    storage order is not its logical order, and .numpy() would ship the wrong
    bytes without complaint."""
    t = t.detach().contiguous().cpu()
    tb = tag.encode()
    body = t.numpy().tobytes()
    hdr = struct.pack("<HBB", len(tb), t.dim(), _CODES[t.dtype])
    hdr += struct.pack(f"<{t.dim()}i", *t.shape)
    hdr += struct.pack("<Q", len(body))
    return hdr + tb + body


def recv(sock):
    """bytes on a socket -> (tag, tensor)."""
    taglen, ndim, dcode = struct.unpack("<HBB", _recvall(sock, 4))
    shape = struct.unpack(f"<{ndim}i", _recvall(sock, 4 * ndim))
    (nbytes,) = struct.unpack("<Q", _recvall(sock, 8))
    tag = _recvall(sock, taglen).decode()
    buf = _recvall(sock, nbytes)
    t = torch.frombuffer(bytearray(buf), dtype=_DTYPES[dcode]).reshape(shape)
    return tag, t


class LocalTransport:
    """Rung 1a. No socket, no bytes -- just a function call."""

    def __init__(self, modules):
        self.modules = modules

    def call(self, tag, x):
        return self.modules[tag](x)


class SocketTransport:
    """Rungs 1b and 2. Loopback or Ethernet; the client cannot tell."""

    def __init__(self, host, port, device="cuda"):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device = device

    def call(self, tag, x):
        _sendall(self.sock, pack(tag, x))
        _, y = recv(self.sock)
        return y.to(self.device)

    def close(self):
        self.sock.close()


class Remote(nn.Module):
    """Stand-in for a module that now lives elsewhere."""

    def __init__(self, tag, transport):
        super().__init__()
        self.tag, self.transport = tag, transport

    def forward(self, x):
        return self.transport.call(self.tag, x)


# ---------------------------------------------------------------------------
# the surgery
# ---------------------------------------------------------------------------

# BaseModel._predict_once routes the graph with each module's `.f` (from) and
# `.i` (index), and reads `.type`/`.np` for printing. A bare nn.Module has none
# of them, so a naive swap raises AttributeError or -- worse -- silently
# reroutes. Carry them across.
_GRAPH_ATTRS = ("f", "i", "type", "np")


def split(core, tags):
    """Replace each named module with a Remote stub.

    `tags` are dotted paths relative to `core`, e.g. "model.0" or "model.22.dfl".
    Returns {tag: real module} for the server to hold.
    """
    taken = {}
    for tag in tags:
        parent, _, leaf = tag.rpartition(".")
        owner = core.get_submodule(parent) if parent else core
        real = getattr(owner, leaf) if not leaf.isdigit() else owner[int(leaf)]
        stub = Remote(tag, None)
        for a in _GRAPH_ATTRS:
            if hasattr(real, a):
                setattr(stub, a, getattr(real, a))
        if leaf.isdigit():
            owner[int(leaf)] = stub
        else:
            setattr(owner, leaf, stub)
        taken[tag] = real
    return taken


def attach(core, transport, tags=None):
    """Point Remote stubs at a transport.

    tags=None routes every stub to the same place, which is what you want when
    one device holds all the shielded modules. Pass a list to route selectively,
    so different tags can land on different destinations:

        attach(core, fpga,   tags=["model.0"])
        attach(core, enclave, tags=["model.22.dfl"])
    """
    n = 0
    for m in core.modules():
        if isinstance(m, Remote) and (tags is None or m.tag in tags):
            m.transport = transport
            n += 1
    return n


def serve(modules, host="0.0.0.0", port=9999, device="cuda"):
    """Hold the real modules; answer calls until the client disconnects."""
    modules = {k: v.to(device).eval() for k, v in modules.items()}
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"shield server on {host}:{port}, device={device}, "
          f"holding {list(modules)}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  client {addr}", flush=True)
        try:
            while True:
                tag, x = recv(conn)
                with torch.no_grad():
                    y = modules[tag](x.to(device))
                _sendall(conn, pack(tag, y))
        except (ConnectionError, struct.error):
            print("  client gone", flush=True)
        finally:
            conn.close()
