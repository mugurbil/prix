#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pickle-free wire format for PriX messages.

Only plain data can be sent: None, bool, int, float, str, bytes, lists, tuples, dicts, and
numeric NumPy/JAX arrays and scalars. Decoding never executes code, so a malicious peer can
at worst send wrong values, not take over the receiving process.
"""

import struct

import numpy as np

_ALLOWED_KINDS = frozenset("biufc")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")
_MAX_DEPTH = 64


def _as_numpy(obj):
    """Returns obj as a NumPy array if it is a NumPy or JAX array, else None."""
    if isinstance(obj, np.ndarray):
        return obj
    # JAX arrays (and anything else exposing the array protocol with a dtype/shape).
    if hasattr(obj, "__array__") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        return np.asarray(obj)
    return None


def encode(obj):
    """Encodes obj into a list of buffers (bytes or memoryviews) without copying array data."""
    parts = []
    size = [0]  # running byte offset, used to 8-byte-align array data

    def put(b):
        parts.append(b)
        size[0] += len(b)

    def enc(o, depth):
        if depth > _MAX_DEPTH:
            raise ValueError("PriX wire: object nested too deeply")
        if o is None:
            put(b"N")
        elif o is True:
            put(b"T")
        elif o is False:
            put(b"F")
        elif isinstance(o, int):
            if -(2**63) <= o < 2**63:
                put(b"i" + _I64.pack(o))
            else:
                raw = o.to_bytes((o.bit_length() + 8) // 8, "little", signed=True)
                put(b"I" + _U32.pack(len(raw)) + raw)
        elif isinstance(o, float):
            put(b"f" + _F64.pack(o))
        elif isinstance(o, str):
            raw = o.encode("utf-8")
            put(b"s" + _U32.pack(len(raw)) + raw)
        elif isinstance(o, (bytes, bytearray)):
            put(b"y" + _U32.pack(len(o)))
            put(bytes(o))
        elif isinstance(o, np.generic):
            enc_array(np.asarray(o), b"g")
        elif isinstance(o, (list, tuple)):
            put((b"l" if isinstance(o, list) else b"t") + _U32.pack(len(o)))
            for item in o:
                enc(item, depth + 1)
        elif isinstance(o, dict):
            put(b"d" + _U32.pack(len(o)))
            for k, v in o.items():
                enc(k, depth + 1)
                enc(v, depth + 1)
        else:
            arr = _as_numpy(o)
            if arr is None:
                raise TypeError(f"PriX wire: cannot send objects of type {type(o).__name__}")
            enc_array(arr, b"a")

    def enc_array(arr, tag):
        if arr.dtype.kind not in _ALLOWED_KINDS:
            raise TypeError(f"PriX wire: cannot send arrays of dtype {arr.dtype}")
        if not arr.flags.c_contiguous:  # (np.ascontiguousarray would turn 0-d into 1-d)
            arr = arr.copy(order="C")
        dt = arr.dtype.str.encode("ascii")
        header = tag + bytes([len(dt)]) + dt + bytes([arr.ndim]) + b"".join(_U64.pack(d) for d in arr.shape)
        header += _U64.pack(arr.nbytes)
        pad = (-(size[0] + len(header) + 1)) % 8
        put(header + bytes([pad]) + b"\0" * pad)
        if arr.nbytes:
            put(memoryview(arr).cast("B"))

    enc(obj, 0)
    return parts


def decode(buf):
    """Decodes a message produced by encode(). `buf` must be a bytearray (arrays alias it)."""
    view = memoryview(buf)
    pos = [0]

    def take(n):
        p = pos[0]
        if p + n > len(view):
            raise ValueError("PriX wire: truncated message")
        pos[0] = p + n
        return view[p : p + n]

    def u32():
        return _U32.unpack(take(4))[0]

    def dec(depth):
        if depth > _MAX_DEPTH:
            raise ValueError("PriX wire: object nested too deeply")
        tag = bytes(take(1))
        if tag == b"N":
            return None
        if tag == b"T":
            return True
        if tag == b"F":
            return False
        if tag == b"i":
            return _I64.unpack(take(8))[0]
        if tag == b"I":
            return int.from_bytes(take(u32()), "little", signed=True)
        if tag == b"f":
            return _F64.unpack(take(8))[0]
        if tag == b"s":
            return str(take(u32()), "utf-8")
        if tag == b"y":
            return bytes(take(u32()))
        if tag in (b"l", b"t"):
            items = [dec(depth + 1) for _ in range(u32())]
            return items if tag == b"l" else tuple(items)
        if tag == b"d":
            out = {}
            for _ in range(u32()):
                k = dec(depth + 1)
                out[k] = dec(depth + 1)
            return out
        if tag in (b"a", b"g"):
            dt = np.dtype(str(take(take(1)[0]), "ascii"))
            if dt.kind not in _ALLOWED_KINDS:
                raise ValueError(f"PriX wire: refusing dtype {dt}")
            ndim = take(1)[0]
            shape = tuple(_U64.unpack(take(8))[0] for _ in range(ndim))
            nbytes = _U64.unpack(take(8))[0]
            take(take(1)[0])  # alignment padding
            if nbytes != dt.itemsize * int(np.prod(shape, dtype=np.int64)):
                raise ValueError("PriX wire: array size does not match its shape")
            start = pos[0]
            take(nbytes)
            arr = np.frombuffer(buf, dtype=dt, count=nbytes // dt.itemsize, offset=start).reshape(shape)
            return arr[()] if tag == b"g" else arr
        raise ValueError(f"PriX wire: unknown tag {tag!r}")

    obj = dec(0)
    if pos[0] != len(view):
        raise ValueError("PriX wire: trailing bytes in message")
    return obj
