#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

import prix.communicator as comm
from prix.config import cfg
from prix.encoder import DATATYPE, FixedPointEncoder, decode, encode, resolve_device, to_jax
from prix.provider import _eval_ring_op, _int64_matmul, get_default_provider
from prix.random import (
    _przs_jit,
    count_wraps,
    generate_kbit_random_tensor,
    generate_random_ring_element,
)

SENTINEL = -1


class Ptype(str, Enum):
    arithmetic = "arithmetic"
    binary = "binary"


ptype = Ptype


class _EncoderProxy:
    """Compatibility proxy exposing CrypTen/Curl `.encoder` attributes on `Array`."""

    def __init__(self, array_obj):
        self._array = array_obj

    @property
    def precision_bits(self):
        return self._array.precision

    @precision_bits.setter
    def precision_bits(self, val):
        self._array.precision = int(val) if val is not None else cfg.encoder.precision_bits

    @property
    def _precision_bits(self):
        return self._array.precision

    @_precision_bits.setter
    def _precision_bits(self, val):
        self._array.precision = int(val) if val is not None else cfg.encoder.precision_bits

    @property
    def scale(self):
        return int(2 ** self._array.precision)

    @scale.setter
    def scale(self, val):
        self._array.precision = int(round(math.log2(max(1, val))))

    @property
    def _scale(self):
        return self.scale

    @_scale.setter
    def _scale(self, val):
        self.scale = val

    def encode(self, x, device=None):
        return encode(x, precision=self._array.precision, device=device or self._array.device)

    def decode(self, x):
        return decode(x, precision=self._array.precision)


class IgnoreEncodings:
    """Context manager to temporarily ignore fixed-point encodings (set precision=0)."""

    def __init__(self, list_of_tensors):
        self.list_of_tensors = [t for t in list_of_tensors if isinstance(t, Array)]
        self.encodings_cache = [t.precision for t in self.list_of_tensors]

    def __enter__(self):
        for t in self.list_of_tensors:
            t.precision = 0

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for i, t in enumerate(self.list_of_tensors):
            t.precision = self.encodings_cache[i]


class Array:
    """
    Secret-shared tensor object in PriX supporting both additive arithmetic
    sharing (`ptype="arithmetic"`) and XOR binary sharing (`ptype="binary"`).
    """

    # Make NumPy defer to our reflected operators (np_array @ x, np_array + x, ...).
    # Without this, NumPy treats an Array as a sequence (via __len__/__getitem__) and
    # applies the operator element by element, which is wrong and can desync parties.
    __array_ufunc__ = None

    def __init__(
        self,
        array=None,
        src: int = 0,
        precision: int = None,
        device=None,
        ptype=Ptype.arithmetic,
        size=None,
        broadcast_size: bool = False,
        requires_grad: bool = False,
    ):
        self.ptype = Ptype(ptype)
        if precision is None:
            self.precision = 0 if self.ptype == Ptype.binary else int(cfg.encoder.precision_bits)
        else:
            self.precision = int(precision)
        self.device = device
        self._is_parameter = False
        self._plain_encoded = None

        if src == SENTINEL:
            self.share = None
            return

        if isinstance(array, list) and len(array) == 0:
            self.share = jnp.array([], dtype=DATATYPE)
            if device is not None:
                self.share = jax.device_put(self.share, resolve_device(device))
            return

        if isinstance(array, Array):
            self.share = array.share.copy()
            self.precision = array.precision if precision is None else int(precision)
            self.device = array.device if device is None else device
            self.ptype = array.ptype
            self._is_parameter = getattr(array, "_is_parameter", False)
            self._plain_encoded = getattr(array, "_plain_encoded", None)
            return

        world_size = comm.get().get_world_size()
        assert (
            isinstance(src, int) and 0 <= src < world_size
        ), f"Specified source party {src} does not exist (world_size={world_size})"

        rank = comm.get().get_rank()
        if rank == src:
            assert array is not None or size is not None, "Source must provide data array or size"

        if device is None and array is not None and hasattr(array, "device"):
            device = array.device
            self.device = device

        enc_precision = 0 if self.ptype == Ptype.binary else self.precision

        if not broadcast_size:
            assert (
                array is not None or size is not None
            ), "Must specify array or size, or set broadcast_size=True"

        if array is not None:
            encoded = encode(array, precision=enc_precision, device=device)
            size = encoded.shape
        else:
            encoded = None
            if size is not None:
                size = tuple(size)

        if broadcast_size:
            size = comm.get().broadcast_obj(size, src)

        self.share = Array.PRZS(size, device=device, ptype=self.ptype).share
        if rank == src and encoded is not None:
            if self.ptype == Ptype.binary:
                self.share = self.share ^ encoded
            else:
                self.share = self.share + encoded

    @staticmethod
    def new(*args, **kwargs):
        return Array(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, src=0, device=None, ptype=Ptype.arithmetic):
        result = Array(src=SENTINEL, precision=precision, device=device, ptype=ptype)
        result.share = to_jax(share, device=device, dtype=DATATYPE)
        return result

    @staticmethod
    def PRZS(*size, device=None, ptype=Ptype.arithmetic):
        """Generate a Pseudo-Random Sharing of Zero across all parties."""
        import prix

        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        result = Array(src=SENTINEL, precision=(0 if ptype == Ptype.binary else None), device=device, ptype=ptype)
        g0 = prix.generators.get("prev")
        g1 = prix.generators.get("next")
        if g0 is not None and g1 is not None:
            if jax.default_backend() == "gpu":
                shape = tuple(size)
                n = int(np.prod(shape)) if len(shape) > 0 else 1
                s0, c0 = g0.next_offset(n)
                s1, c1 = g1.next_offset(n)
                diff = _przs_jit(c0, c1, s0, s1, shape, ptype == Ptype.binary)
                result.share = to_jax(diff, device=device, dtype=DATATYPE)
            else:
                u0 = g0.next_uint64(size)
                u1 = g1.next_uint64(size)
                diff = (u0 ^ u1) if ptype == Ptype.binary else (u0 - u1)
                result.share = to_jax(diff.view(np.int64), device=device, dtype=DATATYPE)
        elif ptype == Ptype.binary:
            s0 = generate_kbit_random_tensor(size, bitlength=64, generator=g0, device=device)
            s1 = generate_kbit_random_tensor(size, bitlength=64, generator=g1, device=device)
            result.share = s0 ^ s1
        else:
            s0 = generate_random_ring_element(size, generator=g0, device=device)
            s1 = generate_random_ring_element(size, generator=g1, device=device)
            result.share = s0 - s1
        return result

    @staticmethod
    def PRSS(*size, device=None):
        """Generate a Pseudo-Random Secret Share from local random ring elements."""
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        share = generate_random_ring_element(size, device=device)
        return Array.from_shares(share, precision=None, device=device, ptype=Ptype.arithmetic)

    @staticmethod
    def rand(*sizes, bits=None, device=None, ptype=Ptype.arithmetic):
        """Generate uniform random encrypted Array in [0, 1) (arithmetic) or k-bit (binary)."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        precision_bits = cfg.encoder.precision_bits if bits is None else bits
        bin_arr = Array(src=SENTINEL, precision=precision_bits, device=device, ptype=Ptype.binary)
        bin_arr.share = generate_kbit_random_tensor(sizes, bitlength=precision_bits, device=device)
        if Ptype(ptype) == Ptype.binary:
            return bin_arr
        return bin_arr.to(Ptype.arithmetic, precision=precision_bits, bits=precision_bits)

    @property
    def rank(self):
        return comm.get().get_rank()

    @property
    def scale(self):
        return int(2 ** self.precision)

    @property
    def encoder(self):
        return _EncoderProxy(self)

    @encoder.setter
    def encoder(self, value):
        if isinstance(value, (FixedPointEncoder, _EncoderProxy)):
            self.precision = value.precision_bits
        elif isinstance(value, int):
            self.precision = value

    @property
    def shape(self):
        return self.share.shape

    @property
    def ndim(self):
        return self.share.ndim

    @property
    def dtype(self):
        return self.share.dtype

    @property
    def data(self):
        return self.share

    @data.setter
    def data(self, value):
        self.share = jnp.asarray(value, dtype=DATATYPE)

    def copy(self):
        result = Array(src=SENTINEL, precision=self.precision, device=self.device, ptype=self.ptype)
        result.share = self.share
        return result

    def clone(self):
        return self.copy()

    def shallow_copy(self):
        return self.copy()

    def copy_(self, other):
        self.share = other.share
        self.precision = other.precision
        self.ptype = other.ptype
        self.device = other.device
        return self

    def __repr__(self):
        if getattr(cfg.debug, "debug_mode", False) and comm.is_initialized():
            plain = self.get_plain_text()
            return f"Array(share={self.share}, plain_text={plain}, ptype={self.ptype})"
        return f"Array(share={self.share}, precision={self.precision}, ptype={self.ptype})"

    def __bool__(self):
        raise RuntimeError("Cannot evaluate encrypted Arrays to boolean values")

    # Reveal and Reconstruction
    @staticmethod
    def reveal_batch(tensor_or_list, dst=None):
        if isinstance(tensor_or_list, Array):
            return tensor_or_list.reveal(dst=dst)
        assert isinstance(tensor_or_list, list), f"Invalid input type: {type(tensor_or_list)}"
        if not tensor_or_list:
            return []
        op = "bxor" if tensor_or_list[0].ptype == Ptype.binary else "sum"
        shares = [t.share for t in tensor_or_list]
        if dst is None:
            return comm.get().all_reduce(shares, op=op, batched=True)
        return comm.get().reduce(shares, dst, op=op, batched=True)

    def reveal(self, dst=None):
        """Reveals the raw ring element without fixed-point downscaling."""
        if self.share is None or self.share.size < 1:
            return jnp.empty(self.share.shape if self.share is not None else (0,), dtype=DATATYPE)
        op = "bxor" if self.ptype == Ptype.binary else "sum"
        if dst is None:
            return comm.get().all_reduce(self.share, op=op)
        return comm.get().reduce(self.share, dst, op=op)

    def reconstruct(self, dst=None):
        """Decrypts and decodes the shared Array."""
        if self.share is not None and self.share.size < 1:
            return jnp.empty(self.share.shape, dtype=jnp.float32)
        revealed = self.reveal(dst=dst)
        return decode(revealed, self.precision)

    def get_plain_text(self, dst=None):
        """Alias for reconstruct(dst=dst) for CrypTen/Curl compatibility."""
        return self.reconstruct(dst=dst)

    # Rescaling encodings
    def encode_(self, new_precision):
        if isinstance(new_precision, (FixedPointEncoder, _EncoderProxy)):
            new_precision = new_precision.precision_bits
        new_precision = int(new_precision)
        if self.precision == new_precision:
            return self
        elif self.precision < new_precision:
            scale_factor = 2 ** (new_precision - self.precision)
            self.share = self.share * scale_factor
        else:
            scale_factor = 2 ** (self.precision - new_precision)
            self.div_(scale_factor)
        self.precision = new_precision
        return self

    def encode_as_(self, other):
        return self.encode_(other.precision)

    def encode_as(self, other):
        return self.clone().encode_(other.precision)

    # Type / Device conversion (A2B and B2A)
    def to(self, *args, **kwargs):
        if "ptype" in kwargs:
            target_ptype = kwargs.pop("ptype")
            return self._to_ptype(target_ptype, **kwargs)
        elif args and (isinstance(args[0], Ptype) or args[0] in ("arithmetic", "binary")):
            return self._to_ptype(args[0], **kwargs)
        else:
            device = args[0] if args else kwargs.get("device", self.device)
            if device is not None:
                self.device = device
                self.share = jax.device_put(self.share, resolve_device(device))
            return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, *args, **kwargs):
        return self.to("cuda")

    def _to_ptype(self, target_ptype, **kwargs):
        target_ptype = Ptype(target_ptype)
        if self.ptype == target_ptype:
            return self.clone()
        if self.ptype == Ptype.arithmetic and target_ptype == Ptype.binary:
            return self._A2B()
        elif self.ptype == Ptype.binary and target_ptype == Ptype.arithmetic:
            return self._B2A(**kwargs)
        raise TypeError(f"Cannot convert {self.ptype} to {target_ptype}")

    def _A2B(self):
        world_size = comm.get().get_world_size()
        if world_size == 2:
            from prix import circuit

            provider = get_default_provider()
            s_shape = self.size()
            a, b, c = provider.generate_binary_triple(s_shape, s_shape, device=self.device)
            is_rank0 = self.rank == 0
            ed_share = _a2b_pre_and(self.share, a.share, b.share, is_rank0)
            ed = comm.get().all_reduce(ed_share, op="bxor")
            c.share = _binary_and_combine(c.share, ed[0], ed[1], a.share, b.share, is_rank0)
            P = Array.from_shares(self.share, precision=0, device=self.device, ptype=Ptype.binary)
            carry, _ = circuit._spk_circuit(c, P)
            binary_tensor = P.shallow_copy()
            binary_tensor.share = _a2b_finalize(P.share, carry.share)
        else:
            shares = [
                Array(self.share, src=i, precision=0, device=self.device, ptype=Ptype.binary)
                for i in range(world_size)
            ]
            binary_tensor = Array.stack(shares).sum(dim=0)
        binary_tensor.precision = self.precision
        return binary_tensor

    def _B2A(self, precision=None, bits=None):
        if bits is None:
            bits = 64
        if bits == 1:
            binary_bit = self & 1
            arithmetic_tensor = _b2a_single_bit(binary_bit)
        else:
            shifts = jnp.arange(bits, dtype=DATATYPE).reshape((bits,) + (1,) * self.dim())
            binary_bits = Array(src=SENTINEL, precision=0, device=self.device, ptype=Ptype.binary)
            binary_bits.share = (self.share[None, ...] >> shifts) & jnp.int64(1)
            arithmetic_bits = _b2a_single_bit(binary_bits)

            u_shifts = jnp.arange(bits, dtype=jnp.uint64)
            multiplier = jax.lax.bitcast_convert_type(jnp.uint64(1) << u_shifts, DATATYPE)
            multiplier = multiplier.reshape((bits,) + (1,) * self.dim())
            arithmetic_bits.share = arithmetic_bits.share * multiplier
            arithmetic_tensor = arithmetic_bits.sum(dim=0)

        target_prec = cfg.encoder.precision_bits if precision is None else int(precision)
        arithmetic_tensor.precision = target_prec
        if target_prec > self.precision:
            scale = 2 ** (target_prec - self.precision)
            arithmetic_tensor.share = arithmetic_tensor.share * scale
        return arithmetic_tensor

    # Arithmetic / Beaver operations
    def _arithmetic_function_(self, y, op, *args, **kwargs):
        return self._arithmetic_function(y, op, inplace=True, *args, **kwargs)

    def _arithmetic_function(self, y, op, inplace=False, *args, **kwargs):
        assert op in (
            "add",
            "sub",
            "mul",
            "matmul",
            "conv1d",
            "conv2d",
        ), f"Provided op `{op}` is not a supported arithmetic function"

        additive_func = op in ("add", "sub")
        private = isinstance(y, Array)
        public = not private

        result = self if inplace else self.clone()

        if public:
            if isinstance(y, (int, float, np.number)):
                if additive_func:
                    if result.rank == 0:
                        y_int = int(round((1 << result.precision) * float(y)))
                        if y_int != 0:
                            result.share = _add_scalar_jit(
                                result.share, y_int if op == "add" else -y_int
                            )
                    return result
                if (
                    op == "mul"
                    and cfg.encoder.trunc_method.prod == "crypten"
                    and comm.get().get_world_size() <= 2
                ):
                    y_int = int(round((1 << result.precision) * float(y)))
                    trunc_div = (1 << result.precision) if self.precision > 0 else 1
                    result.share = _mul_div_scalar_jit(result.share, y_int, trunc_div)
                    return result

            if (
                not additive_func
                and cfg.encoder.trunc_method.prod == "fission"
                and result.precision > cfg.encoder.precision_bits
            ):
                result = result.egk_trunc_pr(62, result.precision - cfg.encoder.precision_bits)
                result.precision = cfg.encoder.precision_bits

            y_enc = encode(y, precision=result.precision, device=self.device)

            if additive_func:
                if result.rank == 0:
                    result.share = result.share + y_enc if op == "add" else result.share - y_enc
                else:
                    result.share = jnp.broadcast_arrays(result.share, y_enc)[0]
            else:
                result.share = _eval_ring_op(op, result.share, y_enc, *args, **kwargs)
        else:
            if additive_func:
                y_copy = y.clone()
                if result.precision > y_copy.precision:
                    y_copy.encode_as_(result)
                elif result.precision < y_copy.precision:
                    result.encode_as_(y_copy)
                result.share = (
                    _add_jit(result.share, y_copy.share)
                    if op == "add"
                    else _sub_jit(result.share, y_copy.share)
                )
            else:
                y_arg = y
                if cfg.encoder.trunc_method.prod == "fission":
                    if result.precision > cfg.encoder.precision_bits:
                        result = result.egk_trunc_pr(
                            62, result.precision - cfg.encoder.precision_bits
                        )
                        result.precision = cfg.encoder.precision_bits
                    if y_arg.precision > cfg.encoder.precision_bits:
                        y_arg = y_arg.egk_trunc_pr(
                            62, y_arg.precision - cfg.encoder.precision_bits
                        )
                        y_arg.precision = cfg.encoder.precision_bits
                do_local_trunc = (
                    result.precision > 0
                    and y.precision > 0
                    and cfg.encoder.trunc_method.prod == "crypten"
                    and comm.get().get_world_size() <= 2
                    and op in ("mul", "matmul")
                )
                trunc_div = (1 << result.precision) if do_local_trunc else 1
                beaver_out = _beaver_protocol(
                    op, result, y_arg, *args, _trunc_div=trunc_div, **kwargs
                )
                result.share = beaver_out.share
                if do_local_trunc:
                    return result

        if not additive_func:
            if public:
                if self.precision > 0:
                    method = cfg.encoder.trunc_method.prod
                    if method == "crypten":
                        return result.div_(2 ** result.precision)
                    elif method == "egk":
                        return result.egk_trunc_pr(62, result.precision)
                    elif method == "fission":
                        result.precision = 2 * result.precision
                        return result
                    else:
                        raise ValueError(f"Unsupported truncation method {method}")
                else:
                    result.precision = self.precision
            else:
                if result.precision > 0 and y.precision > 0:
                    method = cfg.encoder.trunc_method.prod
                    if method == "crypten":
                        return result.div_(2 ** result.precision)
                    elif method == "egk":
                        return result.egk_trunc_pr(62, result.precision)
                    elif method == "fission":
                        # Use the precision of the operand actually multiplied: under fission,
                        # y may have been truncated above (y_arg), so y.precision can be stale.
                        result.precision = result.precision + y_arg.precision
                        return result
                    else:
                        raise ValueError(f"Unsupported truncation method {method}")
                elif y.precision > 0:
                    result.precision = y.precision
        return result

    def add(self, y):
        if self.ptype == Ptype.binary:
            from prix import circuit

            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.add(self, y)
        return self._arithmetic_function(y, "add")

    def add_(self, y):
        if self.ptype == Ptype.binary:
            res = self.add(y)
            self.share = res.share
            return self
        return self._arithmetic_function_(y, "add")

    def sub(self, y):
        return self._arithmetic_function(y, "sub")

    def sub_(self, y):
        return self._arithmetic_function_(y, "sub")

    def mul(self, y):
        if isinstance(y, (int, np.integer)):
            result = self.clone()
            result.share = self.share * int(y)
            return result
        return self._arithmetic_function(y, "mul")

    def mul_(self, y):
        if isinstance(y, (int, np.integer)):
            self.share = self.share * int(y)
            return self
        if isinstance(y, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(y.dtype, jnp.integer):
            self.share = self.share * jnp.asarray(y, dtype=DATATYPE)
            return self
        return self._arithmetic_function_(y, "mul")

    def matmul(self, y):
        return _matmul_promoting_1d(self, y, lambda a, b: a._arithmetic_function(b, "matmul"))

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        if isinstance(other, Array):
            return other.matmul(self)
        return _matmul_promoting_1d(other, self, lambda a, b: b._public_rmatmul(a))

    def _public_rmatmul(self, other):
        """Public `other` @ private `self` (both at least 2-D)."""
        # Public @ Private:
        result = self.clone()
        if (
            cfg.encoder.trunc_method.prod == "fission"
            and result.precision > cfg.encoder.precision_bits
        ):
            result = result.egk_trunc_pr(62, result.precision - cfg.encoder.precision_bits)
            result.precision = cfg.encoder.precision_bits
        other_enc = encode(other, precision=result.precision, device=self.device)
        result.share = jnp.matmul(other_enc, result.share)
        if self.precision > 0:
            method = cfg.encoder.trunc_method.prod
            if method == "crypten":
                return result.div_(2 ** result.precision)
            elif method == "egk":
                return result.egk_trunc_pr(62, result.precision)
            elif method == "fission":
                result.precision = 2 * result.precision
                return result
        return result

    def conv1d(self, kernel, **kwargs):
        return self._arithmetic_function(kernel, "conv1d", **kwargs)

    def conv2d(self, kernel, **kwargs):
        return self._arithmetic_function(kernel, "conv2d", **kwargs)

    def div(self, y):
        result = self.clone()
        if isinstance(y, Array):
            result.share = jnp.broadcast_arrays(result.share, y.share)[0]
            return result.mul(y.reciprocal())
        elif isinstance(y, (jnp.ndarray, np.ndarray)):
            result.share = jnp.broadcast_arrays(result.share, jnp.asarray(y))[0]
        return result.div_(y)

    def div_(self, y):
        if isinstance(y, Array):
            res = self.mul(y.reciprocal())
            self.share = res.share
            self.precision = res.precision
            return self
        if isinstance(y, float) and int(y) == y:
            y = int(y)
        if isinstance(y, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(y.dtype, jnp.floating):
            if bool(jnp.all(jnp.mod(y, 1.0) == 0)):
                y = jnp.asarray(y, dtype=DATATYPE)

        if isinstance(y, (int, np.integer)) or (
            isinstance(y, (jnp.ndarray, np.ndarray)) and jnp.issubdtype(y.dtype, jnp.integer)
        ):
            if comm.get().get_world_size() > 2:
                _truncate(self, y)
            elif isinstance(y, (int, np.integer)):
                self.share = _div_scalar_jit(self.share, int(y))
            else:
                y_arr = jnp.asarray(y, dtype=DATATYPE)
                self.share = jax.lax.div(self.share, y_arr)
            return self

        if isinstance(y, float):
            return self.mul_(1.0 / y)
        y_arr = jnp.asarray(y, dtype=jnp.float32)
        return self.mul_(jnp.reciprocal(y_arr))

    def divmod(self, y):
        divisor = self.div(y)
        remainder = self - divisor * y
        return divisor, remainder

    def mod(self, y):
        _, remainder = self.divmod(y)
        return remainder

    def egk_trunc_pr(self, l, m):
        if m == 0:
            return self.clone()
        result = self.clone()
        result.share = _egk_trunc_pr(result, l, m).share
        return result

    def egk_truncmod_pr(self, l, m):
        divisor = self.egk_trunc_pr(l, m)
        with IgnoreEncodings([self, divisor]):
            remainder = self - divisor * (2 ** m)
        return divisor, remainder

    def neg_(self):
        self.share = _neg_jit(self.share)
        return self

    def neg(self):
        return self.clone().neg_()

    def square_(self):
        method = cfg.encoder.trunc_method.prod
        do_local_trunc = (
            self.precision > 0
            and method != "egk"
            and comm.get().get_world_size() <= 2
        )
        trunc_div = (1 << self.precision) if do_local_trunc else 1
        sq = _square(self, trunc_div=trunc_div)
        if self.precision > 0 and not do_local_trunc:
            if method == "egk":
                sq = sq.egk_trunc_pr(62, self.precision)
            else:
                sq = sq.div_(2 ** self.precision)
        self.share = sq.share
        return self

    def square(self):
        return self.clone().square_()

    def evaluate_lut(self, lut):
        result = self.clone()
        result.share = _evaluate_lut(result, lut).share
        return result

    def evaluate_bior_lut(self, luts, scale, bias):
        result = self.clone()
        result.share = _evaluate_bior_lut(result, luts, scale, bias).share
        return result

    def evaluate_embed(self, embed):
        result = self.clone()
        result.share = _evaluate_embed(result, embed.share, embed_obj=embed).share
        result.precision = embed.precision
        return result

    def shuffle(self):
        result = self.clone()
        shuffled, inv_permutation = _shuffle(result)
        result.share = shuffled.share
        return result, inv_permutation

    def unshuffle(self, inv_permutation):
        result = self.clone()
        result.share = _unshuffle(result, inv_permutation).share
        return result

    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise (with precision=0)."""
        if comm.get().get_world_size() == 2:
            if getattr(cfg.mpc, "protocol", "beaver") == "fss":
                from prix import fss

                return fss.fss_ltz(self)
            from prix import circuit

            provider = get_default_provider()
            if (
                hasattr(provider, "generate_ltz_rng_and_pre")
                and not provider.tracing
                and len(provider.tuple_cache) == 0
            ):
                is_rank0 = self.rank == 0
                (
                    ed_share,
                    a0_sh,
                    b0_sh,
                    c0_sh,
                    a_all,
                    b_all,
                    c_all,
                    rA_sh,
                    rB_sh,
                ) = provider.generate_ltz_rng_and_pre(self.share, k_rounds=6)
                ed = comm.get().all_reduce(ed_share, op="bxor")
                sp_share, ef_share = circuit._a2b_post_and_spk_round0_pre(
                    self.share, c0_sh, ed, a0_sh, b0_sh, a_all, b_all, is_rank0
                )
                for i in range(5):
                    ef = comm.get().all_reduce(ef_share, op="bxor")
                    sp_share, ef_share = circuit._spk_step_fused(
                        sp_share, ef, a_all, b_all, c_all, i, is_rank0
                    )
                ef_last = comm.get().all_reduce(ef_share, op="bxor")
                z_share = circuit._spk_round_last_and_ltz_b2a_pre(
                    sp_share, self.share, ef_last, a_all, b_all, c_all, rB_sh, is_rank0
                )
                z = comm.get().all_reduce(z_share, op="bxor")
                out_share = _b2a_combine(rA_sh, z, is_rank0)
                return Array.from_shares(
                    out_share, precision=0, device=self.device, ptype=Ptype.arithmetic
                )
        result = self._to_ptype(Ptype.binary)
        result.share = (result.share >> jnp.int64(63)) & jnp.int64(1)
        result = _b2a_single_bit(result)
        result.precision = 0
        return result

    def _eqz_2PC(self):
        """Returns 1 for elements == 0 in 2PC, 0 otherwise (with precision=0)."""
        from prix import circuit

        xor_share = self.share if self.rank == 0 else -self.share
        bitwise_equal = ~Array.from_shares(
            xor_share, precision=0, device=self.device, ptype=Ptype.binary
        )
        P = circuit._p_circuit(bitwise_equal)
        eq_bin = circuit._get_sign_bit(P)
        result = eq_bin._to_ptype(Ptype.arithmetic, precision=0, bits=1)
        result.precision = 0
        return result

    # Bitwise operations (for binary shares)
    def __ixor__(self, y):
        if isinstance(y, Array):
            self.share = self.share ^ y.share
        elif isinstance(y, (int, np.integer, jnp.ndarray, np.ndarray)):
            y_arr = jnp.asarray(y, dtype=DATATYPE)
            if self.rank == 0:
                self.share = self.share ^ y_arr
            else:
                self.share = jnp.broadcast_arrays(self.share, y_arr)[0]
        else:
            raise TypeError(f"Cannot XOR {type(y)} with {type(self)}")
        return self

    def __xor__(self, y):
        return self.clone().__ixor__(y)

    def __rxor__(self, y):
        return self.__xor__(y)

    def __iand__(self, y):
        if isinstance(y, Array):
            self.share = _binary_and(self, y).share
        elif isinstance(y, (int, np.integer, jnp.ndarray, np.ndarray)):
            self.share = self.share & jnp.asarray(y, dtype=DATATYPE)
        else:
            raise TypeError(f"Cannot AND {type(y)} with {type(self)}")
        return self

    def __and__(self, y):
        return self.clone().__iand__(y)

    def __rand__(self, y):
        return self.__and__(y)

    def __or__(self, y):
        return (self & y) ^ self ^ y

    def __ior__(self, y):
        res = self.__or__(y)
        self.share = res.share
        return self

    def __ror__(self, y):
        return self.__or__(y)

    def __invert__(self):
        result = self.clone()
        if result.rank == 0:
            result.share = result.share ^ jnp.int64(-1)
        return result

    def lshift_(self, value):
        self.share = self.share << int(value)
        return self

    def lshift(self, value):
        return self.clone().lshift_(value)

    def rshift_(self, value):
        self.share = self.share >> int(value)
        return self

    def rshift(self, value):
        return self.clone().rshift_(value)

    __lshift__ = lshift
    __rshift__ = rshift
    __ilshift__ = lshift_
    __irshift__ = rshift_

    # Regular Tensor Manipulation Methods
    def size(self, dim=None):
        if dim is None:
            return self.share.shape
        return self.share.shape[dim]

    def dim(self):
        return self.share.ndim

    def nelement(self):
        return self.share.size

    def numel(self):
        return self.share.size

    def __len__(self):
        return len(self.share)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(
                jnp.asarray( idx ) if isinstance(idx, list) else idx for idx in index
            )
        elif isinstance(index, list):
            index = jnp.asarray(index)
        result = self.shallow_copy()
        result.share = self.share[index]
        if getattr(self, "_is_parameter", False):
            result._is_parameter = True
            if getattr(self, "_plain_encoded", None) is not None:
                result._plain_encoded = self._plain_encoded[index]
        return result

    def __setitem__(self, index, value):
        if not isinstance(value, Array):
            value = Array(value, src=0, precision=self.precision, device=self.device, ptype=self.ptype)
        self.share = self.share.at[index].set(value.share)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        result = self.shallow_copy()
        result.share = self.share.reshape(shape)
        if getattr(self, "_is_parameter", False):
            result._is_parameter = True
            if getattr(self, "_plain_encoded", None) is not None:
                result._plain_encoded = self._plain_encoded.reshape(shape)
        return result

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1):
        ndim = self.dim()
        if ndim == 0:
            return self.reshape((1,))
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        new_shape = (
            self.shape[:start_dim]
            + (-1,)
            + self.shape[end_dim + 1 :]
        )
        return self.reshape(new_shape)

    def transpose(self, dim0=0, dim1=1):
        if isinstance(dim0, (tuple, list)):
            return self.permute(*dim0)
        axes = list(range(self.dim()))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        result = self.shallow_copy()
        result.share = jnp.transpose(self.share, axes)
        if getattr(self, "_is_parameter", False):
            result._is_parameter = True
            if getattr(self, "_plain_encoded", None) is not None:
                result._plain_encoded = jnp.transpose(self._plain_encoded, axes)
        return result

    def t(self):
        assert self.dim() <= 2, "t() expects a tensor with <= 2 dimensions"
        if self.dim() < 2:
            return self.clone()
        return self.transpose(0, 1)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        result = self.shallow_copy()
        result.share = jnp.transpose(self.share, dims)
        if getattr(self, "_is_parameter", False):
            result._is_parameter = True
            if getattr(self, "_plain_encoded", None) is not None:
                result._plain_encoded = jnp.transpose(self._plain_encoded, dims)
        return result

    def squeeze(self, dim=None):
        result = self.shallow_copy()
        if dim is None:
            result.share = jnp.squeeze(self.share)
        else:
            if self.size(dim) == 1:
                result.share = jnp.squeeze(self.share, axis=dim)
            else:
                result.share = self.share
        return result

    def unsqueeze(self, dim):
        result = self.shallow_copy()
        result.share = jnp.expand_dims(self.share, axis=dim)
        return result

    def expand(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        target_shape = list(sizes)
        offset = len(target_shape) - self.dim()
        for i in range(self.dim()):
            if target_shape[offset + i] == -1:
                target_shape[offset + i] = self.shape[i]
        result = self.shallow_copy()
        result.share = jnp.broadcast_to(self.share, tuple(target_shape))
        return result

    def repeat(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        result = self.shallow_copy()
        result.share = jnp.tile(self.share, sizes)
        return result

    def roll(self, shifts, dims=None):
        result = self.shallow_copy()
        result.share = jnp.roll(self.share, shift=shifts, axis=dims)
        return result

    def flip(self, dims):
        result = self.shallow_copy()
        result.share = jnp.flip(self.share, axis=dims)
        return result

    def narrow(self, dim, start, length):
        result = self.shallow_copy()
        result.share = jax.lax.slice_in_dim(self.share, start, start + length, axis=dim)
        return result

    def unbind(self, dim=0):
        n = self.size(dim)
        return tuple(self.narrow(dim, i, 1).squeeze(dim) for i in range(n))

    def split(self, split_size, dim=0):
        n = self.size(dim)
        if isinstance(split_size, int):
            if split_size <= 0:
                return (self.clone(),)
            indices = list(range(split_size, n, split_size))
        else:
            indices = list(np.cumsum(split_size)[:-1])
        arrays = jnp.split(self.share, indices, axis=dim)
        results = []
        for arr in arrays:
            r = self.shallow_copy()
            r.share = arr
            results.append(r)
        return tuple(results)

    def chunk(self, chunks, dim=0):
        n = self.size(dim)
        if n == 0:
            return (self.clone(),)
        chunk_size = max(1, math.ceil(n / chunks))
        return self.split(chunk_size, dim=dim)

    def index_select(self, dim, index):
        idx = jnp.asarray(index, dtype=jnp.int64).reshape(-1)
        result = self.shallow_copy()
        result.share = jnp.take(self.share, idx, axis=dim)
        return result

    def gather(self, dim, index):
        idx = index.share if isinstance(index, Array) else jnp.asarray(index, dtype=jnp.int64)
        result = self.shallow_copy()
        result.share = jnp.take_along_axis(self.share, idx, axis=dim)
        return result

    def take(self, index, dimension=None):
        idx = jnp.asarray(index, dtype=jnp.int64)
        result = self.shallow_copy()
        if dimension is None or self.dim() == 0:
            result.share = jnp.take(self.share, idx)
        else:
            result.share = jnp.take(self.share, idx, axis=dimension)
        return result

    def pad(self, pad, mode="constant", value=0):
        assert mode == "constant", f"Padding with mode {mode} is currently unsupported"
        result = self.shallow_copy()
        if isinstance(pad[0], (tuple, list)):
            pad_width = list(pad)
        else:
            pad_width = [(0, 0)] * self.dim()
            for i in range(0, len(pad), 2):
                dim_idx = self.dim() - 1 - (i // 2)
                pad_width[dim_idx] = (pad[i], pad[i + 1])

        if isinstance(value, (int, float)):
            enc_val = int(encode(value, precision=self.precision).item()) if self.rank == 0 else 0
            result.share = jnp.pad(result.share, pad_width, mode="constant", constant_values=enc_val)
        elif isinstance(value, Array):
            assert value.dim() == 0, "Private values used for padding must be 0-dimensional"
            result.share = jnp.pad(
                result.share, pad_width, mode="constant", constant_values=int(value.share.item())
            )
        else:
            raise TypeError(f"Cannot pad Array with {type(value)}")
        return result

    @staticmethod
    def cat(tensors, dim=0, axis=None):
        if axis is not None:
            dim = axis
        assert isinstance(tensors, (list, tuple)) and len(tensors) > 0
        non_empty = [t for t in tensors if t.nelement() > 0]
        if not non_empty:
            return tensors[0].clone()
        result = non_empty[0].clone()
        result.share = jnp.concatenate([t.share for t in non_empty], axis=dim)
        return result

    @staticmethod
    def stack(tensors, dim=0, axis=None):
        if axis is not None:
            dim = axis
        assert isinstance(tensors, (list, tuple)) and len(tensors) > 0
        result = tensors[0].clone()
        result.share = jnp.stack([t.share for t in tensors], axis=dim)
        return result

    def sum(self, dim=None, keepdim=False, axis=None, keepdims=None):
        if axis is not None:
            dim = axis
        if keepdims is not None:
            keepdim = keepdims
        if self.ptype == Ptype.binary:
            if dim is None:
                x = self.flatten()
            else:
                x = self.transpose(0, dim)
            while x.size(0) > 1:
                extra = None
                if x.size(0) % 2 == 1:
                    extra = x[0]
                    x = x[1:]
                half = x.size(0) // 2
                x = x[:half] + x[half:]
                if extra is not None:
                    x.share = jnp.concatenate([x.share, jnp.expand_dims(extra.share, 0)], axis=0)
            if dim is None:
                return x.squeeze()
            x = x.transpose(0, dim)
            return x if keepdim else x.squeeze(dim)

        result = self.shallow_copy()
        result.share = jnp.sum(self.share, axis=dim, keepdims=keepdim)
        return result

    def cumsum(self, dim=0):
        result = self.shallow_copy()
        result.share = jnp.cumsum(self.share, axis=dim)
        return result

    def trace(self):
        result = self.shallow_copy()
        result.share = jnp.trace(self.share)
        return result

    def mean(self, dim=None, keepdim=False, axis=None, keepdims=None):
        if axis is not None:
            dim = axis
        if keepdims is not None:
            keepdim = keepdims
        result = self.sum(dim=dim, keepdim=keepdim)
        if self.dim() == 0:
            return result
        divisor = self.nelement() // max(1, result.nelement())
        return result.div(divisor)

    def var(self, dim=None, unbiased=False, keepdim=False):
        if dim is None:
            m = self.mean()
            result = (self - m).square().sum()
        else:
            m = self.mean(dim=dim, keepdim=True)
            result = (self - m).square().sum(dim=dim, keepdim=keepdim)
        divisor = self.nelement() // max(1, result.nelement())
        if unbiased:
            divisor -= 1
        if divisor in (0, 1):
            return result
        return result.div(divisor)

    def prod(self, dim=None, keepdim=False):
        if dim is None:
            return self.flatten().prod(dim=0)
        result = self.clone()
        while result.size(dim) > 1:
            size = result.size(dim)
            x, y, remainder = result.split([size // 2, size // 2, size % 2], dim=dim)
            result = x.mul(y)
            result = Array.cat([result, remainder], dim=dim)
        if not keepdim:
            result = result.squeeze(dim)
        return result

    def dot(self, y, weights=None):
        result = self * weights if weights is not None else self.clone()
        return result.mul(y).sum()

    def ger(self, y):
        assert self.dim() == 1 and y.dim() == 1, "Outer product must be on 1D tensors"
        return self.view((-1, 1)).matmul(y.view((1, -1)))

    def avg_pool2d(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        if ceil_mode:
            raise NotImplementedError("ceil_mode is not supported for avg_pool2d")
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s = k if stride is None else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        p = (padding, padding) if isinstance(padding, int) else tuple(padding)

        x = self.share
        added_batch = False
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)
            added_batch = True

        # x is (N, C, H, W)
        window_shape = (1, 1, k[0], k[1])
        window_strides = (1, 1, s[0], s[1])
        pad_spec = [(0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])]
        summed = jax.lax.reduce_window(
            x,
            init_value=jnp.int64(0),
            computation=jax.lax.add,
            window_dimensions=window_shape,
            window_strides=window_strides,
            padding=pad_spec,
        )
        if added_batch:
            summed = jnp.squeeze(summed, 0)

        result = self.shallow_copy()
        result.share = summed
        pool_size = k[0] * k[1]
        return result.div(pool_size)

    # Operator overloads
    __add__ = add
    __iadd__ = add_
    __radd__ = add
    __sub__ = sub
    __isub__ = sub_

    def __rsub__(self, other):
        if isinstance(other, (int, float, np.number)) and self.ptype == Ptype.arithmetic:
            result = self.clone()
            if result.rank == 0:
                y_int = int(round((1 << result.precision) * float(other)))
                result.share = _rsub_scalar_jit(result.share, y_int)
            else:
                result.share = _neg_jit(result.share)
            return result
        return (-self) + other

    __mul__ = mul
    __imul__ = mul_

    def __rmul__(self, other):
        if isinstance(other, Array):
            return other.mul(self)
        return self.mul(other)

    __div__ = div
    __truediv__ = div
    __itruediv__ = div_
    __neg__ = neg
    __mod__ = mod


@jax.jit
def _add_jit(x_share, y_share):
    return x_share + y_share


@jax.jit
def _sub_jit(x_share, y_share):
    return x_share - y_share


@jax.jit
def _neg_jit(x_share):
    return -x_share


@jax.jit
def _add_scalar_jit(x_share, y_int):
    return x_share + y_int


@jax.jit
def _rsub_scalar_jit(x_share, y_int):
    return y_int - x_share


@functools.partial(jax.jit, static_argnames=("trunc_div",))
def _mul_div_scalar_jit(x_share, y_int, trunc_div: int):
    res = x_share * y_int
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


@jax.jit
def _div_scalar_jit(x_share, y_int):
    return jax.lax.div(x_share, y_int)


@functools.partial(jax.jit, static_argnames=("is_rank0",))
def _a2b_pre_and(x_share, a_share, b_share, is_rank0):
    e_s = (x_share ^ a_share) if is_rank0 else a_share
    f_s = b_share if is_rank0 else (x_share ^ b_share)
    return jnp.stack([e_s, f_s], axis=0)


@jax.jit
def _beaver_pre_same_shape(x_share, y_share, a_share, b_share):
    return jnp.stack([x_share - a_share, y_share - b_share], axis=0)


@jax.jit
def _beaver_pre_diff(x_share, y_share, a_share, b_share):
    return x_share - a_share, y_share - b_share


@functools.partial(jax.jit, static_argnames=("trunc_div", "is_rank0"))
def _beaver_mul_combine_ed(c_share, ed, a_share, b_share, trunc_div: int, is_rank0: bool):
    eps = ed[0]
    delta = ed[1]
    b_eff = (b_share + delta) if is_rank0 else b_share
    res = c_share + eps * b_eff + a_share * delta
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


@functools.partial(jax.jit, static_argnames=("trunc_div", "is_rank0"))
def _beaver_mul_combine(c_share, eps, delta, a_share, b_share, trunc_div: int, is_rank0: bool):
    b_eff = (b_share + delta) if is_rank0 else b_share
    res = c_share + eps * b_eff + a_share * delta
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


def _ndim(x):
    if isinstance(x, Array):
        return x.dim()
    return np.ndim(x)


def _matmul_promoting_1d(a, b, fn):
    """
    Applies matmul `fn(a, b)` with numpy semantics for 1-D operands: a 1-D left operand is
    treated as a row vector and a 1-D right operand as a column vector, and the added
    dimensions are removed from the result. The underlying kernels require >= 2-D operands.
    """
    a_dim, b_dim = _ndim(a), _ndim(b)
    if a_dim >= 2 and b_dim >= 2:
        return fn(a, b)
    if a_dim == 0 or b_dim == 0:
        raise ValueError("matmul does not accept 0-d (scalar) operands; use * instead")
    if a_dim == 1:
        a = a.unsqueeze(0) if isinstance(a, Array) else np.expand_dims(np.asarray(a), 0)
    if b_dim == 1:
        b = b.unsqueeze(-1) if isinstance(b, Array) else np.expand_dims(np.asarray(b), -1)
    out = fn(a, b)
    if b_dim == 1:
        out = out.squeeze(-1)
    if a_dim == 1:
        out = out.squeeze(-1 if b_dim == 1 else -2)
    return out


@functools.partial(jax.jit, static_argnames=("trunc_div", "is_rank0"))
def _beaver_matmul_combine_ed(c_share, ed, a_share, b_share, trunc_div: int, is_rank0: bool):
    eps = ed[0]
    delta = ed[1]
    b_eff = (b_share + delta) if is_rank0 else b_share
    lhs = jnp.stack([eps, a_share], axis=0)
    rhs = jnp.stack([b_eff, delta], axis=0)
    prods = _int64_matmul(lhs, rhs)
    res = c_share + prods[0] + prods[1]
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


@functools.partial(jax.jit, static_argnames=("trunc_div", "is_rank0"))
def _beaver_matmul_combine(c_share, eps, delta, a_share, b_share, trunc_div: int, is_rank0: bool):
    b_eff = (b_share + delta) if is_rank0 else b_share
    if eps.ndim > b_eff.ndim:
        pad = (1,) * (eps.ndim - b_eff.ndim)
        b_eff = b_eff.reshape(pad + b_eff.shape)
        delta = delta.reshape(pad + delta.shape)
    elif b_eff.ndim > eps.ndim:
        pad = (1,) * (b_eff.ndim - eps.ndim)
        eps = eps.reshape(pad + eps.shape)
        a_share = a_share.reshape(pad + a_share.shape)
    lhs = jnp.stack([eps, a_share], axis=0)
    rhs = jnp.stack([b_eff, delta], axis=0)
    prods = _int64_matmul(lhs, rhs)
    res = c_share + prods[0] + prods[1]
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


@functools.partial(jax.jit, static_argnames=("trunc_div",))
def _linear_mul_combine(c_share, eps, y_share, trunc_div: int):
    res = c_share + eps * y_share
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


@functools.partial(jax.jit, static_argnames=("trunc_div",))
def _linear_matmul_combine(c_share, eps, y_share, trunc_div: int):
    if eps.ndim > y_share.ndim:
        pad = (1,) * (eps.ndim - y_share.ndim)
        y_share = y_share.reshape(pad + y_share.shape)
    elif y_share.ndim > eps.ndim:
        pad = (1,) * (y_share.ndim - eps.ndim)
        eps = eps.reshape(pad + eps.shape)
    res = c_share + _int64_matmul(eps, y_share)
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


# Protocol helper functions
def _beaver_protocol(op, x, y, *args, _trunc_div=1, **kwargs):
    """Performs Beaver protocol for additively secret-shared Arrays x and y."""
    provider = get_default_provider()
    is_rank0 = comm.get().get_rank() == 0
    if (
        getattr(cfg.mpc, "protocol", "beaver") == "fss"
        and not getattr(cfg.mpc, "active_security", False)
    ):
        if op == "mul" and getattr(x, "_is_parameter", False) and not getattr(y, "_is_parameter", False):
            x, y = y, x
        if getattr(y, "_is_parameter", False):
            a, c = provider.generate_linear_triple(
                x.size(),
                y.size(),
                op,
                device=x.device,
                _w_plain=getattr(y, "_plain_encoded", None),
                *args,
                **kwargs,
            )
            eps_s = _sub_jit(x.share, a.share)
            epsilon = comm.get().all_reduce(eps_s, op="sum")
            if op == "mul":
                c.share = _linear_mul_combine(c.share, epsilon, y.share, _trunc_div)
            elif op == "matmul":
                c.share = _linear_matmul_combine(c.share, epsilon, y.share, _trunc_div)
            else:
                z = _eval_ring_op(op, epsilon, y.share, *args, **kwargs)
                c.share = c.share + z
                if _trunc_div > 1:
                    c.share = jax.lax.div(c.share, jnp.int64(_trunc_div))
            return c
    if (
        op == "mul"
        and x.size() == y.size()
        and jax.default_backend() == "gpu"
        and hasattr(provider, "generate_mul_triple_and_pre")
        and not provider.tracing
        and len(provider.tuple_cache) == 0
        and not getattr(cfg.mpc, "active_security", False)
    ):
        ed_share, a_sh, b_sh, c_sh = provider.generate_mul_triple_and_pre(x.share, y.share)
        ed = comm.get().all_reduce(ed_share, op="sum")
        out_share = _beaver_mul_combine_ed(c_sh, ed, a_sh, b_sh, _trunc_div, is_rank0)
        return Array.from_shares(out_share, precision=0, device=x.device)

    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), op, device=x.device, *args, **kwargs
    )

    if getattr(cfg.mpc, "active_security", False):
        f, g, h = provider.generate_additive_triple(
            x.size(), y.size(), op, device=x.device, *args, **kwargs
        )
        t = Array.PRSS(a.size(), device=x.device)
        t_plain = t.get_plain_text()
        rho = (t_plain * a - f).get_plain_text()
        sigma = (b - g).get_plain_text()
        triples_check = (t_plain * c - h - sigma * f - rho * g - rho * sigma).get_plain_text()
        if bool(jnp.any(triples_check != 0)):
            raise ValueError("Beaver Triples verification failed!")

    if x.size() == y.size():
        ed_share = _beaver_pre_same_shape(x.share, y.share, a.share, b.share)
        ed = comm.get().all_reduce(ed_share, op="sum")
        if op == "mul":
            c.share = _beaver_mul_combine_ed(
                c.share, ed, a.share, b.share, _trunc_div, is_rank0
            )
            return c
        elif op == "matmul":
            c.share = _beaver_matmul_combine_ed(
                c.share, ed, a.share, b.share, _trunc_div, is_rank0
            )
            return c
        epsilon, delta = ed[0], ed[1]
    else:
        eps_s, delta_s = _beaver_pre_diff(x.share, y.share, a.share, b.share)
        epsilon, delta = comm.get().all_reduce([eps_s, delta_s], op="sum", batched=True)

    if op == "mul":
        c.share = _beaver_mul_combine(
            c.share, epsilon, delta, a.share, b.share, _trunc_div, is_rank0
        )
    elif op == "matmul":
        c.share = _beaver_matmul_combine(
            c.share, epsilon, delta, a.share, b.share, _trunc_div, is_rank0
        )
    else:
        z = _eval_ring_op(op, epsilon, b.share, *args, **kwargs) + _eval_ring_op(
            op, a.share, delta, *args, **kwargs
        )
        if is_rank0:
            z = z + _eval_ring_op(op, epsilon, delta, *args, **kwargs)
        c.share = c.share + z
    return c


@functools.partial(jax.jit, static_argnames=("trunc_div", "is_rank0"))
def _square_combine_jit(r_share, r2_share, eps, trunc_div: int, is_rank0: bool):
    res = r2_share + jnp.int64(2) * r_share * eps
    if is_rank0:
        res = res + eps * eps
    if trunc_div > 1:
        res = jax.lax.div(res, jnp.int64(trunc_div))
    return res


def _square(x, trunc_div=1):
    provider = get_default_provider()
    is_rank0 = comm.get().get_rank() == 0
    if (
        jax.default_backend() == "gpu"
        and hasattr(provider, "square_and_pre")
        and not provider.tracing
        and len(provider.tuple_cache) == 0
    ):
        eps_share, r_sh, r2_sh = provider.square_and_pre(x.share)
        epsilon = comm.get().all_reduce(eps_share, op="sum")
        out_share = _square_combine_jit(r_sh, r2_sh, epsilon, trunc_div, is_rank0)
        return Array.from_shares(out_share, precision=0, device=x.device)

    r, r2 = provider.square(x.size(), device=x.device)
    eps_share = _sub_jit(x.share, r.share)
    epsilon = comm.get().all_reduce(eps_share, op="sum")
    r2.share = _square_combine_jit(r.share, r2.share, epsilon, trunc_div, is_rank0)
    return r2


@jax.jit
def _wraps_pre_jit(x_share, r_share, theta_r_share):
    z_share = x_share + r_share
    beta_xr = (
        ((x_share > 0) & (r_share > 0) & (z_share < 0)).astype(DATATYPE)
        - ((x_share < 0) & (r_share < 0) & (z_share > 0)).astype(DATATYPE)
    )
    return z_share, beta_xr - theta_r_share


@jax.jit
def _wraps_post_2pc_jit(theta_x_share, z0, z1):
    z_sum = z0 + z1
    theta_z = (
        ((z0 > 0) & (z1 > 0) & (z_sum < 0)).astype(DATATYPE)
        - ((z0 < 0) & (z1 < 0) & (z_sum > 0)).astype(DATATYPE)
    )
    return theta_x_share + theta_z


@jax.jit
def _truncate_post_jit(x_share, wrap_share, y_arr, corr_factor):
    return jax.lax.div(x_share, y_arr) - wrap_share * corr_factor


@jax.jit
def _truncate_post_rank0_2pc_jit(x_share, theta_x_share, z0, z1, y_arr, corr_factor):
    z_sum = z0 + z1
    theta_z = (
        ((z0 > 0) & (z1 > 0) & (z_sum < 0)).astype(DATATYPE)
        - ((z0 < 0) & (z1 < 0) & (z_sum > 0)).astype(DATATYPE)
    )
    return jax.lax.div(x_share, y_arr) - (theta_x_share + theta_z) * corr_factor


def _wraps(x):
    provider = get_default_provider()
    r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    z_share, theta_x_share = _wraps_pre_jit(x.share, r.share, theta_r.share)
    theta_z = comm.get().gather(z_share, 0)

    if x.rank == 0:
        if len(theta_z) == 2:
            theta_x_share = _wraps_post_2pc_jit(theta_x_share, theta_z[0], theta_z[1])
        else:
            theta_x_share = theta_x_share + count_wraps(theta_z)
    return Array.from_shares(theta_x_share, precision=0, device=x.device)


def _truncate(x, y):
    if isinstance(y, (int, float, np.number)):
        y_int = int(y)
        y_arr = jnp.int64(y_int)
        corr_factor = jnp.int64(4 * ((1 << 62) // y_int))
    else:
        y_arr = jnp.asarray(y, dtype=DATATYPE)
        corr_factor = jnp.int64(4) * (jnp.int64(1 << 62) // y_arr)
    if comm.get().get_world_size() == 2:
        provider = get_default_provider()
        r, theta_r = provider.wrap_rng(x.size(), device=x.device)
        z_share, theta_x_share = _wraps_pre_jit(x.share, r.share, theta_r.share)
        theta_z = comm.get().gather(z_share, 0)
        if x.rank == 0:
            x.share = _truncate_post_rank0_2pc_jit(
                x.share, theta_x_share, theta_z[0], theta_z[1], y_arr, corr_factor
            )
        else:
            x.share = _truncate_post_jit(x.share, theta_x_share, y_arr, corr_factor)
        return x

    wrap_count = _wraps(x)
    x.share = _truncate_post_jit(x.share, wrap_count.share, y_arr, corr_factor)
    return x


@jax.jit
def _egk_pre_reveal(x_share, r_share, rp_share, b_share, l: int, m: int, is_rank0: bool):
    k = 64
    a_p = jnp.where(is_rank0, x_share + (jnp.int64(1) << (l - 1)), x_share)
    rpp = (jnp.int64(1) << m) * r_share + rp_share
    return (jnp.int64(1) << (k - l - 1)) * (a_p + (jnp.int64(1) << l) * b_share + rpp)


@jax.jit
def _egk_post_reveal(c, r_share, b_share, l: int, m: int, is_rank0: bool):
    k = 64
    two_to_l = jnp.int64(1) << l
    c_p = c >> (k - l - 1)
    c_pl = (c_p >> l) & jnp.int64(1)
    v_share = b_share * (jnp.int64(1) - jnp.int64(2) * c_pl)
    v_share = jnp.where(is_rank0, v_share + c_pl, v_share)
    base = (jnp.int64(1) << (l - m)) * v_share - r_share
    pub_offset = -(jnp.int64(1) << (l - m - 1)) + ((c_p % two_to_l) >> m)
    return jnp.where(is_rank0, base + pub_offset, base)


def _egk_trunc_pr(x, l, m):
    """Probabilistic truncation by m bits using the [EGK+20] protocol."""
    provider = get_default_provider()
    is_rank0 = comm.get().get_rank() == 0
    if (
        jax.default_backend() == "gpu"
        and hasattr(provider, "egk_rng_and_pre")
        and not provider.tracing
        and len(provider.tuple_cache) == 0
    ):
        enc_c_share, r_sh, b_sh = provider.egk_rng_and_pre(x.share, l, m)
        c = comm.get().all_reduce(enc_c_share, op="sum")
        y_share = _egk_post_reveal(c, r_sh, b_sh, int(l), int(m), is_rank0)
        return Array.from_shares(y_share, precision=x.precision, device=x.device)

    tensor_size = x.size()
    r, r_p, b = provider.egk_trunc_pr_rng(tensor_size, l, m, device=x.device)
    enc_c_share = _egk_pre_reveal(
        x.share, r.share, r_p.share, b.share, int(l), int(m), is_rank0
    )
    c = comm.get().all_reduce(enc_c_share, op="sum")
    y_share = _egk_post_reveal(c, r.share, b.share, int(l), int(m), is_rank0)
    return Array.from_shares(y_share, precision=x.precision, device=x.device)


@jax.jit
def _a2b_finalize(p_share, carry_share):
    return p_share ^ (carry_share << jnp.int64(1))


@jax.jit
def _lut_gather_dot(one_hot_share, z_revealed, lut):
    size = lut.shape[0]
    shift_amount = z_revealed % size
    arange = jnp.arange(size, dtype=DATATYPE)
    indices = (arange[None, :] - shift_amount[:, None]) % size
    gathered = jnp.take_along_axis(one_hot_share, indices, axis=1)
    return jnp.sum(gathered * lut, axis=1)


@jax.jit
def _bior_lut_gather_dot(one_hot_share, z_revealed, luts):
    size = luts.shape[1]
    shift_amount = z_revealed % size
    arange = jnp.arange(size, dtype=DATATYPE)
    indices = (arange[None, :] - shift_amount[:, None]) % size
    gathered = jnp.take_along_axis(one_hot_share, indices, axis=1)
    return jnp.sum(gathered * luts[0], axis=1), jnp.sum(gathered * luts[1], axis=1)


def _evaluate_lut(x, lut):
    """Evaluates a 1D Wavelet Look-Up Table (Haar) on secret-shared Array x."""
    lut = to_jax(lut, device=x.device, dtype=DATATYPE)
    size = int(lut.shape[0])
    if (
        comm.get().get_world_size() == 2
        and cfg.encoder.trunc_method.lut in ("wave", "wave_prob")
        and size > 0
        and (size & (size - 1)) == 0
    ):
        from prix import fss

        return fss.wave_haar_lut(x, lut, truncation=0)
    provider = get_default_provider()
    shape = x.size()
    x_flat = x.flatten()

    r, one_hot_r = provider.generate_one_hot(x_flat.size(), size, device=x.device)

    with IgnoreEncodings([x_flat, r]):
        z = x_flat - r
        z_revealed = z.reveal()

    if z_revealed.ndim > 0 and z_revealed.size > 0:
        res_share = _lut_gather_dot(one_hot_r.share, z_revealed, lut)
        result = Array.from_shares(res_share, precision=one_hot_r.precision, device=x.device)
    else:
        shift_amount = z_revealed % size
        one_hot_r = one_hot_r.roll(int(shift_amount.item()))
        lookup = one_hot_r * lut
        result = lookup.sum()
    return result.reshape(shape)


def _evaluate_bior_lut(x, luts, scale, bias):
    """Evaluates a Bior2.2 Wavelet Look-Up Table on secret-shared Array x."""
    provider = get_default_provider()
    luts = to_jax(luts, device=x.device, dtype=DATATYPE)
    size = luts[0].shape[0]
    shape = x.size()
    x_flat = x.flatten()

    r, one_hot_r = provider.generate_one_hot(x_flat.size(), size, device=x.device)

    with IgnoreEncodings([x_flat, r]):
        z = x_flat - r
        z_revealed = z.reveal()

    if z_revealed.ndim > 0 and z_revealed.size > 0:
        s0, s1 = _bior_lut_gather_dot(one_hot_r.share, z_revealed, luts)
        lut0 = Array.from_shares(s0, precision=one_hot_r.precision, device=x.device)
        lut1 = Array.from_shares(s1, precision=one_hot_r.precision, device=x.device)
    else:
        shift_amount = z_revealed % size
        one_hot_r = one_hot_r.roll(int(shift_amount.item()))
        lut0 = (one_hot_r * luts[0]).sum()
        lut1 = (one_hot_r * luts[1]).sum()

    with IgnoreEncodings([scale, lut0, lut1]):
        scaling = scale.flatten()
        lut = (lut1 - lut0) * scaling + (2 ** bias) * lut0
        result = lut.egk_trunc_pr(62, 2 * bias)
        result = result.reshape(shape)
    return result


def _evaluate_embed(x, embed_share, embed_obj=None):
    """Evaluates an embedding lookup using secret-shared indices x."""
    provider = get_default_provider()
    size = embed_share.shape[0]
    shape = x.size() + (embed_share.shape[1],)
    x_flat = x.flatten()

    embed = Array.from_shares(embed_share, precision=0, device=x.device)
    if embed_obj is not None and getattr(embed_obj, "_is_parameter", False):
        embed._is_parameter = True
        embed._plain_encoded = getattr(embed_obj, "_plain_encoded", None)
    r, one_hot_r = provider.generate_one_hot(x_flat.size(), size, device=x.device)

    with IgnoreEncodings([x_flat, r]):
        z = x_flat - r
        shift_amount = z.reveal() % size

    if shift_amount.ndim > 0 and shift_amount.size > 0:
        arange = jnp.arange(size, dtype=DATATYPE)
        indices = (arange[None, :] - shift_amount[:, None]) % size
        one_hot_r = one_hot_r.gather(1, indices)
        lookup = one_hot_r.matmul(embed)
    else:
        one_hot_r = one_hot_r.roll(int(shift_amount.item()))
        lookup = one_hot_r.matmul(embed)
    return lookup.reshape(shape)


def _shuffle(x):
    provider = get_default_provider()
    permutation, inv_permutation = provider.generate_permutation(x.size(0), device=x.device)
    return x[permutation], inv_permutation


def _unshuffle(x, inv_permutation):
    return x[inv_permutation]


@jax.jit
def _binary_and_combine(c_share, eps, delta, a_share, b_share, is_rank0):
    b_eff = jnp.where(is_rank0, b_share ^ delta, b_share)
    return c_share ^ (b_eff & eps) ^ (a_share & delta)


@jax.jit
def _b2a_combine(rA_share, z, is_rank0):
    res = rA_share * (jnp.int64(1) - jnp.int64(2) * z)
    return jnp.where(is_rank0, res + z, res)


def _binary_and(x, y):
    """Performs Beaver AND protocol for binary secret-shared Arrays x and y."""
    provider = get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)
    epsilon, delta = Array.reveal_batch([x ^ a, y ^ b])
    c.share = _binary_and_combine(
        c.share, epsilon, delta, a.share, b.share, comm.get().get_rank() == 0
    )
    return c


def _b2a_single_bit(xB):
    """Converts a single-bit binary-shared Array xB into an arithmetic-shared Array."""
    if comm.get().get_world_size() < 2:
        return Array(xB.share, precision=0, src=0, device=xB.device, ptype=Ptype.arithmetic)

    provider = get_default_provider()
    rA, rB = provider.B2A_rng(xB.size(), device=xB.device)
    if getattr(cfg.mpc, "protocol", "beaver") == "fss":
        from prix import fss

        z = fss.reveal_packed_bits(xB.share ^ rB.share)
    else:
        z = (xB ^ rB).reveal()
    out_share = _b2a_combine(rA.share, z, comm.get().get_rank() == 0)
    return Array.from_shares(out_share, precision=0, device=xB.device, ptype=Ptype.arithmetic)


# Attach approximation, logic, reduction, power, sampling, and pooling functions to Array
def _register_approximations():
    from prix import approximations, circuit

    for func_name in approximations.__all__:
        if func_name in ("LookupTables", "randn"):
            continue
        setattr(Array, func_name, getattr(approximations, func_name))

    def _op_eq(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.eq(self, y)
        return approximations.eq(self, y)

    def _op_ne(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.ne(self, y)
        return approximations.ne(self, y)

    def _op_lt(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.lt(self, y)
        return approximations.lt(self, y)

    def _op_le(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.le(self, y)
        return approximations.le(self, y)

    def _op_gt(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.gt(self, y)
        return approximations.gt(self, y)

    def _op_ge(self, y):
        if self.ptype == Ptype.binary:
            if not isinstance(y, Array):
                y = Array(y, src=0, precision=0, device=self.device, ptype=Ptype.binary)
            return circuit.ge(self, y)
        return approximations.ge(self, y)

    Array.eq = _op_eq
    Array.ne = _op_ne
    Array.lt = _op_lt
    Array.le = _op_le
    Array.gt = _op_gt
    Array.ge = _op_ge
    Array.__eq__ = _op_eq
    Array.__ne__ = _op_ne
    Array.__lt__ = _op_lt
    Array.__le__ = _op_le
    Array.__gt__ = _op_gt
    Array.__ge__ = _op_ge
    Array.__pow__ = approximations.pow


_register_approximations()
