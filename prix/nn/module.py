#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

import prix
from prix.encoder import resolve_device

_PARAM_KEY = None


def _next_init_key():
    global _PARAM_KEY
    if _PARAM_KEY is None:
        _PARAM_KEY = jax.random.PRNGKey(0)
    _PARAM_KEY, subkey = jax.random.split(_PARAM_KEY)
    return subkey


class Module:
    """
    Base Module class for PriX neural network layers and models.
    Supports `.encrypt()` and `.decrypt()` for seamless transition between
    plaintext JAX arrays and secret-shared `prix.Array`s.
    """

    _version = 1
    SUPPORTS_PLAINTEXT_INPUTS = False

    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_requires_grad", {})
        object.__setattr__(self, "encrypted", False)
        object.__setattr__(self, "training", True)

    def __repr__(self):
        encrypted_str = "encrypted" if self.encrypted else "unencrypted"
        return f"{type(self).__name__} {encrypted_str} module"

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for name, param in self._parameters.items():
            self._requires_grad[name] = mode
            if hasattr(param, "__dict__"):
                param.requires_grad = mode
        for module in self.children():
            module.train(mode=mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        for _, module in self.named_children():
            yield module

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def register_module(self, name, module):
        self._modules[name] = module

    def add_module(self, name, module):
        assert name not in self._modules, f"Module {name} already exists."
        self.register_module(name, module)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=""):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.named_children():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def register_parameter(self, name, param, requires_grad=True):
        if name in self._parameters or name in self.__dict__:
            raise ValueError(f"Parameter or field {name} already exists.")
        self._requires_grad[name] = requires_grad
        if hasattr(param, "__dict__"):
            param.requires_grad = requires_grad
        self._parameters[name] = param

    def set_parameter(self, name, param):
        if name not in self._parameters:
            raise ValueError(f"Parameter {name} does not exist.")
        self._parameters[name] = param

    def set_parameter_from_shares(self, name, share, **kwargs):
        assert self.encrypted, "can only set parameters from shares in encrypted models"
        if name not in self._parameters:
            raise ValueError(f"Parameter {name} does not exist.")
        self._parameters[name] = prix.Array.from_shares(share, **kwargs)

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, recurse=True, prefix=None):
        for name, param in self._parameters.items():
            param_name = name if prefix is None else prefix + "." + name
            yield param_name, param
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_parameters(recurse=recurse, prefix=pre)

    def register_buffer(self, name, buffer):
        if name in self._buffers or name in self.__dict__:
            raise ValueError(f"Buffer or field {name} already exists.")
        if hasattr(buffer, "__dict__"):
            buffer.requires_grad = False
        self._buffers[name] = buffer

    def set_buffer(self, name, buffer):
        if name not in self._buffers:
            raise ValueError(f"Buffer {name} does not exist.")
        self._buffers[name] = buffer

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, recurse=True, prefix=None):
        for name, buffer in self._buffers.items():
            buffer_name = name if prefix is None else prefix + "." + name
            yield buffer_name, buffer
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_buffers(recurse=recurse, prefix=pre)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = {"version": Module._version}
        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                destination[prefix + name] = param
        for name, buffer in self.named_buffers(recurse=False):
            if buffer is not None:
                destination[prefix + name] = buffer
        for name, module in self.named_children():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict.copy()

        def load(module, prefix=""):
            local_params = {k: v for k, v in module.named_parameters(recurse=False) if v is not None}
            local_buffers = {k: v for k, v in module.named_buffers(recurse=False) if v is not None}
            for name, param in local_params.items():
                key = prefix + name
                if key in state_dict:
                    val = state_dict[key]
                    if isinstance(param, prix.Array) and isinstance(val, prix.Array):
                        param.copy_(val)
                    else:
                        module.set_parameter(name, val)
                elif strict:
                    raise ValueError(f"Key {key} not found in state_dict.")
            for name, buf in local_buffers.items():
                key = prefix + name
                if key in state_dict:
                    module.set_buffer(name, state_dict[key])
            for name, child in module.named_children():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)
        return self

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device", None)
        if device is not None and device not in ("arithmetic", "binary"):
            self._device = device
        for name, param in list(self._parameters.items()):
            if hasattr(param, "to"):
                self.set_parameter(name, param.to(*args, **kwargs))
            elif device is not None and isinstance(param, (jnp.ndarray, np.ndarray)):
                self.set_parameter(name, jax.device_put(jnp.asarray(param), resolve_device(device)))
        for name, buffer in list(self._buffers.items()):
            if hasattr(buffer, "to"):
                self.set_buffer(name, buffer.to(*args, **kwargs))
            elif device is not None and isinstance(buffer, (jnp.ndarray, np.ndarray)):
                self.set_buffer(name, jax.device_put(jnp.asarray(buffer), resolve_device(device)))
        for module in self.children():
            module.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None):
        return self.to("cuda" if device is None else device)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        fn(self)
        return self

    def encrypt(self, mode=True, src=0):
        """Encrypts (or decrypts when mode=False) all parameters, buffers, and child modules."""
        if mode != self.encrypted:
            self.encrypted = mode
            dev = getattr(self, "_device", None)
            for name, param in list(self._parameters.items()):
                if mode:
                    if not isinstance(param, prix.Array):
                        enc_param = prix.cryptensor(param, src=src, device=dev)
                        enc_param._is_parameter = True
                        if param is not None and prix.comm.get().get_rank() == src:
                            enc_param._plain_encoded = prix.encoder.encode(
                                param, precision=enc_param.precision, device=dev
                            )
                        self.set_parameter(name, enc_param)
                else:
                    if isinstance(param, prix.Array):
                        self.set_parameter(name, param.get_plain_text())

            for name, buffer in list(self._buffers.items()):
                if mode:
                    if isinstance(buffer, (jnp.ndarray, np.ndarray)):
                        self.set_buffer(name, prix.cryptensor(buffer, src=src, device=dev))
                else:
                    if isinstance(buffer, prix.Array):
                        self.set_buffer(name, buffer.get_plain_text())

            for module in self.children():
                module.encrypt(mode=mode, src=src)
        return self

    def decrypt(self):
        return self.encrypt(mode=False)

    def __getattr__(self, name):
        if "_parameters" in self.__dict__ and name in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][name]
        if "_modules" in self.__dict__ and name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        if "_buffers" in self.__dict__ and name in self.__dict__["_buffers"]:
            return self.__dict__["_buffers"][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        modules = self.__dict__.get("_modules")
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value)}' as child module '{name}'")
            modules[name] = value
        else:
            for key in ("_parameters", "_modules", "_buffers"):
                if key in self.__dict__ and name in self.__dict__[key]:
                    self.__dict__[key][name] = value
                    return
            object.__setattr__(self, name, value)


class Container(Module):
    pass


class Sequential(Container):
    """Sequential container of modules."""

    def __init__(self, *module_list):
        super().__init__()
        if len(module_list) == 1 and isinstance(module_list[0], OrderedDict):
            for key, module in module_list[0].items():
                self.add_module(str(key), module)
        elif len(module_list) == 1 and isinstance(module_list[0], list):
            for idx, module in enumerate(module_list[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(module_list):
                self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def _get_abs_string_index(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        abs_idx = self._get_abs_string_index(idx)
        return setattr(self, str(abs_idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                del self._modules[str(k)]
        else:
            del self._modules[self._get_abs_string_index(idx)]
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):
    """Holds submodules in an ordered dictionary."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        self._modules.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def update(self, modules):
        if isinstance(modules, (dict, OrderedDict, ModuleDict)):
            for k, v in modules.items():
                self[k] = v
        else:
            for k, v in modules:
                self[k] = v


class Parameter(Module):
    """Module holding a parameter tensor."""

    def __init__(self, param, trainable=True):
        super().__init__()
        if not isinstance(param, prix.Array):
            param = jnp.asarray(param, dtype=jnp.float32)
        if trainable:
            self.register_parameter("data", param)
        else:
            self.register_buffer("data", param)
        self.encrypted = isinstance(param, prix.Array)

    def forward(self, x=None):
        return self.data


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Linear(Module):
    """Applies a linear transformation: y = x @ weight.T + bias."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(max(1, in_features))
        w = jax.random.uniform(
            _next_init_key(), (out_features, in_features), dtype=jnp.float32, minval=-bound, maxval=bound
        )
        self.register_parameter("weight", w)
        if bias:
            b = jax.random.uniform(
                _next_init_key(), (out_features,), dtype=jnp.float32, minval=-bound, maxval=bound
            )
            self.register_parameter("bias", b)

    def forward(self, x):
        w_t = self.weight.t() if hasattr(self.weight, "t") else jnp.swapaxes(self.weight, -1, -2)
        output = x.matmul(w_t) if hasattr(x, "matmul") else jnp.matmul(x, w_t)
        if "bias" in self._parameters and self.bias is not None:
            output = output.add(self.bias) if hasattr(output, "add") else (output + self.bias)
        return output


class MatMul(Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            self.register_parameter("weight", weight)

    def forward(self, x):
        if "weight" in self._parameters:
            return x.matmul(self.weight)
        assert isinstance(x, (list, tuple)) and len(x) == 2
        return x[0].matmul(x[1])


class MultiheadAttention(Module):
    """
    Multi-head scaled dot-product attention over (batch, seq, embed_dim) inputs.

    Mirrors the core of ``torch.nn.MultiheadAttention`` with ``batch_first=True``, with two
    differences: ``forward`` returns only the attention output (attention weights are not
    returned), and ``attn_mask`` must be a public additive mask broadcastable to
    (batch, num_heads, query_len, key_len). Dropout is not supported (inference only).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not batch_first:
            raise NotImplementedError("only batch_first=True (batch, seq, embed_dim) inputs are supported")
        if dropout:
            raise NotImplementedError("attention dropout is not supported")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x):
        b, t, _ = x.shape
        shape = (b, t, self.num_heads, self.head_dim)
        if isinstance(x, prix.Array):
            return x.reshape(*shape).transpose(1, 2)
        return jnp.swapaxes(jnp.reshape(x, shape), 1, 2)

    def forward(self, query, key=None, value=None, attn_mask=None):
        key = query if key is None else key
        value = key if value is None else value
        b, t, _ = query.shape
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        scale = self.head_dim ** -0.5
        if isinstance(q, prix.Array):
            scores = q.matmul(k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            ctx = scores.softmax(dim=-1).matmul(v)
            ctx = ctx.transpose(1, 2).reshape(b, t, self.embed_dim)
        else:
            scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            ctx = jnp.matmul(jax.nn.softmax(scores, axis=-1), v)
            ctx = jnp.reshape(jnp.swapaxes(ctx, 1, 2), (b, t, self.embed_dim))
        return self.out_proj(ctx)


class Embedding(Module):
    """Secret-shared embedding table evaluated via one-hot Beaver lookup."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        w = jax.random.normal(_next_init_key(), (vocab_size, embed_dim), dtype=jnp.float32) * 0.02
        self.register_parameter("weight", w)

    def forward(self, x):
        if isinstance(x, prix.Array):
            return x.evaluate_embed(self.weight)
        return self.weight[jnp.asarray(x, dtype=jnp.int64)]


class LayerNorm(Module):
    """Layer normalization over the last dimension(s)."""

    def __init__(self, shape, eps=1e-5, bias=True):
        super().__init__()
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        self.shape = shape_tuple
        self.eps = eps
        self.register_parameter("weight", jnp.ones(shape_tuple, dtype=jnp.float32))
        if bias:
            self.register_parameter("bias", jnp.zeros(shape_tuple, dtype=jnp.float32))

    def forward(self, x):
        b = self.bias if "bias" in self._parameters else None
        if isinstance(x, prix.Array):
            return x.layernorm(self.weight, b, eps=self.eps)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        out = (x - mean) / jnp.sqrt(var + self.eps) * self.weight
        if b is not None:
            out = out + b
        return out


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (used in Llama)."""

    def __init__(self, dim_or_eps, eps_or_size=1e-5):
        super().__init__()
        if isinstance(dim_or_eps, float) and isinstance(eps_or_size, int):
            self.eps = dim_or_eps
            dim = eps_or_size
        else:
            dim = dim_or_eps
            self.eps = eps_or_size
        shape_tuple = (dim,) if isinstance(dim, int) else tuple(dim)
        self.register_parameter("weight", jnp.ones(shape_tuple, dtype=jnp.float32))

    def forward(self, x):
        if isinstance(x, prix.Array):
            return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).inv_sqrt()) * self.weight
        return (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)) * self.weight


class Conv1d(Module):
    """1D convolution module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
        bound = 1.0 / math.sqrt(max(1, in_channels * k))
        w = jax.random.uniform(
            _next_init_key(),
            (out_channels, in_channels // groups, k),
            dtype=jnp.float32,
            minval=-bound,
            maxval=bound,
        )
        self.register_parameter("weight", w)
        if bias:
            b = jax.random.uniform(
                _next_init_key(), (out_channels,), dtype=jnp.float32, minval=-bound, maxval=bound
            )
            self.register_parameter("bias", b)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        out = x.conv1d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self._parameters and self.bias is not None:
            out = out.add(self.bias.unsqueeze(-1))
        return out


class Conv2d(Module):
    """2D convolution module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        bound = 1.0 / math.sqrt(max(1, in_channels * k[0] * k[1]))
        w = jax.random.uniform(
            _next_init_key(),
            (out_channels, in_channels // groups, k[0], k[1]),
            dtype=jnp.float32,
            minval=-bound,
            maxval=bound,
        )
        self.register_parameter("weight", w)
        if bias:
            b = jax.random.uniform(
                _next_init_key(), (out_channels,), dtype=jnp.float32, minval=-bound, maxval=bound
            )
            self.register_parameter("bias", b)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        out = x.conv2d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self._parameters and self.bias is not None:
            out = out.add(self.bias.unsqueeze(-1).unsqueeze(-1))
        return out


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.relu() if isinstance(x, prix.Array) else jax.nn.relu(x)


class GELU(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.gelu() if isinstance(x, prix.Array) else jax.nn.gelu(x, approximate=False)


class SiLU(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.silu() if isinstance(x, prix.Array) else jax.nn.silu(x)


SILU = SiLU


class Tanh(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.tanh() if isinstance(x, prix.Array) else jnp.tanh(x)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        if isinstance(x, prix.Array):
            return x.hardtanh(self.min_val, self.max_val)
        return jnp.clip(x, self.min_val, self.max_val)


class ReLU6(Hardtanh):
    def __init__(self, inplace=False):
        super().__init__(min_val=0.0, max_val=6.0, inplace=False)


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid() if isinstance(x, prix.Array) else jax.nn.sigmoid(x)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim) if isinstance(x, prix.Array) else jax.nn.softmax(x, axis=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.log_softmax(self.dim) if isinstance(x, prix.Array) else jax.nn.log_softmax(x, axis=self.dim)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return x.dropout(self.p, training=self.training, inplace=self.inplace)


Dropout2d = Dropout
Dropout3d = Dropout


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return x.avg_pool2d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return x.max_pool2d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return x.adaptive_avg_pool2d(self.output_size)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return x.adaptive_max_pool2d(self.output_size)
