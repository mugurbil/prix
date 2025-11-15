#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import prix


class Module:
    """
    Base Module class that mimics the torch.nn.Module class.
    """

    # allow for versioning of modules:
    _version = 1
    SUPPORTS_PLAINTEXT_INPUTS = False

    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.encrypted = False
        self.train()

    def __repr__(self):
        encrypted_str = "encrypted" if self.encrypted else "unencrypted"
        return f"{type(self).__name__} {encrypted_str} module"

    def forward(self, *args, **kwargs):
        """Perform forward pass on model."""
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        """Sets the module in the specified training mode."""
        for param in self.parameters():
            param.requires_grad = mode
        self.training = mode

        # Recursively set train mode
        for module in self.children():
            module.train(mode=mode)
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)

    def children(self):
        """Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for _, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module and the module itself.
        Yields:
            (string, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def register_module(self, name, module):
        """Registers child module in the module."""
        self._modules[name] = module

    def modules(self):
        """Returns an iterator over all modules in the network.
        Yields:
            Module: a module in the network
        Note:
            Duplicate modules are returned only once.
        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=""):
        """Returns iterator over named modules (non-recursively)."""
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
        """
        Register parameter in the module. This function cannot register
        parameters in child modules.
        """
        if name in self._parameters or hasattr(self, name):
            raise ValueError("Parameter or field %s already exists." % name)
        if requires_grad:
            param.requires_grad = requires_grad
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter(self, name, param):
        """
        Sets value of parameter in the module. This function cannot set
        parameters in child modules.
        """
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter_from_shares(self, name, share, **kwargs):
        """
        Sets value of parameter in the module from shares. This functionality is
        only for MPC-encrypted models.

        Supported named arguments for `MPCTensor` parameters include the `precision`
        of the encoder (default = `None`), the rank of the `src` (default = 0),
        and the `ptype` of the shares (default = `curl.mpc.arithmetic`).

        This function cannot set the parameters in child modules.
        """

        # functionality is only supported when parameters are MPCTensors:
        assert self.encrypted, "can only set parameters from shares in encrypted models"
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        cls = type(self._parameters[name])
        assert hasattr(
            self._parameters[name], "from_shares"
        ), "parameter type {} does not supporting setting from shares".format(cls)

        # load parameters from shares:
        self._parameters[name] = cls.from_shares(share, **kwargs)
        setattr(self, name, self._parameters[name])

    def parameters(self, recurse=True):
        """Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            Array: module parameter
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, recurse=True, prefix=None):
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter and the parameter itself.
        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            (string, Array): Tuple containing the name and parameter
        """
        for name, param in self._parameters.items():
            param_name = name if prefix is None else prefix + "." + name
            yield param_name, param
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_parameters(recurse=recurse, prefix=pre)

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for param in self.parameters():
            param.grad = None

    def update_parameters(self, learning_rate, grad_threshold=100):
        """Performs gradient step on parameters.

        Parameters:
            grad_threshold - Because arithmetic operations can extremely rarely
                    return large incorrect results, we zero-out all elements
                    with magnitude larger than this given threshold. To turn
                    off thresholding, set to `None`.
        """
        assert self.training, "module not in training mode"
        for param in self.parameters():
            if param.grad is None:
                continue
            if not self.encrypted and isinstance(param.grad, prix.Array):
                raise RuntimeError(
                    "Cannot update parameters of unencrypted "
                    "model using encrypted gradients. Encrypt "
                    "model before updating parameters."
                )

            # Threshold gradients to prevent gradient explosion from wrap overflow
            if self.encrypted and grad_threshold is not None:
                # Compute based on square value since abs is more expensive
                square_threshold = grad_threshold * grad_threshold
                grad = param.grad.mul(param.grad.square().lt(square_threshold))
            else:
                grad = param.grad

            param.sub_(grad.mul_(learning_rate))

    def register_buffer(self, name, buffer):
        """
        Register buffer in the module. Buffers are encrypted like parameters but
        they are not updated by parameter updates.
        """
        if name in self._buffers or hasattr(self, name):
            raise ValueError("Buffer or field %s already exists." % name)
        buffer.requires_grad = False
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def set_buffer(self, name, buffer):
        """
        Sets value of buffer in the module. This function cannot set the
        parameters in child modules.
        """
        if name not in self._buffers or not hasattr(self, name):
            raise ValueError("Buffer %s does not exist." % name)
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def add_module(self, name, module):
        """Adds and registers a submodule with a given name"""
        assert name not in self._modules.keys(), "Module %s already exists." % name
        self.register_module(name, module)
