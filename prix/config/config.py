#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from contextlib import contextmanager

import yaml


class ConfigNode(dict):
    """Dictionary wrapper supporting attribute-style dot access."""

    def __init__(self, data=None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            return cls(value)
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc

    def __setattr__(self, name, value):
        if "." in name:
            keys = name.split(".")
            node = self
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], ConfigNode):
                    node[key] = ConfigNode()
                node = node[key]
            node[keys[-1]] = self._wrap(value)
        else:
            self[name] = self._wrap(value)


def _deep_merge(base, override):
    """Recursively merges `override` dict into a copy of `base` dict."""
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


class PrixConfig:
    """Configuration object used to store configurable parameters for PriX."""

    _REPO_CONFIG_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../configs")
    )
    _PKG_CONFIG_DIR = os.path.normpath(os.path.dirname(__file__))
    _REPO_CONFIG_PATH = os.path.join(_REPO_CONFIG_DIR, "default.yaml")
    _PKG_CONFIG_PATH = os.path.join(_PKG_CONFIG_DIR, "default.yaml")
    _DEFAULT_CONFIG_PATH = (
        _REPO_CONFIG_PATH if os.path.exists(_REPO_CONFIG_PATH) else _PKG_CONFIG_PATH
    )

    _PRESET_FILES = {
        "default": "default.yaml",  # Wave bior(5,3) FSS LUTs + SIGMA TR indexing + EGK product truncation
        "curl": "curl.yaml",
        "wave": "wave.yaml",
        "sigma": "sigma.yaml",
        "fss": "sigma.yaml",
        "poly": "poly.yaml",
        "puma": "poly.yaml",
        "nimbus": "poly.yaml",
        "shaft": "shaft.yaml",
        "crypten": "approximations.yaml",
        "approximations": "approximations.yaml",
        "fission": "fission.yaml",
    }

    def __init__(self, config_file=None):
        object.__setattr__(self, "config", ConfigNode())
        self.load_config(config_file)

    @classmethod
    def get_default_config_path(cls):
        return cls._DEFAULT_CONFIG_PATH

    @classmethod
    def resolve_config_path(cls, config_file=None):
        """Resolves a preset name ('curl', 'wave', 'sigma', 'poly', 'shaft', 'crypten', 'fission') or file path."""
        if config_file is None:
            return cls._DEFAULT_CONFIG_PATH
        key = str(config_file).strip().lower()
        if key.endswith(".yaml") and not os.path.exists(config_file):
            key = os.path.basename(key)[:-5]
        if key in cls._PRESET_FILES:
            fname = cls._PRESET_FILES[key]
            repo_candidate = os.path.join(cls._REPO_CONFIG_DIR, fname)
            if os.path.exists(repo_candidate):
                return repo_candidate
            return os.path.join(cls._PKG_CONFIG_DIR, fname)
        return config_file

    def load_config(self, config_file=None):
        """Loads config from a preset name ('curl', 'wave', 'sigma', 'poly', 'shaft', 'crypten', 'fission') or yaml path."""
        resolved = self.resolve_config_path(config_file)
        with open(self._DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as stream:
            base_dict = yaml.safe_load(stream)
        if os.path.abspath(resolved) != os.path.abspath(self._DEFAULT_CONFIG_PATH):
            with open(resolved, "r", encoding="utf-8") as stream:
                override_dict = yaml.safe_load(stream)
            config_dict = _deep_merge(base_dict, override_dict)
        else:
            config_dict = base_dict
        object.__setattr__(self, "config", ConfigNode(config_dict))
        return self

    def set_mode(self, mode):
        """Switches PriX's MPC approximation backend ('curl', 'wave', 'sigma', 'poly', 'shaft', 'crypten', 'fission')."""
        return self.load_config(mode)

    def set_config(self, config):
        if isinstance(config, PrixConfig):
            object.__setattr__(self, "config", copy.deepcopy(config.config))
        elif isinstance(config, dict):
            object.__setattr__(self, "config", ConfigNode(copy.deepcopy(config)))
        else:
            object.__setattr__(self, "config", config)

    def __getattribute__(self, name):
        if name in (
            "config",
            "load_config",
            "set_mode",
            "set_config",
            "get_default_config_path",
            "resolve_config_path",
            "temp_override",
        ):
            return object.__getattribute__(self, name)
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            cfg_node = object.__getattribute__(self, "config")
            keys = name.split(".")
            result = cfg_node
            for key in keys:
                result = getattr(result, key)
            return result

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "config":
            object.__setattr__(self, name, ConfigNode._wrap(value))
            return
        cfg_node = object.__getattribute__(self, "config")
        setattr(cfg_node, name, value)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    @contextmanager
    def temp_override(self, override_dict, value=None):
        """Context manager to temporarily override config parameters."""
        old_config = copy.deepcopy(object.__getattribute__(self, "config"))
        try:
            if isinstance(override_dict, str):
                override_dict = {override_dict: value}
            for k, v in override_dict.items():
                self.__setattr__(k, v)
            yield
        finally:
            object.__setattr__(self, "config", old_config)
