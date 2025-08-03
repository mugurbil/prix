#!/usr/bin/env python3

# Copyright (c) Mehmet Ugurbil.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import re
import sys

import setuptools


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "prix"))

# Read description and requirements.
with open("README.md", encoding="utf8") as f:
    readme = f.read()
with open("requirements.txt") as f:
    reqs = f.read()

# Get version string from module
init_path = os.path.join(os.path.dirname(__file__), "prix/__init__.py")
with open(init_path, "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# Set key package information.
DISTNAME = "prix"
DESCRIPTION = "PriX: Private Accelerated Transformers."
LONG_DESCRIPTION = readme
AUTHOR = "Mehmet Ugurbil"
LICENSE = "MIT licensed, as found in the LICENSE file"
REQUIREMENTS = (reqs.strip().split("\n"),)
VERSION = version

# Run installer.
if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Sorry, Python >=3.10 is required for PriX.")

    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        dependency_links=[],
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/mugurbil/prix",
        author=AUTHOR,
        license=LICENSE,
        tests_require=["pytest"],
        data_files=[("configs", ["configs/default.yaml"])],
    )
