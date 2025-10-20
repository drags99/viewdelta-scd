# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

# Package metadata
NAME = "ViewDelta"
VERSION = "0.1"
DESCRIPTION = "Text Prompted Change Detection"


# Required dependencies
REQUIRED_PACKAGES = [
    "scipy",
    "scikit-learn",
    "pandas",
    "opencv-python",
    "kornia",
    "einops",
    "matplotlib",
    "transformers",
    "sentencepiece",
    "protobuf",
    "wandb",
    "deepspeed",
    "lightning",
    "pyparsing",
    "tables",
    "timm==1.0.13",
]


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
)
