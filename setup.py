#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

from pathlib import Path
from typing import List

from setuptools import find_packages, setup

extra_files: List[str] = [
    "primitives/data/geometric_product.pt",
    "primitives/data/outer_product.pt",
]

_CURRENT_FOLDER = Path(__file__).parent
_PYTHON_FILES_IN_SCRIPTS = list((_CURRENT_FOLDER / "scripts").glob("*.py"))

scripts = [
    s.relative_to(_CURRENT_FOLDER).as_posix()
    for s in _PYTHON_FILES_IN_SCRIPTS
    if "__" not in str(s)
]
package_list = find_packages(exclude=["tests*", "docker", "scripts"])


def get_version(rel_path: Path) -> str:
    """Extracts version information.

    Inspired by item "1." @ https://packaging.python.org/guides/single-sourcing-package-version/.
    """
    package_init_path = Path(__file__).parent / rel_path
    with package_init_path.open() as f:
        for line in f:
            if line.startswith("__version__"):
                delimiter = '"' if '"' in line else "'"
                return line.split(delimiter)[1]
        raise RuntimeError(f"Unable to find version string in {package_init_path}.")


setup(
    name="GATr",
    version=get_version(Path("gatr") / "__init__.py"),
    author="Johann Brehmer, Pim de Haan, Sönke Behrends, and Taco Cohen",
    author_email="jbrehmer@qti.qualcomm.com",
    maintainer="Johann Brehmer",
    maintainer_email="jbrehmer@qti.qualcomm.com",
    url="https://github.com/Qualcomm-AI-research/geometric-algebra-transformer",
    description="The Geometric Algebra Transformer, a universal neural network architecture for geometric data",
    license="See notice in license file",
    packages=package_list,
    scripts=scripts,
    package_data={"gatr": extra_files},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "einops",
        "numpy<1.25",
        "opt_einsum @ git+https://github.com/dgasmith/opt_einsum.git@1a984b7b75f3e532e7129f6aa13f7ddc3da66e10",
        "torch>=2.0",
        "xformers",
    ],
)
