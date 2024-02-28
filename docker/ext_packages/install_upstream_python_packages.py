# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Installs upstream Python packages which are not easily installable via pip.

After a vanilla checkout, the structure of those repositories is roughly as follows:

```text
repository-root/
├── experiments
├── models
├── ... more folders and files ...
└── setup.py
```

We modify those repositories to look as follows:

```text
repository-root/
├── pyproject.toml
└── src
    └── project-name
        ├── experiments
        ├── models
        └── ... more folders and files, possibly some omitted ...
```
where `setup.py` was deleted and a minimalist `pyproject.toml` was added.

After this modification, they are pip-installable, and importable from `project-name`.

To prevent large Docker images and multiple dependency resolution steps, the requirements of those
repositories were directly added to `docker/requirements.txt`.
"""

import contextlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional

from rope.base.project import Project
from rope.refactor import move

logging.basicConfig(level=logging.DEBUG)

SCRIPT_FOLDER = Path(__file__).resolve().parent
SRC_FOLDER = "src"
INIT_PY = "__init__.py"
SEGNN_RELEVANT_FOLDER = "models"


class PythonPackage:
    """Functionality related to installing the Python code given as a repo."""

    def __init__(
        self,
        name: str,
        clone_url: str,
        checksum_sha: str,
        checkout_base_dir: Path,
        patch_file: Optional[Path] = None,
        ordered_refactoring: Optional[List[str]] = None,
    ) -> None:
        self._name = name
        self._clone_url = clone_url
        self._checksum_sha = checksum_sha
        self._base_dir = checkout_base_dir
        self._patch_file = patch_file
        self._ordered_refactoring = ordered_refactoring

        self._logger = logging.getLogger(self._name)

    def install_package(self) -> None:
        """Performs all steps necessary up to and including installation of the package."""
        self.clone_from_upstream()
        self.checkout_commit()
        self.apply_patches()

        self.remove_packaging_files()
        self.add_new_pyproject_file()

        self.add_init_files()
        self.move_to_project_folder()
        self.move_to_src_folder()

        self.pip_install_refactored_package()
        self._logger.info("Installation complete.")

    def _get_checkout_path(self) -> Path:
        return self._base_dir / self._name

    def _get_refactored_path(self) -> Path:
        return self._get_checkout_path() / SRC_FOLDER / self._name

    def clone_from_upstream(self) -> None:
        """Clones the git repository from the clone URL provided upon instantiation."""
        self._logger.info(f"Cloning git repo at {self._clone_url}")
        run_successfully(cmd=["git", "clone", self._clone_url, str(self._get_checkout_path())])

    def checkout_commit(self) -> None:
        """Checks out the commit hash provided upon instantiation."""
        cmd = ["git", "checkout", self._checksum_sha]
        run_successfully(cmd, cwd=self._get_checkout_path())

    def apply_patches(self) -> None:
        """Applies patch from a patch file, if provided upon instantiation."""
        if self._patch_file is not None:
            run_successfully(["git", "apply", str(self._patch_file)], cwd=self._get_checkout_path())

    def add_init_files(self) -> None:
        """Adds an `__init__.py` file for ever subfolder of checkout path (if missing)."""
        for folder, _, _ in os.walk(self._get_checkout_path()):
            folder_path = Path(folder)
            if ".git" in folder_path.parts:
                continue
            init_file = folder_path / INIT_PY
            if not init_file.exists():
                init_file.touch()

    def remove_packaging_files(self) -> None:
        """Removes all packaging files."""
        for filename in ["setup.py", "setup.cfg", "pyproject.toml"]:
            file = self._get_checkout_path() / filename
            file.unlink(missing_ok=True)

    def add_new_pyproject_file(self) -> None:
        """Adds a generic `pyproject.toml` file to the project root."""
        file = self._get_checkout_path() / "pyproject.toml"
        file.write_text(f'[project]\nname = "{self._name}"\nversion = "1.0.0"')

    def move_to_project_folder(self) -> None:
        """Moves Python files and folders into folder with the project's name, adapts imports."""
        # Creates folder `<checkout_base> / project-name / project-name`
        self._logger.info("Moving Python files and folders to project subfolder.")
        project_subfolder = self._get_checkout_path() / self._name
        project_subfolder.mkdir()
        self._move_files(project_subfolder=project_subfolder)

    def _move_files(self, project_subfolder: Path) -> None:
        """Use rope to move files."""

        # Move all files / dirs using rope - this takes care of refactoring the imports, too.
        file_objs = list(self._get_checkout_path().iterdir())
        files = self._sort_files_by_priority([file for file in file_objs if file.is_file()])
        dirs = [file for file in file_objs if file.is_dir()]
        rope_project = Project(str(self._get_checkout_path()))
        rope_target = rope_project.get_resource(self._name)
        for file_obj in files + dirs:
            if not self.should_refactor(file_obj, target_path=project_subfolder):
                continue
            rel_pkg_path = file_obj.relative_to(self._get_checkout_path())
            rope_source = rope_project.get_resource(str(rel_pkg_path))
            mover = move.create_move(rope_project, rope_source)
            changes = mover.get_changes(rope_target)
            rope_project.do(changes)
        # The `__init__.py` in the base folder was not yet moved.
        init_py_src = self._get_checkout_path() / INIT_PY
        init_py_target = self._get_checkout_path() / self._name / INIT_PY
        init_py_src.rename(init_py_target)

    def _sort_files_by_priority(self, files: List[Path]) -> List[Path]:
        """Sorts files for priority refactoring.

        This is necessary as else rope will get this wrong in some cases:
        It can, for example, make a difference if we first refactor `module1.py` and then
        `module2.py`, or the other way around.
        """
        if self._ordered_refactoring is None:
            return files
        files_without_priority = [x for x in files if x.name not in self._ordered_refactoring]
        base = self._get_checkout_path()
        files_with_priority = [base / file for file in self._ordered_refactoring]
        if not set(files_with_priority + files_without_priority) == set(files):
            raise RuntimeError("Could not reproduce the full list of files to refactor, aborting.")
        return files_with_priority + files_without_priority

    def move_to_src_folder(self) -> None:
        """Moves package code to `src` folder.

        That is, we move the code from `<checkout_base_dir> / project-name / project-name`
        to  `<checkout_base_dir> / project-name / src / project-name`.
        This allows easy installation with pip.
        """

        src_folder = self._get_checkout_path() / SRC_FOLDER
        src_folder.mkdir()
        shutil.move(str(self._get_checkout_path() / self._name), src_folder)

    def pip_install_refactored_package(self) -> None:
        """Installs the refactored package using `pip`."""
        self._logger.info("pip-installing refactored package.")
        run_successfully(["pip", "install", str(self._get_checkout_path())])

    @staticmethod
    def should_refactor(file_obj: Path, target_path: Path) -> bool:
        """Decides whether a file object should be refactored or not."""
        refactor = True
        if file_obj.name.startswith("."):
            refactor = False
        if not file_obj.is_dir():
            if not file_obj.suffix == ".py" or file_obj.name == "__init__.py":
                refactor = False
        # Do not move project folder into itself
        if file_obj.name == target_path.name:
            refactor = False
        return refactor


class SegnnPackage(PythonPackage):
    """Special installation for the segnn package required.

    The reason is that rope fails to move all files (utils.py).
    As we are only interested in the models/* files, we restrict to those.
    """

    def _move_files(self, project_subfolder: Path) -> None:
        """Move python files directly."""
        for file_obj in (self._get_checkout_path() / SEGNN_RELEVANT_FOLDER).iterdir():
            shutil.move(str(file_obj), str(project_subfolder))


def install_packages() -> None:
    """Refactors flat Python projects into installable form."""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for package in get_packages(tmp_path=tmp_path):
            package.install_package()


def get_packages(tmp_path: Path) -> List[PythonPackage]:
    """Gets list of all Python packages to install."""
    packages = [
        SegnnPackage(
            name="segnn",
            clone_url="https://github.com/RobDHess/Steerable-E3-GNN.git",
            checksum_sha="1b95898f6f18204b510ae127d7f38cd29f610f4d",
            checkout_base_dir=tmp_path,
        ),
        PythonPackage(
            name="se3_transformer",
            clone_url="https://github.com/FabianFuchsML/se3-transformer-public.git",
            checksum_sha="3ab53da4a501d4aedf128453b91290a4a5d65a45",
            checkout_base_dir=tmp_path,
        ),
        PythonPackage(
            name="coronary_mesh_convolution",
            clone_url="https://github.com/sukjulian/coronary-mesh-convolution",
            checksum_sha="e6234e108b4d4399d8fe948f6f7fbf36e66638fe",
            checkout_base_dir=tmp_path,
            # Prevents installation of vtk
            patch_file=SCRIPT_FOLDER / "coronary-mesh-convolution_vtk-patch.patch",
            ordered_refactoring=["datasets.py", "data.py"],
        ),
    ]
    return packages


def run_successfully(cmd: List[str], **kwargs: Any) -> None:
    """Executes command in a subprocess, and in case of errors, prints stdout/stderr and fails."""
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", check=False, **kwargs
    )
    if proc.returncode != 0:
        print("stdout:", proc.stdout, "stderr:", proc.stderr, sep="\n")
    proc.check_returncode()


def verify_installation_through_imports() -> None:
    """Tests if installation was successful by importing the installed modules."""

    # pylint:disable=import-error,import-outside-toplevel,unused-import
    logger = logging.getLogger("pip installation")
    logger.info("Now running some imports to test installation...")

    # Mute output from imports to avoid distracting information.
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        from coronary_mesh_convolution import datasets  # type: ignore[import]
        from coronary_mesh_convolution.data import MultiscaleData  # type: ignore[import]
        from se3_transformer.equivariant_attention.fibers import Fiber  # type: ignore[import]
        from se3_transformer.equivariant_attention.from_se3cnn.SO3 import (
            irr_repr,  # type: ignore[import]
        )
        from se3_transformer.equivariant_attention.modules import GSE3Res  # type: ignore[import]
        from se3_transformer.utils import utils_profiling  # type: ignore[import]
        from segnn.balanced_irreps import BalancedIrreps  # type: ignore[import]
        from segnn.segnn.segnn import SEGNN  # type: ignore[import]

    logger.info("Imports successful!")


if __name__ == "__main__":
    install_packages()
    verify_installation_through_imports()
