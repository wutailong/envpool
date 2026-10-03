# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Explicit runtime selection shared by the standalone benchmark tools."""

import importlib
import os
import sys
from pathlib import Path

VARIANTS = ("original", "rebuilt", "candidate")


def load_runtime(root: str | None):
    """Load dependencies, optionally enforcing an explicit EnvPool package root."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    package_root = Path(root).expanduser().resolve() if root else None
    if package_root:
        if not (package_root / "envpool" / "__init__.py").is_file():
            raise ValueError("--envpool-root must contain the envpool package")
        sys.path.insert(0, str(package_root))
    numpy = importlib.import_module("numpy")
    envpool = importlib.import_module("envpool")
    if package_root and not Path(envpool.__file__).resolve().is_relative_to(
        package_root / "envpool"
    ):
        raise RuntimeError("EnvPool import did not use --envpool-root")
    return numpy, envpool


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    result = int(value)
    if result <= 0:
        raise ValueError("expected a positive integer")
    return result
