# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Stage two fresh clients with published Python source and original support modules."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    """Hash file bytes, following explicitly selected support-module symlinks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Create a new mixed support package without replacing any existing runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-package", type=Path, required=True)
    parser.add_argument("--classic-binary", type=Path, required=True)
    parser.add_argument("--dummy-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve() / "envpool"
    original = args.original_package.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("output must be a fresh directory")
    if output.is_relative_to(args.source.resolve()) or output.is_relative_to(
        original
    ):
        raise ValueError("output must be outside source and original package")
    if not (source / "__init__.py").is_file():
        raise ValueError("source must contain the published envpool package")
    if not (original / "__init__.py").is_file():
        raise ValueError(
            "original-package must be the original envpool directory"
        )
    if list(source.rglob("*.so")):
        raise ValueError("source checkout unexpectedly contains native outputs")
    replacements = {
        "classic_control/classic_control_envpool.so": args.classic_binary.resolve(),
        "dummy/dummy_envpool.so": args.dummy_binary.resolve(),
    }
    for path in replacements.values():
        if not path.is_file() or path.is_relative_to(original):
            raise ValueError(
                "both rebuilt clients must be separate regular files"
            )
    support = {
        path.relative_to(original).as_posix(): path.resolve()
        for path in original.rglob("*.so")
        if path.relative_to(original).as_posix() not in replacements
    }
    if not support:
        raise ValueError("original support native modules were not found")
    original_hashes = {key: digest(path) for key, path in support.items()}
    output.mkdir(parents=True, exist_ok=False)
    package = output / "envpool"
    shutil.copytree(
        source, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    manifest = {}
    for relative, path in (support | replacements).items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        fresh = relative in replacements
        if fresh:
            shutil.copy2(path, target)
        else:
            target.symlink_to(path)
        manifest[relative] = {
            "sha256": digest(target),
            "rebuilt": fresh,
            "source": str(path),
        }
    for relative, expected in original_hashes.items():
        if digest(support[relative]) != expected:
            raise RuntimeError("original support module changed while staging")
    python_files = {
        path.relative_to(source).as_posix(): digest(path)
        for path in source.rglob("*.py")
    }
    for relative, expected in python_files.items():
        if digest(package / relative) != expected:
            raise RuntimeError("staged Python source differs from checkout")
    result = {
        "scope": "Two fresh clients; other families use original support modules",
        "native": manifest,
        "python_source_hashes": python_files,
        "source_package": str(source),
        "original_package": str(original),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rebuilt": 2, "original_support": len(support)}))


if __name__ == "__main__":
    main()
