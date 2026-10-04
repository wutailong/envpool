# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Run six existing positive Python/API/render checks on the explicitly staged clients."""

import argparse
import hashlib
import importlib
import json
import sys
import unittest
from pathlib import Path

TESTS = (
    "envpool.classic_control.classic_control_test._ClassicControlEnvPoolTest.test_cartpole",
    "envpool.classic_control.classic_control_test._ClassicControlEnvPoolTest.test_cartpole_gymnasium_vector_wrapper",
    "envpool.classic_control.classic_control_render_test.ClassicControlRenderTest.test_render_is_batch_consistent_and_state_invariant",
    "envpool.dummy.dummy_py_envpool_test._DummyEnvPoolTest.test_config",
    "envpool.dummy.dummy_py_envpool_test._DummyEnvPoolTest.test_spec",
    "envpool.dummy.dummy_py_envpool_test._DummyEnvPoolTest.test_env_seed_overrides_sequential_seeding",
)


def main():
    """Verify imports and frozen inputs before and after ordinary positive tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be a fresh file")
    root = args.runtime.resolve()
    package = root / "envpool"
    manifest = json.loads((root / "manifest.json").read_text())

    def verify():
        for relative, row in manifest["native"].items():
            actual = hashlib.sha256(
                (package / relative).read_bytes()
            ).hexdigest()
            if actual != row["sha256"]:
                raise ValueError("native bytes differ from staging manifest")
        for relative, expected in manifest["python_source_hashes"].items():
            actual = hashlib.sha256(
                (package / relative).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise ValueError("Python source differs from staging manifest")

    verify()
    sys.path.insert(0, str(root))
    envpool = importlib.import_module("envpool")
    if Path(envpool.__file__).resolve() != package / "__init__.py":
        raise ValueError("wrong EnvPool package imported")
    imports = {}
    for name in (
        "envpool.classic_control.classic_control_envpool",
        "envpool.dummy.dummy_envpool",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        relative = path.relative_to(package).as_posix()
        if not manifest["native"][relative]["rebuilt"]:
            raise ValueError("smoke client was not rebuilt")
        imports[name] = manifest["native"][relative]["sha256"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    verify()
    report = {
        "tests": result.testsRun,
        "passed": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "selected_tests": TESTS,
        "rebuilt_imports": imports,
        "frozen_native_and_python_unchanged": True,
        "performance_evidence": False,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if not result.wasSuccessful() or result.testsRun != len(TESTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
