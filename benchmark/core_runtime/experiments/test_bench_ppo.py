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
"""Dependency-free checks; deliberately do not import Torch, NumPy or EnvPool."""

import contextlib
import io
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bench_ppo


class FakeArray:
    """Supply exact bytes through the fingerprint's explicit dependency seam."""

    def __init__(self, data, dtype="<f8", shape=(1,)):
        """Store metadata without importing a scientific runtime."""
        self.data = data
        self.dtype = SimpleNamespace(kind=dtype[1], str=dtype)
        self.shape = shape

    def tobytes(self, order):
        """Enforce canonical C-order serialization in the tested helper."""
        if order != "C":
            raise ValueError("expected canonical array bytes")
        return self.data


class FakeScalar:
    """Distinguish NumPy-style scalar tagging from array tagging."""


FAKE_NUMPY = SimpleNamespace(
    ndarray=FakeArray,
    generic=FakeScalar,
    asarray=lambda value: value,
    isfinite=lambda value: SimpleNamespace(all=lambda: True),
)


class PPOHelperTests(unittest.TestCase):
    """Validate the experiment contract and result helpers without running PPO."""

    def args(self, *extras):
        """Build a parse-only command line with dummy paths."""
        return bench_ppo.parse_args([
            "--package",
            "/not/imported/envpool",
            "--label",
            "baseline",
            "--output",
            "unused.json",
            *extras,
        ])

    def test_default_work_budget(self):
        """Use the parity workload by default and support shorter smoke runs."""
        args = self.args()
        self.assertEqual(args.iterations, 100)
        self.assertEqual(args.warmup, 2)
        cfg = bench_ppo.load_config(args.config, args.iterations)
        self.assertEqual(cfg["updates"] * cfg["steps_per_collect"], 256_000)
        self.assertEqual(
            cfg["updates"]
            * cfg["steps_per_collect"]
            * cfg["repeat_per_collect"]
            // cfg["batch_size"],
            8000,
        )
        self.assertEqual(bench_ppo.load_config(args.config, 3)["updates"], 3)

    def test_invalid_arguments(self):
        """Reject empty budgets, unprimed timings, invalid hashes and path labels."""
        for extra in (
            ["--iterations", "0"],
            ["--iterations", "x"],
            ["--warmup", "0"],
            ["--warmup", "-1"],
            ["--native-sha256", "abc"],
            ["--label", "/private/path"],
        ):
            with (
                self.subTest(extra=extra),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    self.args(*extra)
        self.assertEqual(
            self.args("--native-sha256", "A" * 64).native_sha256, "a" * 64
        )

    def test_reject_changed_workload_shape(self):
        """Require deterministic CPU and the stated synchronous collection shape."""
        original = json.loads(bench_ppo.DEFAULT_CONFIG.read_text())
        mutations = {
            "task": "Acrobot-v1",
            "device": "cuda",
            "torch_threads": 2,
            "torch_interop_threads": 2,
            "training_num": 21,
            "steps_per_collect": 2561,
            "batch_size": 63,
            "num_threads": 0,
            "seed": -1,
            "lr": math.nan,
            "value_clip": 1,
            "gamma": 2,
            "unrecognized": "field",
        }
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            for key, value in mutations.items():
                with self.subTest(key=key):
                    config.write_text(json.dumps({**original, key: value}))
                    with self.assertRaises(ValueError):
                        bench_ppo.load_config(config, 1)
            config.write_text(json.dumps(original))
            for count in (0, -1, True):
                with self.assertRaises(ValueError):
                    bench_ppo.load_config(config, count)

    def test_fingerprints_preserve_exact_semantics(self):
        """Ignore mapping insertion order but catch typed one-ULP/shape drift."""
        digest = bench_ppo.fingerprint
        self.assertEqual(
            digest({"a": 1, "b": [2.0]}), digest({"b": [2.0], "a": 1})
        )
        for left, right in (
            (0.0, -0.0),
            (1.0, math.nextafter(1.0, math.inf)),
            (1, 1.0),
            (True, 1),
            ([1], (1,)),
            (["ab", "c"], ["a", "bc"]),
            ({1: 2}, {"1": 2}),
        ):
            with self.subTest(left=left, right=right):
                self.assertNotEqual(digest(left), digest(right))
        value = FakeArray(struct.pack("<d", 1.0))
        same = FakeArray(struct.pack("<d", 1.0))
        self.assertEqual(digest(value, FAKE_NUMPY), digest(same, FAKE_NUMPY))
        for changed in (
            FakeArray(struct.pack("<d", math.nextafter(1.0, math.inf))),
            FakeArray(value.data, dtype=">f8"),
            FakeArray(value.data, shape=(1, 1)),
        ):
            self.assertNotEqual(
                digest(value, FAKE_NUMPY), digest(changed, FAKE_NUMPY)
            )

    def test_fingerprints_reject_unsupported_or_nonfinite(self):
        """Never silently coerce unsupported objects or invalid numeric evidence."""
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                bench_ppo.fingerprint(value)
        for value in (object(), {1.0: "float key"}, {1, 2}):
            with self.assertRaises(TypeError):
                bench_ppo.fingerprint(value)
        with self.assertRaises(TypeError):
            bench_ppo.fingerprint(FakeArray(b"x", dtype="|O8"), FAKE_NUMPY)

    def test_result_output_is_exclusive_and_finite(self):
        """Preserve existing evidence and reject invalid output before opening it."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = {
                "label": "baseline",
                "timing": {"training_seconds": 1.25},
                "semantic_sha256": bench_ppo.fingerprint({"x": 1}),
            }
            bench_ppo.write_result(output, result)
            original = output.read_text()
            self.assertEqual(json.loads(original), result)
            with self.assertRaises(FileExistsError):
                bench_ppo.write_result(output, {"overwrite": True})
            self.assertEqual(output.read_text(), original)
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                bench_ppo.write_result(invalid, {"invalid": math.nan})
            self.assertFalse(invalid.exists())

    def test_native_identity_is_checked_before_import(self):
        """Reject a missing package, wrong binary digest or contaminated process."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            with self.assertRaises(ValueError):
                bench_ppo.load_package(package)
            (package / "__init__.py").write_text(
                "raise RuntimeError('imported!')"
            )
            native = package / bench_ppo.NATIVE_FILE
            native.parent.mkdir()
            native.write_bytes(b"test-only native placeholder")
            with self.assertRaisesRegex(ValueError, "does not match"):
                bench_ppo.load_package(package, "0" * 64)
            with mock.patch.dict(sys.modules, {"envpool": object()}):
                with self.assertRaisesRegex(RuntimeError, "already imported"):
                    bench_ppo.load_package(package)

    def test_help_and_import_need_no_training_dependencies(self):
        """Help works even with site-packages disabled; these tests stay light."""
        result = subprocess.run(
            [sys.executable, "-S", str(Path(bench_ppo.__file__)), "--help"],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("--iterations", result.stdout)
        self.assertIn("--warmup", result.stdout)
        for name in ("envpool", "torch", "numpy", "numba", "tianshou"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
