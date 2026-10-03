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
"""Smoke tests using existing timing records and synthetic arrays only."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from check_rollouts import arrays_equal, compare
from summarize import dispersion, summarize

ROOT = Path(__file__).resolve().parent
INPUTS = [
    ROOT / "results" / name
    for name in [
        "final-pinned-classic.jsonl",
        "final-pinned-heavy.jsonl",
        "final-default-confirm.jsonl",
    ]
]


class ToolTests(unittest.TestCase):
    """Exercise reporting, CLI construction, and exact archive comparisons."""

    def test_summary(self):
        """Reproduce all saved sample counts, medians, and ranges exactly."""
        rows = summarize(INPUTS)
        self.assertEqual(
            rows, json.loads((ROOT / "results/summary.json").read_text())
        )
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            sum(v["samples"] for r in rows for v in r["variants"].values()), 234
        )
        self.assertEqual(
            dispersion(rows).strip(),
            (ROOT / "results/dispersion.txt").read_text().strip(),
        )

    def test_duplicate_samples(self):
        """Reject accidental re-pooling of the same experiment."""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([INPUTS[0], INPUTS[0]])

    def test_help(self):
        """Every runnable script explains itself without importing EnvPool."""
        for script in [
            "bench_core.py",
            "run_matrix.py",
            "check_rollouts.py",
            "summarize.py",
        ]:
            result = subprocess.run(
                [sys.executable, str(ROOT / script), "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("usage:", result.stdout)

    def test_interleaving(self):
        """Construct both repetition orders without launching benchmark workers."""
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_matrix.py"),
                "--python",
                sys.executable,
                "--out",
                "unused.jsonl",
                "--reps",
                "2",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        commands = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 48)
        labels = [
            command[command.index("--variant") + 1] for command in commands
        ]
        self.assertEqual(labels[:3], ["original", "rebuilt", "candidate"])
        self.assertEqual(labels[24:27], ["candidate", "rebuilt", "original"])

    def test_exact_bytes(self):
        """Distinguish signed zeros and dtype while accepting identical NaNs."""
        self.assertFalse(arrays_equal(np.array([0.0]), np.array([-0.0])))
        self.assertFalse(
            arrays_equal(np.array([1], np.int32), np.array([1], np.int64))
        )
        value = np.array([np.nan, 1.0])
        self.assertTrue(arrays_equal(value, value.copy()))

    def test_archive_comparator(self):
        """A one-ULP mutation fails the same comparator used for real rollouts."""
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.npz"
            candidate = Path(directory) / "candidate.npz"
            values = np.array([0.0, 1.0], np.float64)
            np.savez(reference, observation=values)
            np.savez(candidate, observation=values.copy())
            with contextlib.redirect_stdout(io.StringIO()):
                compare(reference, [candidate])
            values[1] = np.nextafter(values[1], np.inf)
            np.savez(candidate, observation=values)
            with self.assertRaisesRegex(AssertionError, "bitwise mismatches"):
                compare(reference, [candidate])


if __name__ == "__main__":
    unittest.main()
