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
"""Reporting regressions using deterministic synthetic records and arrays only."""

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


class ToolTests(unittest.TestCase):
    """Exercise reporting, CLI construction, and exact archive comparisons."""

    def setUp(self):
        """Generate controlled samples rather than depend on archived runs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.samples = Path(self.temporary.name) / "samples.jsonl"
        self.records = []
        for variant, scale in [
            ("original", 100),
            ("rebuilt", 80),
            ("candidate", 120),
        ]:
            for rep in range(3):
                self.records.append({
                    "env": "Synthetic-v0",
                    "num_envs": 8,
                    "batch_size": 4,
                    "threads": 2,
                    "pin": True,
                    "variant": variant,
                    "rep": rep,
                    "env_steps_per_s": scale * (rep + 1),
                    "us_per_call": 2 * (rep + 1),
                    "cpu_seconds": 2 * (rep + 1),
                    "seconds": 2,
                })
        self.write_samples()

    def write_samples(self):
        """Write only this test's temporary synthetic records."""
        self.samples.write_text(
            "".join(json.dumps(row) + "\n" for row in self.records)
        )

    def test_summary(self):
        """Check every numeric field, grouping and formatted dispersion."""
        rows = summarize([self.samples])
        expected = {"config": ["Synthetic-v0", 8, 4, 2, True], "variants": {}}
        for variant, scale in [
            ("original", 100),
            ("rebuilt", 80),
            ("candidate", 120),
        ]:
            expected["variants"][variant] = {
                "median": scale * 2,
                "min": scale,
                "max": scale * 3,
                "cv": 0.5,
                "samples": 3,
                "median_us_per_call": 4,
                "median_cpu_seconds_per_wall_second": 2,
            }
        expected["vs_original_pct"] = (240 / 200 - 1) * 100
        expected["vs_rebuilt_pct"] = 50.0
        self.assertEqual(rows, [expected])
        self.assertAlmostEqual(rows[0]["vs_original_pct"], 20)
        self.assertEqual(
            dispersion(rows),
            "Synthetic-v0 N=8, batch=4, threads=2, affinity=pinned\n"
            "  original  median=200; range=100..300; CV=50.0%; n=3; vector-call latency=4.00 us\n"
            "  rebuilt   median=160; range=80..240; CV=50.0%; n=3; vector-call latency=4.00 us\n"
            "  candidate median=240; range=120..360; CV=50.0%; n=3; vector-call latency=4.00 us\n\n",
        )

    def test_single_sample_and_separate_configuration(self):
        """Keep configurations separate and handle absent rebuilt controls."""
        for variant, rate in [("original", 50), ("candidate", 40)]:
            self.records.append(
                dict(
                    self.records[0],
                    variant=variant,
                    env_steps_per_s=rate,
                    pin=False,
                )
            )
        self.write_samples()
        rows = summarize([self.samples])
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[1]["config"][-1])
        self.assertIsNone(rows[1]["vs_rebuilt_pct"])
        self.assertAlmostEqual(rows[1]["vs_original_pct"], -20)
        for value in rows[1]["variants"].values():
            self.assertEqual(value["cv"], 0)
            self.assertEqual(value["samples"], 1)
            self.assertEqual(value["median"], value["min"])
            self.assertEqual(value["median"], value["max"])
        self.assertIn("affinity=default", dispersion(rows))

    def test_duplicate_samples(self):
        """Reject accidental re-pooling of the same experiment."""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([self.samples, self.samples])

    def test_invalid_variant(self):
        """Unknown labels must not silently enter a comparison."""
        self.records[0]["variant"] = "unexpected"
        self.write_samples()
        with self.assertRaisesRegex(ValueError, "unexpected variant"):
            summarize([self.samples])

    def test_missing_control(self):
        """Each configuration needs both the original and candidate."""
        self.records = [
            row for row in self.records if row["variant"] != "original"
        ]
        self.write_samples()
        with self.assertRaisesRegex(
            ValueError, "requires original and candidate"
        ):
            summarize([self.samples])

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
