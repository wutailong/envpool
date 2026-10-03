"""Lightweight plan and evidence validation, without loading native runtimes."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from summarize_blocks import summarize

HERE = Path(__file__).resolve().parent


class BlockTests(unittest.TestCase):
    """Validate planning without creating an environment."""

    def test_plan_paths_and_order(self):
        """Keep virtualenv paths literal and both treatment positions balanced."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "runtime with spaces" / "envpool"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            python = root / "virtual python"
            python.symlink_to(sys.executable)
            out = root / "unused.jsonl"
            command = [
                sys.executable,
                str(HERE / "run_blocks.py"),
                "--python",
                str(python),
                "--variant",
                f"a={package.parent}",
                "--variant",
                f"b={package.parent}",
                "--out",
                str(out),
                "--cases",
                str(HERE / "cases.json"),
                "--blocks",
                "2",
                "--affinities",
                "default",
                "--dry-run",
            ]
            result = subprocess.run(
                command, check=True, text=True, capture_output=True
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 2 * 8 * 4)
            for start in range(0, len(rows), 4):
                group = rows[start : start + 4]
                self.assertEqual(
                    "".join(x["label"] for x in group),
                    ["abba", "baab"][group[0]["block"]],
                )
            self.assertTrue(
                all(row["command"][0] == str(python) for row in rows)
            )
            self.assertFalse(out.exists())

    def test_four_variant_williams_order(self):
        """Balance all four treatment positions without importing native code."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "runtime" / "envpool"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            command = [
                sys.executable,
                str(HERE / "run_blocks.py"),
                "--python",
                sys.executable,
                "--out",
                str(root / "unused"),
                "--cases",
                str(HERE / "shutdown-cases.json"),
                "--blocks",
                "4",
                "--affinities",
                "default",
                "--dry-run",
            ]
            for label in "abcd":
                command += ["--variant", f"{label}={package.parent}"]
            result = subprocess.run(
                command, check=True, text=True, capture_output=True
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 4 * 4 * 4)
            patterns = ["abdc", "bcad", "cdba", "dacb"]
            for start in range(0, len(rows), 4):
                group = rows[start : start + 4]
                self.assertEqual(
                    "".join(row["label"] for row in group),
                    patterns[group[0]["block"]],
                )

    def test_paired_replicates_order(self):
        """Each treatment is ABBA/BAAB; each same-binary label visits every slot."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["control", "treatment"]:
                package = root / name / "envpool"
                package.mkdir(parents=True)
                (package / "__init__.py").touch()
            command = [
                sys.executable,
                str(HERE / "run_blocks.py"),
                "--python",
                sys.executable,
                "--out",
                str(root / "unused"),
                "--cases",
                str(HERE / "shutdown-cases.json"),
                "--blocks",
                "4",
                "--affinities",
                "default",
                "--dry-run",
                "--paired-replicates",
            ]
            for label in "abcd":
                name = "control" if label in "ab" else "treatment"
                command += ["--variant", f"{label}={root / name}"]
            result = subprocess.run(
                command, check=True, text=True, capture_output=True
            )
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 4 * 4 * 4)
            for start in range(0, len(rows), 4):
                group = rows[start : start + 4]
                self.assertEqual(
                    "".join(row["label"] for row in group),
                    ["acdb", "cabd", "bdca", "dbac"][group[0]["block"]],
                )
                self.assertEqual(
                    "".join(
                        "A" if row["label"] in "ab" else "B" for row in group
                    ),
                    ["ABBA", "BAAB"][group[0]["block"] % 2],
                )
            self.assertFalse((root / "unused").exists())

    def test_incomplete_rejected(self):
        """Never summarize an empty experiment as success."""
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "empty.jsonl"
            p.write_text("")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize(p, "a")


if __name__ == "__main__":
    unittest.main()
