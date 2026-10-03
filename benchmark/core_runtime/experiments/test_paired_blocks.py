"""Test paired-trial analysis using synthetic rows, without native runtimes."""

import copy
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from summarize_blocks import (
    summarize,
    summarize_log_differences,
    summarize_rows,
)
from summarize_paired_blocks import PATTERNS, summarize_paired

HERE = Path(__file__).resolve().parent


class PairedBlockTests(unittest.TestCase):
    """Check the block estimand, provenance limits, and strict plan coverage."""

    def setUp(self):
        """Create a complete synthetic paired experiment."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.path = root / "raw.jsonl"
        self.plan_path = root / "plan.json"
        self.plan = {
            "blocks": 4,
            "seconds": 3,
            "warmup": 400,
            "affinity": "default (all allowed CPUs)",
            "case_count": 2,
            "expected_samples": 32,
            "cases": [
                {"env": "CartPole-v1", "n": n, "batch": n, "threads": 1}
                for n in (20, 64)
            ],
        }
        self.ratios = (1.1, 0.9, 1.2, 0.8)
        self.rows = []
        for block, (base, ratio) in enumerate(
            zip((100, 1000, 10, 10000), self.ratios, strict=True)
        ):
            rates = {
                "a": base / 2,
                "b": base * 2,
                "c": base * ratio / 3,
                "d": base * ratio * 3,
            }
            for case in self.plan["cases"]:
                for slot, label in enumerate(PATTERNS[block]):
                    self.rows.append({
                        "env": case["env"],
                        "num_envs": case["n"],
                        "batch_size": case["batch"],
                        "threads": case["threads"],
                        "pin": False,
                        "variant": label,
                        "block": block,
                        "rep": block,
                        "slot": slot,
                        "order": list(PATTERNS[block]),
                        "expected_samples": self.plan["expected_samples"],
                        "env_steps_per_s": rates[label],
                        "cpu_seconds": 1,
                        "env_steps": 1000,
                        "scheduler_wait_seconds": 0.01,
                        "voluntary_context_switches": 20,
                        "scheduler_thread_set_stable": True,
                        "warmup": 400,
                        "requested_seconds": 3,
                        "seed": 42,
                        "async": False,
                        "affinity": [0, 1],
                        "worker_affinity_offset": -1,
                        "runtime_selection": "explicit-root",
                    })
        self.write_inputs()

    def write_inputs(self):
        """Write the current synthetic rows and plan."""
        self.path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        self.plan_path.write_text(json.dumps(self.plan))

    def analyze(self):
        """Persist modified fixtures and summarize them."""
        self.write_inputs()
        return summarize_paired(self.path, self.plan_path)

    def test_equal_weight_block_estimand_and_aa(self):
        """Geometric replicate means and block weights must not become pooled rates."""
        result = self.analyze()
        expected = 100 * math.expm1(statistics.mean(map(math.log, self.ratios)))
        self.assertEqual(len(result["configurations"]), 2)
        self.assertEqual(result["plan"], self.plan)
        self.assertEqual(result["samples"], 32)
        for config in result["configurations"]:
            self.assertEqual(set(config["metrics"]), set("abcd"))
            primary = config["primary"]
            self.assertEqual(primary["block_ids"], list(range(4)))
            self.assertAlmostEqual(
                primary["block_geomean_effect_pct"], expected
            )
            for actual, ratio in zip(
                primary["block_effects_pct"], self.ratios, strict=True
            ):
                self.assertAlmostEqual(actual, 100 * (ratio - 1))
            for actual, ratio in zip(
                primary["block_log_effects"], self.ratios, strict=True
            ):
                self.assertAlmostEqual(actual, math.log(ratio))
            low, high = primary["exploratory_block_bootstrap_95pct"]
            self.assertLessEqual(low, expected)
            self.assertGreaterEqual(high, expected)
            for name, effect in (("b_over_a", 300), ("d_over_c", 800)):
                aa = config["aa_diagnostics"][name]
                self.assertAlmostEqual(aa["block_geomean_effect_pct"], effect)
                for actual in aa["block_effects_pct"]:
                    self.assertAlmostEqual(actual, effect)
            self.assertEqual(
                config["same_binary_pairs"]["status"], "unverified"
            )

    def test_deterministic_and_raw_input_preserved(self):
        """Keep statistics reproducible and source bytes unchanged."""
        before = self.path.read_bytes()
        result = summarize_paired(self.path, self.plan_path)
        self.assertEqual(result, summarize_paired(self.path, self.plan_path))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(
            result["source_sha256"], hashlib.sha256(before).hexdigest()
        )

    def test_shared_summary_compatibility(self):
        """The original path API and new in-memory API retain the same results."""
        self.assertEqual(
            summarize(self.path, "a"), summarize_rows(self.rows, "a")
        )
        effect = summarize_log_differences(
            [math.log(1.25)] * 4, random.Random(738)
        )
        self.assertAlmostEqual(effect["block_geomean_effect_pct"], 25)
        for endpoint in effect["exploratory_block_bootstrap_95pct"]:
            self.assertAlmostEqual(endpoint, 25)

    def test_measured_hashes_validate_both_pairs(self):
        """Verify both same-binary pairs when hashes are present."""
        for row in self.rows:
            row["binary_sha256"] = ("A" if row["variant"] in "ab" else "b") * 64
        result = self.analyze()
        for config in result["configurations"]:
            identity = config["same_binary_pairs"]
            self.assertEqual(identity["status"], "verified")
            self.assertEqual(
                identity["by_label"]["a"], identity["by_label"]["b"]
            )
            self.assertEqual(
                identity["by_label"]["c"], identity["by_label"]["d"]
            )
            self.assertEqual(
                identity["by_label"]["a"]["binary_sha256"], "a" * 64
            )

    def test_explicit_hash_mapping(self):
        """Accept an explicitly named measured hash mapping."""
        for row in self.rows:
            row["loaded_module_hashes"] = {"runtime.so": "c" * 64}
        self.write_inputs()
        result = summarize_paired(
            self.path, self.plan_path, "loaded_module_hashes"
        )
        self.assertEqual(
            result["configurations"][0]["same_binary_pairs"]["status"],
            "verified",
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            summarize_paired(self.path, self.plan_path, "missing_hash")

    def test_hash_mismatch_drift_missing_or_invalid_rejected(self):
        """Reject inconsistent or malformed binary provenance."""
        for row in self.rows:
            row["binary_sha256"] = ("a" if row["variant"] in "ab" else "b") * 64
        original = copy.deepcopy(self.rows)
        for failure in ("pair", "drift", "missing", "invalid"):
            with self.subTest(failure=failure):
                self.rows = copy.deepcopy(original)
                if failure == "pair":
                    for row in self.rows:
                        if row["variant"] == "b":
                            row["binary_sha256"] = "c" * 64
                elif failure == "drift":
                    self.rows[0]["binary_sha256"] = "c" * 64
                elif failure == "missing":
                    del self.rows[0]["binary_sha256"]
                else:
                    self.rows[0]["binary_sha256"] = "not-a-digest"
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_incomplete_duplicate_and_subset_rejected(self):
        """Reject missing slots, duplicate rows, and omitted cases."""
        original = copy.deepcopy(self.rows)
        for failure in ("empty", "missing_slot", "duplicate", "missing_case"):
            with self.subTest(failure=failure):
                self.rows = copy.deepcopy(original)
                if failure == "empty":
                    self.rows = []
                elif failure == "missing_slot":
                    self.rows.pop()
                elif failure == "duplicate":
                    self.rows[-1] = copy.deepcopy(self.rows[0])
                else:
                    self.rows = [
                        row for row in self.rows if row["num_envs"] == 20
                    ]
                    for row in self.rows:
                        row["expected_samples"] = len(self.rows)
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_mismatched_rows_rejected(self):
        """Reject departures from the planned measurement settings."""
        original = copy.deepcopy(self.rows)
        changes = {
            "variant": "e",
            "order": list("abdc"),
            "block": 4,
            "slot": 4,
            "rep": 1,
            "expected_samples": 31,
            "num_envs": 21,
            "pin": "default",
            "warmup": 200,
            "requested_seconds": 2,
            "seed": 43,
            "async": True,
            "affinity": [1],
            "runtime_selection": "interpreter",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                self.rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_invalid_response_and_missing_fields_rejected(self):
        """Reject invalid rates and incomplete evidence fields."""
        original = copy.deepcopy(self.rows)
        for value in (0, -1, math.inf, math.nan):
            with self.subTest(rate=value):
                self.rows = copy.deepcopy(original)
                self.rows[0]["env_steps_per_s"] = value
                with self.assertRaisesRegex(ValueError, "response rate"):
                    self.analyze()
        self.rows = copy.deepcopy(original)
        del self.rows[0]["warmup"]
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.analyze()

    def test_invalid_plan_rejected(self):
        """Reject internally inconsistent experiment plans."""
        original = copy.deepcopy(self.plan)
        for field, value in (
            ("blocks", 3),
            ("case_count", 1),
            ("expected_samples", 31),
            ("affinity", "unknown"),
        ):
            with self.subTest(field=field):
                self.plan = copy.deepcopy(original)
                self.plan[field] = value
                with self.assertRaises(ValueError):
                    self.analyze()
        self.plan = copy.deepcopy(original)
        self.plan["cases"][1] = copy.deepcopy(self.plan["cases"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.analyze()

    def test_cli_writes_fresh_json_and_refuses_overwrite(self):
        """Write a fresh summary without replacing prior evidence."""
        output = self.path.with_name("summary.json")
        command = [
            sys.executable,
            str(HERE / "summarize_paired_blocks.py"),
            str(self.path),
            "--plan",
            str(self.plan_path),
            "--json-out",
            str(output),
        ]
        before = self.path.read_bytes()
        subprocess.run(command, check=True, capture_output=True, text=True)
        saved = output.read_bytes()
        self.assertEqual(len(json.loads(saved)["configurations"]), 2)
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("new file", second.stderr)
        self.assertEqual(output.read_bytes(), saved)
        command[-1] = str(self.path)
        self.assertNotEqual(
            subprocess.run(command, capture_output=True).returncode, 0
        )
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
