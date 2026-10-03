# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Synthetic checks for the pooled-control Williams comparison."""

import copy
import unittest

from summarize_comparison import summarize


class ComparisonTests(unittest.TestCase):
    """Check effects and reject incomplete or relabeled trials."""

    def rows(self):
        """Build four complete synthetic Williams blocks."""
        labels = ["a", "b", "direct", "fast"]
        patterns = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
        rows = []
        for block, pattern in enumerate(patterns):
            order = [labels[i] for i in pattern]
            for slot, label in enumerate(order):
                rows.append({
                    "env": "CartPole-v1",
                    "num_envs": 20,
                    "batch_size": 20,
                    "threads": 1,
                    "pin": False,
                    "block": block,
                    "slot": slot,
                    "variant": label,
                    "order": order,
                    "expected_samples": 16,
                    "env_steps_per_s": 80.0 if label == "direct" else 100.0,
                    "cpu_seconds": 1.0,
                    "env_steps": 100,
                    "scheduler_wait_seconds": 0.0,
                    "voluntary_context_switches": 0,
                    "scheduler_thread_set_stable": True,
                })
        return rows

    def test_exact_effects(self):
        """Pool two controls and preserve the direct fast-versus-fill contrast."""
        output = summarize(self.rows())
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["blocks"], 4)
        for name, expected in {
            "direct_vs_control": -20,
            "fast_vs_control": 0,
            "fast_vs_direct": 25,
            "b_over_a": 0,
        }.items():
            result = output[0]["contrasts"][name]
            self.assertAlmostEqual(result["block_geomean_effect_pct"], expected)
            for value in (
                result["block_effects_pct"]
                + result["exploratory_block_bootstrap_95pct"]
            ):
                self.assertAlmostEqual(value, expected)

    def test_bad_evidence_rejected(self):
        """Reject missing, duplicate, or relabeled samples."""
        rows = self.rows()
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
        altered = copy.deepcopy(rows)
        altered[-1] = copy.deepcopy(altered[0])
        with self.assertRaises(ValueError):
            summarize(altered)
        altered = copy.deepcopy(rows)
        altered[0]["order"] = ["a", "b", "direct", "fast"]
        with self.assertRaises(ValueError):
            summarize(altered)


if __name__ == "__main__":
    unittest.main()
