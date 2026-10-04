# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Light plan/schema guards; these tests import no training dependencies."""

import copy
import unittest

from run_profile_blocks import (
    CONDITIONS,
    ORDERS,
    validate_orders,
    validate_result,
)


class ProfileBlocksTest(unittest.TestCase):
    """Check the fixed factorial schedule and rejected schema changes."""

    def setUp(self):
        """Prepare a dependency-free valid result record."""
        self.condition = {"native_sha256": "a" * 64, "mode": "plain"}
        self.data = {
            "config": {
                "training_num": 20,
                "num_threads": 2,
                "updates": 100,
                "steps_per_collect": 2560,
                "repeat_per_collect": 2,
                "batch_size": 64,
                "torch_threads": 1,
                "torch_interop_threads": 1,
            },
            "budget": {
                "updates": 100,
                "env_steps": 256000,
                "optimizer_steps": 8000,
            },
            "provenance": {
                "native_sha256": "a" * 64,
                "expected_native_sha256_verified": True,
                "script_sha256": "b" * 64,
            },
            "measurement": {"warmup_updates": 2},
        }

    def test_balanced_orders_and_four_duplicated_conditions(self):
        """Each native/mode cell has genuine paired replicas."""
        validate_orders()
        self.assertEqual(sum(map(len, ORDERS)), 32)
        self.assertEqual(len(set(CONDITIONS.values())), 4)
        for pair in ("ab", "cd", "ef", "gh"):
            self.assertEqual(CONDITIONS[pair[0]], CONDITIONS[pair[1]])

    def test_modes_are_explicit(self):
        """Do not accidentally pool plain and profiled observations."""
        validate_result(self.data, self.condition, "b" * 64)
        observed = copy.deepcopy(self.data)
        observed["phase_profile"] = {}
        with self.assertRaises(ValueError):
            validate_result(observed, self.condition, "b" * 64)
        self.condition["mode"] = "profile"
        validate_result(observed, self.condition, "b" * 64)
        with self.assertRaises(ValueError):
            validate_result(self.data, self.condition, "b" * 64)

    def test_hash_and_budget_mismatches_fail(self):
        """Fail closed when work or runtime identity differs."""
        for group, key, bad in (
            ("config", "num_threads", 1),
            ("budget", "env_steps", 255999),
            ("provenance", "native_sha256", "c" * 64),
            ("provenance", "expected_native_sha256_verified", False),
            ("provenance", "script_sha256", "c" * 64),
            ("measurement", "warmup_updates", 3),
        ):
            with self.subTest(group=group, key=key):
                data = copy.deepcopy(self.data)
                data[group][key] = bad
                with self.assertRaises(ValueError):
                    validate_result(data, self.condition, "b" * 64)


if __name__ == "__main__":
    unittest.main()
