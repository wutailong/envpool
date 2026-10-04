# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Synthetic, dependency-free checks for complete-block profile attribution."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from phase_clock import PARENTS, PhaseClock
from run_profile_blocks import CONDITIONS, ORDERS
from summarize_profile_blocks import (
    BUDGET,
    ROOT,
    fingerprint,
    summarize,
    summarize_trial,
    validate_profile,
    validate_trial,
)


def fixture():
    """Construct only the mandatory 32 tiny aggregate records, never train."""
    config = json.loads((ROOT / "ppo/config.json").read_text())
    hashes = {
        "original": "a" * 64,
        "retained": "b" * 64,
        "plain": "c" * 64,
        "profile": "d" * 64,
    }
    conditions = {
        label: {
            "runtime": runtime,
            "mode": mode,
            "package": runtime,
            "script": mode,
            "native_sha256": hashes[runtime],
            "script_sha256": hashes[mode],
        }
        for label, (runtime, mode) in CONDITIONS.items()
    }
    plan = {
        "blocks": 4,
        "samples": 32,
        "orders": list(ORDERS),
        "conditions": conditions,
        "budget": BUDGET.copy(),
        "warmup_updates": 2,
    }
    complete = {
        "samples": 32,
        "semantic_parity": True,
        "frozen_native_and_script_hashes_match": True,
    }
    semantics = {
        "config": fingerprint(config),
        "initial": "e" * 64,
        "final": "f" * 64,
        "metrics": "0" * 64,
    }
    times = dict(
        zip("abcdefgh", (100, 80, 60, 70, 50, 40, 30, 35), strict=True)
    )
    rows = []
    for block, order in enumerate(ORDERS):
        for slot, label in enumerate(order):
            runtime, mode = CONDITIONS[label]
            wall = times[label]
            provenance = {
                "native_sha256": hashes[runtime],
                "script_sha256": hashes[mode],
                "native_module": "envpool.classic_control.classic_control_envpool",
                "package_path_verified": True,
                "native_path_verified": True,
                "expected_native_sha256_verified": True,
                "reference_config_sha256": "1" * 64,
                "python_version": "3.12.0",
                "versions": {
                    "tianshou": "0.5.1",
                    "envpool": "0.8.4",
                    "torch": "2.5.0",
                    "numpy": "1.26.4",
                    "gymnasium": "0.29.1",
                    "numba": "0.60.0",
                },
                "torch_threads": 1,
                "torch_interop_threads": 1,
                "torch_deterministic": True,
                "cpu_affinity": [1, 2],
                "thread_environment": dict.fromkeys(
                    (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                    ),
                    "1",
                ),
            }
            data = {
                "schema_version": 1,
                "label": "sample",
                "config": copy.deepcopy(config),
                "budget": {**BUDGET, "train_episodes": 1200},
                "measurement": {
                    "warmup_updates": 2,
                    "fresh_seeded_run_after_warmup": True,
                },
                "timing": {
                    "training_seconds": wall,
                    "process_cpu_seconds": wall * 1.1,
                    "env_steps_per_second": 256000 / wall,
                    "optimizer_steps_per_second": 8000 / wall,
                },
                "semantic_fingerprints": copy.deepcopy(semantics),
                "semantic_sha256": fingerprint(semantics),
                "provenance": provenance,
            }
            if mode == "profile":
                observer = PhaseClock()
                for name, percent in zip(
                    PARENTS, (1, 20, 70, 1, 10, 2), strict=True
                ):
                    observer._stats[name] = {
                        "calls": 12800
                        if name == "env_step"
                        else 1000
                        if name == "env_reset"
                        else 100,
                        "exceptions": 0,
                        "inclusive_ns": wall * percent * 10_000_000,
                        "direct_child_ns": wall * 12 * 10_000_000
                        if name == "collect"
                        else 0,
                    }
                data["phase_profile"] = observer.report(wall, 100, 12800)
                data["instrumentation"] = {
                    "schema_version": 1,
                    "clock": "time.perf_counter_ns",
                    "clock_resolution_seconds": 1e-9,
                    "semantic_hashes_include_phase_data": False,
                    "saves_weights_or_rollout_arrays": False,
                }
                provenance.update(
                    reference_script_sha256=hashes["plain"],
                    phase_helper_sha256="2" * 64,
                )
            rows.append({
                "block": block,
                "slot": slot,
                "variant": label,
                "runtime": runtime,
                "mode": mode,
                "order": list(order),
                "started_unix": 1000 + len(rows),
                "data": data,
            })
    return plan, rows, complete


class ProfileSummaryTest(unittest.TestCase):
    """Reject malformed trials and preserve effect directions and block units."""

    def setUp(self):
        """Build one small complete aggregate trial per test."""
        self.plan, self.rows, self.complete = fixture()

    def test_complete_balance_and_signs(self):
        """Faster rate is positive; negative observer and A/A effects survive."""
        result = summarize_trial(self.plan, self.rows, self.complete)
        contrasts = result["contrasts"]
        self.assertAlmostEqual(
            contrasts["native_plain_rate"]["block_geomean_effect_pct"], 100
        )
        self.assertGreater(
            contrasts["native_profile_rate_descriptive"][
                "block_geomean_effect_pct"
            ],
            0,
        )
        for runtime in ("original", "retained"):
            self.assertLess(
                contrasts[f"{runtime}_observer_time_overhead"][
                    "block_geomean_effect_pct"
                ],
                0,
            )
            profile = result["profiles"][runtime]
            self.assertEqual(profile["runs"], 8)
            self.assertEqual(
                profile["exact_counts_each_run"]["env_step"], 12800
            )
            self.assertAlmostEqual(
                profile["context"]["env_call_share"]["median"], 0.12
            )
            self.assertAlmostEqual(
                profile["context"]["conditional_2x_env_call_total_speedup"][
                    "median"
                ],
                1 / 0.94,
            )
            self.assertAlmostEqual(
                profile["context"][
                    "conditional_zero_cost_env_call_total_speedup"
                ]["median"],
                1 / 0.88,
            )
        self.assertAlmostEqual(
            contrasts["aa_b_over_a_rate"]["block_geomean_effect_pct"], 25
        )
        self.assertLess(
            contrasts["aa_d_over_c_rate"]["block_geomean_effect_pct"], 0
        )
        self.assertEqual(
            len(contrasts["native_plain_rate"]["block_log_contrasts"]), 4
        )
        self.assertEqual(len(result["raw_rows"]), 32)
        self.assertEqual(result["uncertainty"]["bootstrap_resamples"], 10000)

    def test_balance_schema_and_frozen_identity_guards(self):
        """No incomplete, shuffled, changed-budget or mismatched provenance trial."""
        mutations = [
            lambda p, r, c: r.pop(),
            lambda p, r, c: r.reverse(),
            lambda p, r, c: p["orders"].reverse(),
            lambda p, r, c: p["conditions"]["b"].update(native_sha256="9" * 64),
            lambda p, r, c: r[0].update(runtime="retained"),
            lambda p, r, c: r[0]["data"]["config"].update(seed=1),
            lambda p, r, c: r[0]["data"]["budget"].update(env_steps=1),
            lambda p, r, c: r[0]["data"]["provenance"].update(
                script_sha256="9" * 64
            ),
            lambda p, r, c: r[0]["data"]["provenance"].update(cpu_affinity=[2]),
            lambda p, r, c: r[0]["data"]["provenance"]["versions"].update(
                torch="other"
            ),
            lambda p, r, c: r[0]["data"]["provenance"].update(torch_threads=2),
            lambda p, r, c: r[0]["data"].update(semantic_sha256="9" * 64),
            lambda p, r, c: r[0]["data"]["timing"].update(
                training_seconds=math.nan
            ),
            lambda p, r, c: c.update(semantic_parity=False),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                plan, rows, complete = copy.deepcopy((
                    self.plan,
                    self.rows,
                    self.complete,
                ))
                mutate(plan, rows, complete)
                with self.assertRaises(ValueError):
                    validate_trial(plan, rows, complete)

    def test_phase_nesting_counts_and_partition_guards(self):
        """Inclusive collection cannot be double-counted or invalid resets hidden."""
        mutations = [
            lambda d: d["phase_profile"]["phases"]["env_step"].update(
                parent=None
            ),
            lambda d: d["phase_profile"]["phases"]["env_step"].update(
                calls=12799
            ),
            lambda d: d["phase_profile"]["phases"]["collect"].update(calls=99),
            lambda d: d["phase_profile"]["phases"]["env_reset"].update(
                calls=1001
            ),
            lambda d: d["phase_profile"]["phases"]["env_reset"].update(
                calls=1201
            ),
            lambda d: d["phase_profile"]["phases"]["env_step"].update(
                exceptions=1
            ),
            lambda d: d["phase_profile"]["phases"]["collect"].update(
                direct_child_ns=0
            ),
            lambda d: d["phase_profile"]["disjoint_partition_ns"].update(
                collect_excluding_env_calls_ns=12_000_000_000
            ),
            lambda d: d["phase_profile"]["validation"].update(
                wall_roundoff_clamped_ns=1
            ),
            lambda d: d["provenance"].update(reference_script_sha256="9" * 64),
            lambda d: d["provenance"].update(phase_helper_sha256="9" * 64),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                rows = copy.deepcopy(self.rows)
                mutate(
                    next(
                        row["data"] for row in rows if row["mode"] == "profile"
                    )
                )
                with self.assertRaises(ValueError):
                    validate_trial(self.plan, rows, self.complete)

    def test_raw_source_hash_and_missing_schema(self):
        """Hash exact raw bytes and fail malformed schemas before bootstrapping."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "plan.json").write_text(json.dumps(self.plan))
            (path / "complete.json").write_text(json.dumps(self.complete))
            raw = "".join(json.dumps(row) + "\n" for row in self.rows).encode()
            (path / "samples.jsonl").write_bytes(raw)
            result = summarize(path)
            self.assertEqual(
                result["source_sha256"]["samples.jsonl"],
                hashlib.sha256(raw).hexdigest(),
            )
            del self.rows[0]["data"]["config"]
            (path / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows)
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                summarize(path)

    def test_reported_roundoff_is_accounted_once(self):
        """Accept an explicit sub-microsecond wall clamp, never hide its sign."""
        data = next(
            row["data"] for row in self.rows if row["mode"] == "profile"
        )
        profile = data["phase_profile"]
        wall = profile["training_wall_ns"] - profile["raw_loop_other_ns"] - 500
        data["timing"]["training_seconds"] = wall / 1e9
        profile["training_wall_ns"] = wall
        profile["raw_loop_other_ns"] = -500
        profile["disjoint_partition_ns"]["loop_other_ns"] = 0
        profile["validation"]["wall_roundoff_clamped_ns"] = 500
        validate_profile(data)
        profile["validation"]["wall_roundoff_clamped_ns"] = 0
        with self.assertRaises(ValueError):
            validate_profile(data)


if __name__ == "__main__":
    unittest.main()
