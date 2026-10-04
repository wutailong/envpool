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
"""Validate layouts, adapters, and paired plans without native EnvPool runs."""

import collections
import contextlib
import copy
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_interleaved as bench
import numpy as np
import run_interleaved_blocks as runner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from summarize_paired_blocks import _validate


class FakePool:
    """Return grouped arrays with shuffled env order and changing player count."""

    _action_keys = (
        "players.action",
        "players.env_id",
        "list_action",
        "env_id",
        "players.id",
    )
    _state_keys = (
        "info:players.id",
        "info:env_id",
        "info:players.env_id",
        "obs:dyn",
    )

    def __init__(self, player_counts=(1, 2, 3, 4, 1)):
        """Use distinct player IDs so a wrongly paired permutation is visible."""
        self.states = []
        for step, players in enumerate(player_counts):
            env_ids = np.roll(np.arange(bench.NUM_ENVS, dtype=np.int32), step)
            player_env_ids = np.repeat(env_ids, players)
            player_ids = np.arange(len(player_env_ids), dtype=np.int32)
            dynamic = np.empty(len(player_env_ids), dtype=object)
            for index, env_id in enumerate(player_env_ids):
                dynamic[index] = np.zeros((int(env_id) + 1, 10))
            self.states.append((player_ids, env_ids, player_env_ids, dynamic))
        self.sent = []

    def _recv(self):
        return self.states.pop(0)

    def _send(self, action):
        self.sent.append(tuple(action))


class InterleavedBenchmarkTest(unittest.TestCase):
    """Guard the explicit stress workload and its inherited measurement plan."""

    def test_permutations_cover_one_to_four_players(self):
        """Keep order within each env and scatter p > 1 at N-row intervals."""
        num_envs = 5
        env_ids = np.array([4, 0, 3, 1, 2])
        permutations = bench.make_permutations(np, num_envs)
        self.assertEqual(set(permutations), {5, 10, 15, 20})
        for players in range(1, 5):
            permutation = permutations[num_envs * players]
            grouped = np.repeat(env_ids, players)
            np.testing.assert_array_equal(
                np.sort(permutation), np.arange(len(grouped))
            )
            np.testing.assert_array_equal(
                grouped[permutation], np.tile(env_ids, players)
            )
            for env_id in env_ids:
                locations = np.flatnonzero(grouped[permutation] == env_id)
                np.testing.assert_array_equal(
                    np.diff(locations), [num_envs] * (players - 1)
                )
                np.testing.assert_array_equal(
                    permutation[locations], np.flatnonzero(grouped == env_id)
                )
            if players == 1:
                np.testing.assert_array_equal(permutation, np.arange(num_envs))

    def test_all_player_fields_share_permutation_env_fields_stay_ordered(self):
        """Exercise warmup and timed paths without native or performance work."""
        env = FakePool()
        states = env.states.copy()
        list_action = np.zeros((bench.NUM_ENVS, 6))
        loop = bench.InterleavedDummyLoop(env, list_action, warmup=3)
        permutations = loop.permutations.copy()
        loop.validate(np, 4)
        for index, previous in enumerate(states[:-1]):
            self.assertEqual(loop.step(), bench.NUM_ENVS)
            action = dict(zip(env._action_keys, env.sent[-1], strict=True))
            permutation = permutations[len(previous[0])]
            self.assertIs(action["env_id"], previous[1])
            self.assertIs(action["list_action"], list_action)
            for name, source in (
                ("players.env_id", previous[2]),
                ("players.id", previous[0]),
                ("players.action", previous[0]),
            ):
                np.testing.assert_array_equal(action[name], source[permutation])
                self.assertFalse(np.shares_memory(action[name], source))
            if index == 2:
                loop.validate(np, 4)
                self.assertEqual(loop.step, loop._timed_step)
        self.assertEqual(loop.warmup_player_counts, {1, 2, 3})
        self.assertEqual(loop.scattered_warmup_steps, 2)
        self.assertEqual(loop.warmup_completed, 3)
        for count, permutation in permutations.items():
            self.assertIs(loop.permutations[count], permutation)
        loop.validate(np, 4)

    def test_reject_invalid_grouping_and_counts(self):
        """Reject partial, duplicate, unequal, ungrouped, or excessive rows."""
        env_ids = np.array([2, 0, 3, 1])
        valid = np.repeat(env_ids, 2)
        self.assertEqual(
            bench.validate_layout(np, env_ids, valid, np.arange(8), 4), 2
        )
        cases = (
            (env_ids[:-1], valid, np.arange(8)),
            (np.array([0, 0, 2, 3]), valid, np.arange(8)),
            (env_ids, valid, np.arange(7)),
            (env_ids, valid[:-1], np.arange(7)),
            (env_ids, np.array([2, 0, 0, 0, 3, 3, 1, 1]), np.arange(8)),
            (env_ids, np.tile(env_ids, 2), np.arange(8)),
            (env_ids, np.array([], dtype=int), np.array([], dtype=int)),
            (env_ids, np.repeat(env_ids, 5), np.arange(20)),
            (env_ids.reshape(2, 2), valid, np.arange(8)),
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                bench.validate_layout(np, *values, num_envs=4)

    def test_warmup_rejects_bad_layout_before_send(self):
        """Do not silently fall back to the grouped workload or mixed counts."""
        env = FakePool((2, 2))
        state = list(env.states[0])
        state[2] = np.tile(state[1], 2)
        env.states[0] = tuple(state)
        loop = bench.InterleavedDummyLoop(env, object(), warmup=2)
        with self.assertRaisesRegex(ValueError, "grouped"):
            loop.step()
        self.assertEqual(env.sent, [])

    def test_warmup_must_send_scattered_rows(self):
        """Reject a run whose warmup never actually submits a p > 1 action."""
        env = FakePool((1, 1, 1))
        loop = bench.InterleavedDummyLoop(env, object(), warmup=2)
        loop.step()
        loop.step()
        with self.assertRaisesRegex(ValueError, "never sent interleaved"):
            loop.validate(np, 4)

    def test_synchronized_spec_is_actually_passed(self):
        """Override env_seed at native spec construction, retaining other args."""
        calls = []

        class FakeSpec:
            _config_keys = ("seed", "env_seed", "num_envs", "max_num_players")
            _default_config_values = (42, [], 64, 4)

            def __init__(self, values):
                calls.append(values)

        native = SimpleNamespace(
            __file__="/runtime/envpool/dummy/native.so",
            _DummyEnvSpec=FakeSpec,
            _DummyEnvPool=object(),
        )
        with patch.object(
            bench.importlib, "import_module", return_value=native
        ):
            adapted = bench.synchronized_module("envpool.dummy.dummy_envpool")
        adapted._DummyEnvSpec((42, [99] * 64, 64, 4))
        self.assertEqual(calls, [(42, [42] * 64, 64, 4)])
        self.assertIs(adapted._DummyEnvPool, native._DummyEnvPool)
        self.assertEqual(adapted.__file__, native.__file__)
        self.assertIs(adapted._DummyEnvSpec._config_keys, FakeSpec._config_keys)
        self.assertEqual(FakeSpec._default_config_values, (42, [], 64, 4))

    def test_main_preserves_measured_row_and_restores_hooks(self):
        """Exercise the executable adapter using a fake inherited measurement."""
        original_loop = bench.inherited.DummyLoop
        original_importlib = bench.inherited.importlib
        measured = {
            "env": "DummyPlayers4",
            "variant": "original",
            "binary_sha256": "a" * 64,
            "binary_path": "/native/dummy.so",
            "runtime_root": "/control",
            "seconds": 3.01,
            "cpu_seconds": 4.25,
            "environment_responses": 8192,
            "scheduler_wait_seconds": 0.1,
            "seed": 42,
            "max_num_players": 4,
        }

        def fake_main():
            loop = bench.inherited.DummyLoop(FakePool((1, 2, 3, 1)), object())
            loop.validate(np, 4)
            for _ in range(3):
                loop.step()
            loop.validate(np, 4)
            bench.inherited.print(json.dumps(measured), flush=True)

        captured = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                ["bench", "--max-num-players", "4", "--warmup", "3"],
            ),
            patch.object(bench.inherited, "main", fake_main),
            contextlib.redirect_stdout(captured),
        ):
            bench.main()
        expected = measured | {
            "env": bench.CASE_NAME,
            "env_seed": [42] * 64,
            "player_order": bench.PLAYER_ORDER,
            "warmup_player_counts": [1, 2, 3],
            "scattered_warmup_steps": 2,
        }
        self.assertEqual(json.loads(captured.getvalue()), expected)
        self.assertIs(bench.inherited.DummyLoop, original_loop)
        self.assertIs(bench.inherited.importlib, original_importlib)
        self.assertFalse(hasattr(bench.inherited, "print"))

    def test_cli_rejects_wrong_case_and_short_warmup(self):
        """Reject invalid stress inputs before native runtime loading."""
        for args in (("--max-num-players", "1"), ("--warmup", "1")):
            with (
                self.subTest(args=args),
                patch.object(sys, "argv", ["bench", *args]),
                patch.object(bench.inherited, "main") as main,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                bench.main()
            main.assert_not_called()

    def test_runner_only_adapts_declared_hooks(self):
        """Leave inherited subprocess, file creation, and hashing code intact."""
        original = (
            runner.inherited.make_plan,
            runner.inherited.commands,
            runner.inherited.validate_sample,
        )

        def check_hooks():
            self.assertIs(runner.inherited.make_plan, runner.make_plan)
            self.assertIs(runner.inherited.commands, runner.commands)
            self.assertIs(
                runner.inherited.validate_sample, runner.validate_sample
            )

        with patch.object(
            runner.inherited, "main", side_effect=check_hooks
        ) as main:
            runner.main()
        main.assert_called_once_with()
        self.assertEqual(
            original,
            (
                runner.inherited.make_plan,
                runner.inherited.commands,
                runner.inherited.validate_sample,
            ),
        )

    def test_balanced_plan_and_summary_schema(self):
        """Verify all 32 slots, same-root pairs, metadata, and summary inputs."""
        roots = {
            label: "/control" if label in "ab" else "/candidate"
            for label in "abcd"
        }
        plan = runner.make_plan(roots, "/venv/bin/python")
        records = list(runner.commands(plan))
        self.assertEqual(len(records), 32)
        self.assertEqual(records, list(runner.commands(plan)))
        self.assertEqual(plan["blocks"], 8)
        self.assertEqual(plan["case_count"], 1)
        self.assertEqual(plan["cases"][0]["env_seed"], [42] * 64)
        self.assertEqual(plan["cases"][0]["player_order"], bench.PLAYER_ORDER)
        self.assertEqual(plan["orders"], ["acdb", "cabd", "bdca", "dbac"])
        counts = collections.Counter()
        rows = []
        for item in records:
            label, command = item["label"], item["command"]
            self.assertEqual(Path(command[1]).name, "bench_interleaved.py")
            self.assertEqual(command[0], "/venv/bin/python")
            self.assertEqual(
                command[command.index("--envpool-root") + 1], roots[label]
            )
            self.assertEqual(
                command[command.index("--max-num-players") + 1], "4"
            )
            counts[label, item["slot"]] += 1
            row = {
                "env": bench.CASE_NAME,
                "num_envs": 64,
                "batch_size": 64,
                "threads": 4,
                "max_num_players": 4,
                "pin": False,
                "async": False,
                "block": item["block"],
                "rep": item["block"],
                "slot": item["slot"],
                "variant": "original" if label in "ab" else "candidate",
                "order": item["order"],
                "expected_samples": 32,
                "warmup": 400,
                "requested_seconds": 3.0,
                "seed": 42,
                "env_seed": [42] * 64,
                "player_order": bench.PLAYER_ORDER,
                "warmup_player_counts": [1, 2, 3],
                "scattered_warmup_steps": 260,
                "affinity": [0, 1, 2, 3],
                "worker_affinity_offset": -1,
                "runtime_selection": "explicit-root",
                "runtime_root": roots[label],
                "binary_sha256": ("a" if label in "ab" else "b") * 64,
                "environment_responses": 8192,
                "env_steps": 8192,
                "vector_calls": 128,
            }
            runner.validate_sample(row, item, plan)
            row["variant"] = label
            rows.append(row)
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {2})
        evidence = _validate(rows, plan, "binary_sha256")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            next(iter(evidence.values()))["same_binary_pairs"]["status"],
            "verified",
        )
        for key, value in (
            ("player_order", "grouped"),
            ("env_seed", []),
            ("scattered_warmup_steps", 0),
            ("warmup_player_counts", [1]),
            ("variant", "candidate"),
        ):
            bad = copy.deepcopy(rows[0])
            bad["variant"] = "original"
            bad[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "interleaving"),
            ):
                runner.validate_sample(bad, records[0], plan)


if __name__ == "__main__":
    unittest.main()
