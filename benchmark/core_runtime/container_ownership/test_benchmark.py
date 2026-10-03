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
"""Check benchmark routing and paired schema without running native workloads."""

import collections
import sys
import unittest
from pathlib import Path

from bench_container import DummyLoop
from run_container_blocks import commands, make_plan, validate_sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from summarize_paired_blocks import _validate


class FakePool:
    """Expose deliberately reordered keys and changing multiplayer mappings."""

    _action_keys = (
        "players.action",
        "players.env_id",
        "list_action",
        "env_id",
        "players.id",
    )
    _state_keys = ("info:players.id", "info:env_id", "info:players.env_id")

    def __init__(self):
        """Keep consecutive responses with different environment/player order."""
        self.states = [
            ([0, 1, 0], [9, 3], [9, 9, 3]),
            ([0, 0, 1, 2], [3, 9], [3, 9, 9, 9]),
            ([0, 0], [9, 3], [9, 3]),
        ]
        self.sent = []

    def _recv(self):
        return self.states.pop(0)

    def _send(self, action):
        self.sent.append(tuple(action))


class ContainerBenchmarkTest(unittest.TestCase):
    """Guard action identity, per-environment units, balance, and summary input."""

    def test_action_fields_follow_returned_ids(self):
        """Use schema names, retain the preallocation, and count envs not players."""
        env = FakePool()
        list_action = object()
        first, second = env.states[:2]
        loop = DummyLoop(env, list_action)
        for previous in (first, second):
            self.assertEqual(loop.step(), 2)
            action = dict(zip(env._action_keys, env.sent[-1], strict=True))
            self.assertIs(action["env_id"], previous[1])
            self.assertIs(action["players.env_id"], previous[2])
            self.assertIs(action["players.action"], previous[0])
            self.assertIs(action["players.id"], previous[0])
            self.assertIs(action["list_action"], list_action)

    def test_balanced_commands_and_summary_schema(self):
        """Validate every planned row using the real paired-summary validator."""
        roots = {
            label: "/baseline" if label in "ab" else "/candidate"
            for label in "abcd"
        }
        plan = make_plan(roots, "/venv/bin/python")
        records = list(commands(plan))
        self.assertEqual(len(records), 64)
        self.assertEqual(records, list(commands(plan)))
        counts = collections.Counter()
        rows = []
        for item in records:
            label = item["label"]
            command = item["command"]
            self.assertEqual(command[0], "/venv/bin/python")
            self.assertEqual(
                command[command.index("--envpool-root") + 1], roots[label]
            )
            counts[item["env"], label, item["slot"]] += 1
            players = 1 if item["env"] == "DummyPlayers1" else 4
            row = {
                "env": item["env"],
                "num_envs": 64,
                "batch_size": 64,
                "threads": 4,
                "max_num_players": players,
                "pin": False,
                "async": False,
                "block": item["block"],
                "rep": item["block"],
                "slot": item["slot"],
                "variant": label,
                "order": item["order"],
                "expected_samples": 64,
                "warmup": 400,
                "requested_seconds": 3.0,
                "seed": 42,
                "affinity": [0, 1, 2, 3],
                "worker_affinity_offset": -1,
                "runtime_selection": "explicit-root",
                "runtime_root": roots[label],
                "binary_sha256": ("a" if label in "ab" else "b") * 64,
                "environment_responses": 8192,
                "env_steps": 8192,
                "vector_calls": 128,
            }
            validate_sample(row, item, plan)
            rows.append(row)
        self.assertEqual(len(counts), 2 * 4 * 4)
        self.assertEqual(set(counts.values()), {2})
        evidence = _validate(rows, plan, "binary_sha256")
        self.assertEqual(len(evidence), 2)
        for configuration in evidence.values():
            self.assertEqual(
                configuration["same_binary_pairs"]["status"], "verified"
            )

        rows[0]["max_num_players"] = 4 if rows[0]["max_num_players"] == 1 else 1
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            validate_sample(rows[0], records[0], plan)


if __name__ == "__main__":
    unittest.main()
