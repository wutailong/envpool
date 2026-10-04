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
"""Reuse the normal Container harness with synchronized, interleaved players.

This case requires --max-num-players 4. Only the action layout and
per-environment seeds differ from the inherited harness.
The inherited main owns runtime selection, normal conversion, timing, counters,
and native hashes. Its printed row is augmented after measurement finishes.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "container_ownership")
)

import bench_container as inherited

NUM_ENVS = inherited.NUM_ENVS
MAX_PLAYERS = 4
CASE_NAME = "DummyInterleavedPlayers4"
PLAYER_ORDER = "interleaved-by-player-index"
ENV_SEEDS = [inherited.SEED] * NUM_ENVS


def make_permutations(np, num_envs=NUM_ENVS, max_players=MAX_PLAYERS):
    """Precompute all supported row orders once, outside the timed loop."""
    return {
        num_envs * players: np
        .arange(num_envs * players)
        .reshape(num_envs, players)
        .T.ravel()
        for players in range(1, max_players + 1)
    }


def validate_layout(np, env_ids, player_env_ids, player_ids, num_envs=NUM_ENVS):
    """Reject anything other than a full, grouped, equal-player response."""
    if (
        env_ids.ndim != 1
        or player_env_ids.ndim != 1
        or player_ids.ndim != 1
        or not np.array_equal(np.sort(env_ids), np.arange(num_envs))
        or len(player_env_ids) != len(player_ids)
        or len(player_ids) % num_envs
    ):
        raise ValueError("expected a full batch with equal player counts")
    players = len(player_ids) // num_envs
    if not 1 <= players <= MAX_PLAYERS or not np.array_equal(
        player_env_ids, np.repeat(env_ids, players)
    ):
        raise ValueError("expected grouped, equal-player response rows")
    return players


class InterleavedDummyLoop(inherited.DummyLoop):
    """Copy all three player fields with one cached permutation per step."""

    def __init__(self, env, list_action, warmup):
        """Prepare permutations and a validation-only warmup step method."""
        super().__init__(env, list_action)
        self.np = importlib.import_module("numpy")
        self.permutations = make_permutations(self.np)
        action_keys = tuple(env._action_keys)
        self.player_mapping = tuple(
            pair
            for pair in self.mapping
            if pair[0] != action_keys.index("env_id")
        )
        self.env_target = action_keys.index("env_id")
        self.player_env_target = action_keys.index("players.env_id")
        self.player_env_source = self.state_keys.index("info:players.env_id")
        self.player_id_source = self.state_keys.index("info:players.id")
        self.warmup = warmup
        self.warmup_completed = 0
        self.warmup_player_counts = set()
        self.scattered_warmup_steps = 0
        self.step = self._warmup_step

    def _validate_layout(self):
        return validate_layout(
            self.np,
            self.state[self.env_id_index],
            self.state[self.player_env_source],
            self.state[self.player_id_source],
        )

    def _prepare_action(self):
        permutation = self.permutations[len(self.state[self.player_id_source])]
        self.action[self.env_target] = self.state[self.env_id_index]
        for target, source in self.player_mapping:
            self.action[target] = self.state[source][permutation]

    def _advance(self):
        self.env._send(self.action)
        self.state = self.env._recv()
        return len(self.state[self.env_id_index])

    def _warmup_step(self):
        players = self._validate_layout()
        self._prepare_action()
        # This checks the actual action passed to native _send, not merely the
        # precomputed permutation. For p > 1, each env's rows are N apart.
        self.np.testing.assert_array_equal(
            self.action[self.player_env_target],
            self.np.tile(self.state[self.env_id_index], players),
        )
        count = self._advance()
        self.warmup_player_counts.add(players)
        self.scattered_warmup_steps += players > 1
        self.warmup_completed += 1
        if self.warmup_completed == self.warmup:
            self.step = self._timed_step
        return count

    def _timed_step(self):
        self._prepare_action()
        return self._advance()

    def validate(self, np, max_num_players):
        """Keep Container validation and assert real warmup scatter coverage."""
        super().validate(np, max_num_players)
        self._validate_layout()
        if self.warmup_completed == self.warmup:
            if not self.scattered_warmup_steps:
                raise ValueError("warmup never sent interleaved p > 1 actions")


def synchronized_module(name):
    """Wrap only spec construction; preserve the native pool and binary path."""
    module = importlib.import_module(name)
    if name != "envpool.dummy.dummy_envpool":
        raise ValueError("unexpected module requested by inherited harness")
    native_spec = module._DummyEnvSpec

    def make_spec(values):
        values = list(values)
        values[native_spec._config_keys.index("env_seed")] = ENV_SEEDS.copy()
        return native_spec(tuple(values))

    make_spec._config_keys = native_spec._config_keys
    make_spec._default_config_values = native_spec._default_config_values
    return SimpleNamespace(
        __file__=module.__file__,
        _DummyEnvSpec=make_spec,
        _DummyEnvPool=module._DummyEnvPool,
    )


def main():
    """Adapt inherited setup and output without changing its measurement loop."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-num-players", type=int, choices=(MAX_PLAYERS,))
    parser.add_argument("--warmup", type=int, default=inherited.WARMUP)
    args, _ = parser.parse_known_args()
    if args.warmup < 2:
        parser.error("interleaving requires at least two untimed warmup steps")
    loops = []

    def make_loop(env, list_action):
        loop = InterleavedDummyLoop(env, list_action, args.warmup)
        loops.append(loop)
        return loop

    rows = []

    def capture_row(text, *, flush):
        rows.append(json.loads(text))

    with (
        patch.object(inherited, "__doc__", __doc__),
        patch.object(inherited, "DummyLoop", make_loop),
        patch.object(
            inherited,
            "importlib",
            SimpleNamespace(import_module=synchronized_module),
        ),
        patch.object(inherited, "print", capture_row, create=True),
    ):
        inherited.main()
    if len(rows) != 1 or len(loops) != 1:
        raise RuntimeError("inherited harness did not emit exactly one sample")
    row = rows[0]
    row.update(
        env=CASE_NAME,
        player_order=PLAYER_ORDER,
        env_seed=ENV_SEEDS,
        warmup_player_counts=sorted(loops[0].warmup_player_counts),
        scattered_warmup_steps=loops[0].scattered_warmup_steps,
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
