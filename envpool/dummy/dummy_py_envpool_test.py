# Copyright 2021 Garena Online Private Limited
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
"""Unit test for dummy envpool and speed benchmark."""

import gc
import os
import sys
import time
import weakref
from typing import Any

import numpy as np
from absl import logging
from absl.testing import absltest
from envpool.dummy.dummy_envpool import _DummyEnvPool, _DummyEnvSpec

from envpool.python.api import py_env
from envpool.python.protocol import EnvPool

DummyEnvSpec, _DummyDMEnvPool, _ = py_env(_DummyEnvSpec, _DummyEnvPool)


def _make_dummy_dm_env() -> EnvPool:
    config = DummyEnvSpec.gen_config(
        num_envs=2,
        batch_size=2,
        max_num_players=4,
    )
    return _DummyDMEnvPool(DummyEnvSpec(config))


def _make_multiplayer_action(
    player_count: int,
    players_env_id: np.ndarray | None = None,
) -> dict[str, object]:
    players: dict[str, np.ndarray] = {
        "id": np.arange(player_count, dtype=np.int32),
        "action": np.arange(player_count, dtype=np.int32),
    }
    action: dict[str, Any] = {
        "env_id": np.array([0, 1], dtype=np.int32),
        "list_action": np.zeros((2, 6), dtype=np.float64),
        "players": players,
    }
    if players_env_id is not None:
        players["env_id"] = players_env_id
    return action


class _DummyEnvPoolTest(absltest.TestCase):
    def test_readonly_reset_rejection_releases_input(self) -> None:
        """Rejected positive ID arrays must not retain a Python owner."""
        config = DummyEnvSpec.gen_config(
            num_envs=2, batch_size=2, num_threads=1, max_num_players=1
        )
        env = _DummyEnvPool(_DummyEnvSpec(config))
        ids = np.arange(2, dtype=np.int32)
        ids.setflags(write=False)
        observer = weakref.ref(ids)
        refs = sys.getrefcount(ids)
        for _ in range(16):
            with self.assertRaisesRegex(ValueError, "array is not writeable"):
                env._reset(ids)
            self.assertEqual(sys.getrefcount(ids), refs)
        del ids
        gc.collect()
        self.assertIsNone(observer())
        env._reset(np.arange(2, dtype=np.int32))
        state = dict(zip(env._state_keys, env._recv(), strict=True))
        np.testing.assert_array_equal(state["info:env_id"], [0, 1])

    def test_readonly_send_releases_partial_conversions(self) -> None:
        """A late conversion error must release both prior and failing fields."""
        config = DummyEnvSpec.gen_config(
            num_envs=2, batch_size=2, num_threads=1, max_num_players=1
        )
        env = _DummyEnvPool(_DummyEnvSpec(config))
        env._reset(np.arange(2, dtype=np.int32))
        env._recv()
        action = (
            np.arange(2, dtype=np.int32),
            np.arange(2, dtype=np.int32),
            np.zeros((2, 6), dtype=np.float64),
            np.zeros(2, dtype=np.int32),
            np.zeros(2, dtype=np.int32),
        )
        action[-1].setflags(write=False)
        observers = [weakref.ref(field) for field in action]
        refs = [sys.getrefcount(field) for field in action]
        for _ in range(16):
            with self.assertRaisesRegex(ValueError, "array is not writeable"):
                env._send(action)
            self.assertEqual([sys.getrefcount(field) for field in action], refs)
        action[-1].setflags(write=True)
        env._send(action)
        state = dict(zip(env._state_keys, env._recv(), strict=True))
        np.testing.assert_array_equal(state["info:env_id"], [0, 1])
        del action, env
        gc.collect()
        self.assertTrue(all(observer() is None for observer in observers))

    def test_config(self) -> None:
        ref_config_keys = [
            "num_envs",
            "batch_size",
            "num_threads",
            "max_num_players",
            "thread_affinity_offset",
            "base_path",
            "seed",
            "env_seed",
            "gym_reset_return_info",
            "state_num",
            "action_num",
            "max_episode_steps",
        ]
        default_conf = _DummyEnvSpec._default_config_values
        self.assertTrue(isinstance(default_conf, tuple))
        config_keys = _DummyEnvSpec._config_keys
        self.assertTrue(isinstance(config_keys, list))
        self.assertEqual(len(default_conf), len(config_keys))
        self.assertEqual(sorted(config_keys), sorted(ref_config_keys))

    def test_spec(self) -> None:
        conf = _DummyEnvSpec._default_config_values
        env_spec = _DummyEnvSpec(conf)
        state_spec = env_spec._state_spec
        action_spec = env_spec._action_spec
        state_keys = env_spec._state_keys
        action_keys = env_spec._action_keys
        self.assertTrue(isinstance(state_spec, tuple))
        self.assertTrue(isinstance(action_spec, tuple))
        state_spec = dict(zip(state_keys, state_spec, strict=False))
        action_spec = dict(zip(action_keys, action_spec, strict=False))
        # default value of state_num is 10
        self.assertEqual(state_spec["obs:raw"][1][-1], 10)
        self.assertEqual(state_spec["obs:dyn"][1][1][-1], 10)
        # change conf and see if it can successfully change state_spec
        # directly send dict or expose config as dict?
        conf = dict(zip(_DummyEnvSpec._config_keys, conf, strict=False))
        conf["state_num"] = 666
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        state_spec = dict(zip(state_keys, env_spec._state_spec, strict=False))
        self.assertEqual(state_spec["obs:raw"][1][-1], 666)

    def test_envpool(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = num_envs = 100
        conf["batch_size"] = batch = 31
        conf["num_threads"] = os.cpu_count()
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        env = _DummyEnvPool(env_spec)
        state_keys = env._state_keys
        total = 100000
        env._reset(np.arange(num_envs, dtype=np.int32))
        t = time.time()
        for _ in range(total):
            state = dict(zip(state_keys, env._recv(), strict=False))
            action = {
                "env_id": state["info:env_id"],
                "players.env_id": state["info:players.env_id"],
                "list_action": np.zeros((batch, 6), dtype=np.float64),
                "players.id": state["info:players.id"],
                "players.action": state["info:players.id"],
            }
            env._send(tuple(action.values()))
        duration = time.time() - t
        fps = total * batch / duration
        logging.info(f"FPS = {fps:.6f}")

    def test_xla(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = 100
        conf["batch_size"] = 31
        conf["num_threads"] = os.cpu_count()
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        env = _DummyEnvPool(env_spec)
        xla_failed = False
        try:
            _ = env._xla()
        except RuntimeError:
            logging.info(
                "XLA on Dummy failed because dummy has Container typed state."
            )
            xla_failed = True
        self.assertTrue(xla_failed)

    def test_env_seed_overrides_sequential_seeding(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = 3
        conf["batch_size"] = 3
        conf["max_num_players"] = 1
        conf["env_seed"] = [1, 3, 5]
        env = _DummyEnvPool(_DummyEnvSpec(tuple(conf.values())))
        env._reset(np.arange(3, dtype=np.int32))

        action = (
            np.arange(3, dtype=np.int32),
            np.arange(3, dtype=np.int32),
            np.zeros((3, 6), dtype=np.float64),
            np.zeros((3,), dtype=np.int32),
            np.zeros((3,), dtype=np.int32),
        )

        env._recv()  # consume reset output

        env._send(action)
        state = dict(zip(env._state_keys, env._recv(), strict=False))
        np.testing.assert_array_equal(
            state["done"],
            np.array([True, False, False]),
        )

        env._send(action)
        _ = env._recv()

        env._send(action)
        state = dict(zip(env._state_keys, env._recv(), strict=False))
        np.testing.assert_array_equal(
            state["done"],
            np.array([True, True, False]),
        )


class _DummyEnvPoolLifetimeTest(absltest.TestCase):
    @staticmethod
    def _make_env(max_num_players: int) -> EnvPool:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf.update(
            num_envs=8,
            batch_size=8,
            num_threads=1,
            max_num_players=max_num_players,
            seed=5,
            # Dummy initializes only these two columns of obs:raw.
            state_num=2,
        )
        return _DummyEnvPool(_DummyEnvSpec(tuple(conf.values())))

    @staticmethod
    def _recv(env: EnvPool) -> dict[str, np.ndarray]:
        return dict(zip(env._state_keys, env._recv(), strict=True))

    @staticmethod
    def _action(state: dict[str, np.ndarray], step: int) -> list[np.ndarray]:
        env_id = state["info:env_id"].copy()
        action = {
            "env_id": env_id,
            "players.env_id": state["info:players.env_id"].copy(),
            "list_action": np.repeat(
                (env_id.astype(np.float64) + step)[:, None], 6, axis=1
            ),
            "players.action": state["info:players.id"].copy() + step,
            "players.id": state["info:players.id"].copy(),
        }
        return [action[key] for key in _DummyEnvPool._action_keys]

    @staticmethod
    def _copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        copied = {key: value.copy() for key, value in state.items()}
        # An object-array copy alone would still share its Container arrays.
        for index, value in enumerate(state["obs:dyn"]):
            copied["obs:dyn"][index] = value.copy()
        return copied

    def _assert_state_equal(
        self,
        actual: dict[str, np.ndarray],
        expected: dict[str, np.ndarray],
    ) -> None:
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in actual.items():
            self.assertEqual(value.shape, expected[key].shape, key)
            self.assertEqual(value.dtype, expected[key].dtype, key)
            if key == "obs:dyn":
                for inner, expected_inner in zip(
                    value, expected[key], strict=True
                ):
                    np.testing.assert_array_equal(inner, expected_inner)
            else:
                np.testing.assert_array_equal(value, expected[key], err_msg=key)

    @staticmethod
    def _ordered_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        env_order = np.argsort(state["info:env_id"])
        player_order = np.lexsort((
            state["info:players.id"],
            state["info:players.env_id"],
        ))
        player_keys = {
            "reward",
            "discount",
            "obs:raw",
            "obs:dyn",
            "info:players.env_id",
            "info:players.id",
            "info:players.done",
        }
        return {
            key: value[player_order if key in player_keys else env_order]
            for key, value in state.items()
        }

    def test_observations_survive_steps_and_pool_destruction(self) -> None:
        for max_num_players in (1, 4):
            with self.subTest(max_num_players=max_num_players):
                env = self._make_env(max_num_players)
                env._reset(np.arange(8, dtype=np.int32))
                state = self._recv(env)
                raw_view = state["obs:raw"][:, :1]
                dynamic_view = state["obs:dyn"][-1][:, :1]
                expected_raw = raw_view.copy()
                expected_dynamic = dynamic_view.copy()
                retained = []
                for step in range(64):
                    env._send(self._action(state, step))
                    state = self._recv(env)
                    if step < 4:
                        retained.append((state, self._copy_state(state)))
                    for inner, env_id in zip(
                        state["obs:dyn"],
                        state["info:players.env_id"],
                        strict=True,
                    ):
                        self.assertEqual(inner.shape, (int(env_id) + 1, 2))
                        np.testing.assert_array_equal(
                            inner, np.full(inner.shape, env_id, dtype=np.int32)
                        )

                for actual, expected in retained:
                    self._assert_state_equal(actual, expected)
                del state, env
                gc.collect()
                # The reset batch's outer object array is already gone; its
                # retained inner view must own the dynamic allocation itself.
                np.testing.assert_array_equal(raw_view, expected_raw)
                np.testing.assert_array_equal(dynamic_view, expected_dynamic)
                for actual, expected in retained:
                    self._assert_state_equal(actual, expected)

    def test_temporary_actions_match_contiguous_trajectory(self) -> None:
        for max_num_players in (1, 4):
            for variant in ("contiguous", "strided", "cast", "strided_cast"):
                with self.subTest(
                    max_num_players=max_num_players, variant=variant
                ):
                    env = self._make_env(max_num_players)
                    reference = self._make_env(max_num_players)
                    env._reset(np.arange(8, dtype=np.int32))
                    reference._reset(np.arange(8, dtype=np.int32))
                    state = self._recv(env)
                    expected = self._recv(reference)
                    self._assert_state_equal(
                        self._ordered_state(state),
                        self._ordered_state(expected),
                    )
                    for step in range(32):
                        expected_action = self._action(expected, step)
                        action = self._action(state, step)
                        if "cast" in variant:
                            action = [
                                value.astype(
                                    np.float32
                                    if value.dtype == np.float64
                                    else np.int64
                                )
                                for value in action
                            ]
                        if "strided" in variant:
                            action = [
                                np.repeat(value, 2, axis=-1)[..., ::2]
                                for value in action
                            ]
                            self.assertTrue(
                                all(
                                    not value.flags.c_contiguous
                                    for value in action
                                )
                            )
                        env._send(action)
                        # Drop every caller-owned array (including strided
                        # bases) before receiving any asynchronous work.
                        del action
                        gc.collect()
                        # Reuse similarly sized allocations while the queued
                        # work still has to own its original/coerced arrays.
                        replacement = tuple(
                            np.zeros_like(value) for value in expected_action
                        )
                        reference._send(expected_action)
                        state = self._recv(env)
                        expected = self._recv(reference)
                        del replacement
                        self._assert_state_equal(
                            self._ordered_state(state),
                            self._ordered_state(expected),
                        )
                    del env, reference


class _EnvPoolMixinRegressionTest(absltest.TestCase):
    def test_from_repeats_env_id_for_uniform_multiplayer_action(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(player_count=6)
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1],
            np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        )

    def test_recv_cache_handles_variable_player_counts(self) -> None:
        env = _make_dummy_dm_env()
        env._last_players_env_id = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        action = _make_multiplayer_action(player_count=5)
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1], np.array([0, 0, 1, 1, 1], dtype=np.int32)
        )

    def test_from_preserves_explicit_players_env_id(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(
            player_count=5,
            players_env_id=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        )
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1], np.array([0, 0, 1, 1, 1], dtype=np.int32)
        )

    def test_from_raises_when_players_env_id_is_ambiguous(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(player_count=5)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot infer players.env_id"
        ):
            env._from(action)


if __name__ == "__main__":
    absltest.main()
