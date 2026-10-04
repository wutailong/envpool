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
"""Positive-input tests for the isolated max_num_players cache candidate."""

import argparse
import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--native", action="store_true")
PARSER.add_argument(
    "--source",
    type=Path,
    required=True,
    help="Complete envpool/python source directory to check",
)
PARSER.add_argument(
    "--runtime",
    type=Path,
    required=True,
    help="Package parent containing the built envpool runtime",
)
ARGS, UNITTEST_ARGS = PARSER.parse_known_args()
sys.path.insert(0, str(ARGS.runtime))
# The package namespace keeps all generated classes bound to one mixin identity.
PACKAGE = types.ModuleType("wrapper_metadata_candidate")
PACKAGE.__path__ = [str(ARGS.source)]
sys.modules[PACKAGE.__name__] = PACKAGE
MODULE = importlib.import_module(PACKAGE.__name__ + ".envpool")
EnvPoolMixin = MODULE.EnvPoolMixin


@contextmanager
def config_call_count():
    """Count Python calls only; collect no timing samples."""
    counts = {"config": 0}
    code = EnvPoolMixin.config.fget.__code__
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is code:
            counts["config"] += 1

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


class CountingSpec:
    """Expose fresh metadata while counting config and action-spec reads."""

    _config_keys = ("num_envs", "max_num_players")
    _action_keys = ("env_id", "players.env_id", "action")
    _action_spec = ((np.int32, (-1,)),) * 3

    def __init__(self, max_num_players):
        """Initialize the configured fixture and counters."""
        self.values = (2, max_num_players)
        self.config_reads = 0
        self.action_spec_reads = 0

    @property
    def _config_values(self):
        self.config_reads += 1
        return self.values

    @property
    def action_array_spec(self):
        """Build a fresh action-spec mapping for every access."""
        self.action_spec_reads += 1
        return {
            key: SimpleNamespace(shape=(-1,), dtype=np.dtype(np.int32))
            for key in self._action_keys
        }


class Pool(EnvPoolMixin):
    """Supply the Python wrapper with a positive-input spec fixture."""

    def __init__(self, max_num_players=4):
        """Initialize the configured fixture and counters."""
        self._spec = self.spec = CountingSpec(max_num_players)
        self._state_keys = ("info:players.env_id",)

    def _recv(self):
        return [self.received_players]

    def _to(self, state, reset, return_info):
        return state


def action(player_count, env_ids=(0, 1)):
    """Build a positive simplified action for the selected player count."""
    return {
        "env_id": np.asarray(env_ids, dtype=np.int32),
        "action": np.arange(player_count, dtype=np.int32),
    }


class WrapperMetadataTest(unittest.TestCase):
    """Check scalar caching and unchanged positive-input wrapper behavior."""

    def test_config_remains_fresh_and_mutations_are_isolated(self):
        """Keep public config dictionaries independent of the scalar cache."""
        env = Pool()
        before = env.config
        before["max_num_players"] = 1
        first = env._from(action(6))
        after = env.config
        self.assertIsNot(before, after)
        self.assertEqual(after["max_num_players"], 4)
        after["max_num_players"] = 1
        np.testing.assert_array_equal(first[1], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(env._from(action(6))[1], first[1])
        self.assertEqual(env.config["max_num_players"], 4)

    def test_pool_caches_are_independent(self):
        """Keep scalar cache values independent across pool instances."""
        single = Pool(1)
        multi = Pool(4)
        np.testing.assert_array_equal(single._from(action(2))[1], [0, 1])
        np.testing.assert_array_equal(
            multi._from(action(6))[1], [0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(single._max_num_players, 1)
        self.assertEqual(multi._max_num_players, 4)
        self.assertEqual(single.spec.config_reads, 1)
        self.assertEqual(multi.spec.config_reads, 1)

    def test_uniform_multiplayer_mapping(self):
        """Preserve uniform player mapping and requested environment order."""
        env = Pool(4)
        for _ in range(3):
            converted = env._from(action(6, (1, 0)))
            np.testing.assert_array_equal(converted[1], [1, 1, 1, 0, 0, 0])
            self.assertEqual(converted[1].dtype, np.int32)

    def test_exact_maximum_mapping_before_and_after_warming(self):
        """Accept the exact configured player maximum before and after warming."""
        env = Pool(4)
        self.assertFalse(hasattr(env, "_max_num_players"))
        expected = [0, 0, 0, 0, 1, 1, 1, 1]
        np.testing.assert_array_equal(env._from(action(8))[1], expected)
        self.assertEqual(env._max_num_players, 4)
        env._from(action(6))
        np.testing.assert_array_equal(env._from(action(8))[1], expected)
        self.assertEqual(env.spec.config_reads, 1)

    def test_variable_mapping_preserves_requested_order(self):
        """Reuse variable player mappings in requested environment order."""
        env = Pool(4)
        env._last_players_env_id = np.asarray([0, 0, 1, 1, 1], dtype=np.int32)
        converted = env._from(action(5, (1, 0)))
        np.testing.assert_array_equal(converted[1], [1, 1, 1, 0, 0])
        self.assertEqual(env.spec.config_reads, 1)

    def test_one_player_per_requested_environment(self):
        """Keep one-player-per-environment mappings unchanged."""
        env = Pool(4)
        converted = env._from(action(2))
        np.testing.assert_array_equal(converted[1], [0, 1])

    def test_explicit_mapping_bypasses_config_and_inference(self):
        """Bypass metadata reads when the action supplies its player mapping."""
        env = Pool(4)
        values = action(5)
        explicit = np.asarray([0, 0, 1, 1, 1], dtype=np.int32)
        values["players"] = {"env_id": explicit}
        converted = env._from(values)
        self.assertIs(converted[1], explicit)
        self.assertEqual(env.spec.config_reads, 0)
        self.assertEqual(env.spec.action_spec_reads, 0)
        self.assertFalse(hasattr(env, "_max_num_players"))

    def test_config_view_is_built_once_for_repeated_single_player_inference(
        self,
    ):
        """Construct one config view for repeated single-player inference."""
        env = Pool(1)
        with config_call_count() as counts:
            for _ in range(8):
                env._from(action(2))
        self.assertEqual(counts["config"], 1)
        self.assertEqual(env.spec.config_reads, 1)
        self.assertEqual(env.spec.action_spec_reads, 0)

    def test_config_view_is_built_once_without_caching_action_specs(self):
        """Cache only the scalar while continuing to reconstruct action specs."""
        env = Pool(4)
        with config_call_count() as counts:
            for _ in range(8):
                env._from(action(6))
        self.assertEqual(counts["config"], 1)
        self.assertEqual(env.spec.config_reads, 1)
        self.assertEqual(env.spec.action_spec_reads, 8)

    def test_dynamic_override_retains_both_original_config_reads(self):
        """Read dynamic config overrides at both original inference points."""

        class DynamicPool(Pool):
            @property
            def config(self):
                self.reads += 1
                return {"max_num_players": next(self.limits)}

        env = DynamicPool()
        env.reads = 0
        env.limits = iter((1, 2, 4, 2, 4))
        np.testing.assert_array_equal(env._from(action(2))[1], [0, 1])
        for _ in range(2):
            np.testing.assert_array_equal(
                env._from(action(6))[1], [0, 0, 0, 1, 1, 1]
            )
        self.assertEqual(env.reads, 5)
        self.assertEqual(env.spec.config_reads, 0)
        self.assertFalse(hasattr(env, "_max_num_players"))

    def test_inherited_override_remains_dynamic(self):
        """Honor config overrides inherited through another Python subclass."""

        class OverridePool(Pool):
            @property
            def config(self):
                self.reads += 1
                return {"max_num_players": next(self.limits)}

        class ChildPool(OverridePool):
            pass

        env = ChildPool()
        env.reads = 0
        env.limits = iter((2, 4))
        np.testing.assert_array_equal(
            env._from(action(6))[1], [0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(env.reads, 2)
        self.assertFalse(hasattr(env, "_max_num_players"))

    def test_override_installed_after_cache_is_honored(self):
        """Honor a config override installed after the native cache is warm."""

        class MutablePool(Pool):
            pass

        env = MutablePool(1)
        env._from(action(2))
        self.assertEqual(env._max_num_players, 1)
        reads = []
        limits = iter((2, 4))

        def config(self):
            reads.append(True)
            return {"max_num_players": next(limits)}

        MutablePool.config = property(config)
        np.testing.assert_array_equal(
            env._from(action(6))[1], [0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(len(reads), 2)
        self.assertEqual(env._max_num_players, 1)
        del MutablePool.config
        np.testing.assert_array_equal(env._from(action(2))[1], [0, 1])
        self.assertEqual(env.spec.config_reads, 1)

    def test_override_installed_between_first_and_fallback_reads(self):
        """Honor an override installed during player-action inference."""

        class MutablePool(Pool):
            def _player_action_count(self, adict):
                type(self).config = property(
                    lambda self: {"max_num_players": 4}
                )
                return super()._player_action_count(adict)

        env = MutablePool(2)
        np.testing.assert_array_equal(
            env._from(action(6))[1], [0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(env._max_num_players, 2)

    def test_override_removed_between_first_and_fallback_reads(self):
        """Read native config after an override disappears during inference."""

        class MutablePool(Pool):
            @property
            def config(self):
                return {"max_num_players": 2}

            def _player_action_count(self, adict):
                del type(self).config
                return super()._player_action_count(adict)

        env = MutablePool(4)
        np.testing.assert_array_equal(
            env._from(action(6))[1], [0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(env._max_num_players, 4)
        self.assertEqual(env.spec.config_reads, 1)

    def test_plain_action_still_copies_and_normalizes_dtype(self):
        """Preserve action and environment-ID copying and dtype normalization."""
        env = Pool(1)
        original = np.arange(2, dtype=np.int32)
        env_ids = np.arange(2, dtype=np.int64)
        converted = env._from(original, env_ids)
        np.testing.assert_array_equal(converted[-1], original)
        self.assertFalse(np.shares_memory(converted[-1], original))
        self.assertEqual(converted[0].dtype, np.int32)
        self.assertFalse(np.shares_memory(converted[0], env_ids))
        cast_action = np.arange(2, dtype=np.int64)
        converted = env._from(cast_action, env_ids)
        self.assertEqual(converted[-1].dtype, np.int32)
        self.assertFalse(np.shares_memory(converted[-1], cast_action))

    def test_scalar_and_list_environment_normalization(self):
        """Preserve positive scalar and list ID normalization during inference."""
        env = Pool(1)
        for env_ids, expected in ((1, [1]), ([1, 0], [1, 0])):
            values = {
                "env_id": env_ids,
                "action": np.ones(len(expected), dtype=np.int32),
            }
            players_env_id = env._infer_players_env_id(values)
            np.testing.assert_array_equal(players_env_id, expected)
            self.assertEqual(players_env_id.dtype, np.int32)

    def test_recv_mapping_still_copies_caller_visible_array(self):
        """Retain a defensive copy of the caller-visible received mapping."""
        env = Pool(4)
        env.received_players = np.asarray([0, 0, 1, 1, 1], dtype=np.int32)
        returned = env.recv()
        self.assertIs(returned[0], env.received_players)
        self.assertFalse(
            np.shares_memory(env._last_players_env_id, returned[0])
        )
        returned[0][:] = 1
        np.testing.assert_array_equal(env._from(action(5))[1], [0, 0, 1, 1, 1])


if ARGS.native:
    from envpool.dummy.dummy_envpool import _DummyEnvPool, _DummyEnvSpec

    API = importlib.import_module(PACKAGE.__name__ + ".api")
    DummyEnvSpec, DummyDMEnvPool, DummyGymnasiumEnvPool = API.py_env(
        _DummyEnvSpec, _DummyEnvPool
    )

    class NativeDummyMetadataTest(unittest.TestCase):
        """Check generated wrappers against the existing native Dummy extension."""

        @staticmethod
        def make_env(max_num_players, pool_type=DummyDMEnvPool):
            """Construct a native-backed wrapper without resetting or stepping it."""
            config = DummyEnvSpec.gen_config(
                num_envs=2,
                batch_size=2,
                num_threads=1,
                max_num_players=max_num_players,
            )
            return pool_type(DummyEnvSpec(config))

        @staticmethod
        def multiplayer_action(player_count, explicit=None):
            """Build a positive Dummy action with an optional explicit mapping."""
            players = {
                "id": np.arange(player_count, dtype=np.int32),
                "action": np.arange(player_count, dtype=np.int32),
            }
            if explicit is not None:
                players["env_id"] = explicit
            return {
                "env_id": np.array([0, 1], dtype=np.int32),
                "list_action": np.zeros((2, 6), dtype=np.float64),
                "players": players,
            }

        def test_native_normal_path_constructs_one_config_view(self):
            """Check mappings and single config-view construction on both APIs."""
            for pool_type in (DummyDMEnvPool, DummyGymnasiumEnvPool):
                for max_num_players, player_count in ((1, 2), (4, 6), (4, 8)):
                    with self.subTest(
                        wrapper=pool_type.__name__, player_count=player_count
                    ):
                        env = self.make_env(max_num_players, pool_type)
                        self.assertIs(type(env).config, EnvPoolMixin.config)
                        expected = np.repeat([0, 1], player_count // 2)
                        with config_call_count() as counts:
                            for _ in range(8):
                                converted = env._from(
                                    self.multiplayer_action(player_count)
                                )
                                np.testing.assert_array_equal(
                                    converted[1], expected
                                )
                        self.assertEqual(counts["config"], 1)
                        self.assertEqual(env._max_num_players, max_num_players)
                        env.close()

        def test_native_config_is_fresh_and_pools_are_independent(self):
            """Keep native public config views fresh and pool caches independent."""
            single = self.make_env(1)
            multi = self.make_env(4)
            changed = multi.config
            changed["max_num_players"] = 1
            np.testing.assert_array_equal(
                single._from(self.multiplayer_action(2))[1], [0, 1]
            )
            np.testing.assert_array_equal(
                multi._from(self.multiplayer_action(6))[1], [0, 0, 0, 1, 1, 1]
            )
            self.assertIsNot(changed, multi.config)
            self.assertEqual(multi.config["max_num_players"], 4)
            changed = multi.config
            changed["max_num_players"] = 1
            np.testing.assert_array_equal(
                multi._from(self.multiplayer_action(6))[1], [0, 0, 0, 1, 1, 1]
            )
            self.assertEqual(single._max_num_players, 1)
            self.assertEqual(multi._max_num_players, 4)
            single.close()
            multi.close()

        def test_native_variable_mapping_and_explicit_bypass(self):
            """Preserve variable and explicit player mappings on both native APIs."""
            for pool_type in (DummyDMEnvPool, DummyGymnasiumEnvPool):
                with self.subTest(wrapper=pool_type.__name__):
                    env = self.make_env(4, pool_type)
                    expected = np.asarray([0, 0, 1, 1, 1], dtype=np.int32)
                    with config_call_count() as counts:
                        converted = env._from(
                            self.multiplayer_action(5, expected)
                        )
                    self.assertEqual(counts["config"], 0)
                    self.assertIs(converted[1], expected)
                    self.assertFalse(hasattr(env, "_max_num_players"))
                    env._last_players_env_id = expected.copy()
                    np.testing.assert_array_equal(
                        env._from(self.multiplayer_action(5))[1], expected
                    )
                    env.close()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *UNITTEST_ARGS])
