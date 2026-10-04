# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Light deterministic tests only: never import or execute a training runtime."""

import ast
import json
import random
import unittest
from pathlib import Path

import profile_ppo
from phase_clock import PhaseClock, ProfiledEnv


class FakeClock:
    """Return a fixed sequence; excess timing calls fail the test."""

    def __init__(self, *times):
        """Retain the supplied clock readings without sampling real time."""
        self.times = iter(times)
        self.calls = 0

    def __call__(self):
        """Return the next supplied reading and count the sample."""
        self.calls += 1
        return next(self.times)


class FakeEnv:
    """Keep argument references and a shared return object for proxy checks."""

    def __init__(self):
        """Create a fixed fake environment with shared return values."""
        self.calls = []
        self.result = ({"obs": []}, {"info": []})
        self.error = None
        self.is_async = False

    def __len__(self):
        """Return the reference vector width."""
        return 20

    def step(self, *args, **kwargs):
        """Retain argument references and return or raise the supplied object."""
        self.calls.append(("step", args, kwargs))
        if self.error:
            raise self.error
        return self.result

    def reset(self, *args, **kwargs):
        """Retain reset arguments without injecting defaults."""
        self.calls.append(("reset", args, kwargs))
        return self.result


class PhaseClockTests(unittest.TestCase):
    """Exercise exact nesting, inactivity, forwarding, failures and budgets."""

    def test_nested_partition_and_counts(self):
        """Verify inclusive children and exclusive parents form one partition."""
        clock = FakeClock(0, 10, 30, 35, 40, 50, 60, 100, 110, 115, 120, 123)
        obs = PhaseClock(clock)
        obs.active = True

        def collect():
            obs.call("env_step", lambda: None)
            obs.call("env_reset", lambda: None)

        obs.call("collect", collect)
        obs.call("ppo_update", lambda: None)
        obs.call("buffer_reset", lambda: None)
        obs.call("policy_train", lambda: None)
        obs.active = False
        result = obs.report(150 / 1e9, 1, 1)
        phases = result["phases"]
        self.assertEqual(phases["collect"]["inclusive_ns"], 50)
        self.assertEqual(phases["collect"]["exclusive_ns"], 25)
        self.assertEqual(phases["env_step"]["inclusive_ns"], 20)
        self.assertEqual(phases["env_reset"]["inclusive_ns"], 5)
        self.assertEqual(result["raw_loop_other_ns"], 52)
        self.assertEqual(sum(result["disjoint_partition_ns"].values()), 150)
        self.assertEqual(clock.calls, 12)
        json.dumps(result, allow_nan=False)

    def test_inactive_setup_and_warmup_do_not_count_or_read_clock(self):
        """Exclude setup and disposable warmup from clocks and counters."""
        clock = FakeClock()
        obs = PhaseClock(clock)
        env = FakeEnv()
        proxy = ProfiledEnv(env, obs)
        self.assertIs(proxy.reset(), env.result)
        self.assertIs(obs.call("collect", proxy.step, [1]), env.result)
        self.assertEqual(clock.calls, 0)
        self.assertTrue(all(v["calls"] == 0 for v in obs._stats.values()))

    def test_count_only_and_exact_proxy_forwarding(self):
        """Preserve argument aliases, attributes and RNG in count-only mode."""
        obs = PhaseClock(None)
        env = FakeEnv()
        proxy = ProfiledEnv(env, obs)
        action, ids, options = [], [], {}
        state = random.getstate()
        self.assertEqual(len(proxy), 20)
        self.assertIs(proxy.calls, env.calls)
        proxy.marker = options
        self.assertIs(env.marker, options)
        del proxy.marker
        self.assertFalse(hasattr(env, "marker"))
        obs.active = True
        returned = obs.call("collect", proxy.step, action, env_id=ids)
        self.assertIs(returned, env.result)
        self.assertIs(env.calls[-1][1][0], action)
        self.assertIs(env.calls[-1][2]["env_id"], ids)
        obs.call("collect", proxy.reset, env_id=ids, options=options)
        self.assertEqual(env.calls[-1][1], ())  # No injected default argument.
        self.assertIs(env.calls[-1][2]["options"], options)
        obs.active = False
        result = obs.report(expected_env_step_calls=1)
        self.assertEqual(result["mode"], "count-only")
        self.assertIsNone(result["phases"]["env_step"]["inclusive_ns"])
        self.assertIsNone(result["disjoint_partition_ns"])
        self.assertEqual(random.getstate(), state)

    def test_original_exception_identity_and_balanced_stack(self):
        """Re-raise the identical exception while balancing nested accounting."""
        obs = PhaseClock(FakeClock(0, 2, 5, 9))
        env = FakeEnv()
        env.error = KeyboardInterrupt("same exception")
        proxy = ProfiledEnv(env, obs)
        obs.active = True
        with self.assertRaises(KeyboardInterrupt) as caught:
            obs.call("collect", proxy.step)
        self.assertIs(caught.exception, env.error)
        obs.active = False
        result = obs.report(10 / 1e9)
        self.assertEqual(result["phases"]["env_step"]["exceptions"], 1)
        self.assertEqual(result["phases"]["collect"]["exceptions"], 1)
        self.assertEqual(result["phases"]["collect"]["exclusive_ns"], 6)

    def test_invalid_nesting_and_budget_rejected(self):
        """Reject unexpected nesting and incorrect measured call counts."""
        obs = PhaseClock(None)
        obs.active = True
        obs.call("env_step", lambda: None)
        obs.active = False
        with self.assertRaisesRegex(ValueError, "nesting"):
            obs.report()
        for kwargs in ({"expected_updates": 1}, {"expected_env_step_calls": 1}):
            with self.assertRaises(ValueError):
                PhaseClock(None).report(**kwargs)

    def test_negative_integer_intervals_rejected(self):
        """Reject negative elapsed and exclusive integer-clock intervals."""
        for clock, nested in (
            (FakeClock(4, 3), False),
            (FakeClock(0, 1, 8, 5), True),
        ):
            obs = PhaseClock(clock)
            obs.active = True
            obs.call(
                "collect",
                lambda obs=obs, nested=nested: (
                    obs.call("env_step", lambda: None) if nested else None
                ),
            )
            obs.active = False
            with self.assertRaisesRegex(ValueError, "negative"):
                obs.report(1.0)

    def test_wall_tolerance_is_explicit_not_silent(self):
        """Report tolerated envelope roundoff and reject larger discrepancies."""
        obs = PhaseClock(FakeClock(0, 2000))
        obs.active = True
        obs.call("collect", lambda: None)
        obs.active = False
        report = obs.report(1500 / 1e9)
        self.assertEqual(report["raw_loop_other_ns"], -500)
        self.assertEqual(report["validation"]["wall_roundoff_clamped_ns"], 500)
        self.assertEqual(report["disjoint_partition_ns"]["loop_other_ns"], 0)
        with self.assertRaisesRegex(ValueError, "exceed"):
            obs.report(500 / 1e9)
        for invalid in (None, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                obs.report(invalid)

    def test_report_requires_finished_region(self):
        """Require measurement to finish before building a report."""
        obs = PhaseClock(None)
        obs.active = True
        with self.assertRaisesRegex(ValueError, "inactive"):
            obs.report()


class RemoveObservation(ast.NodeTransformer):
    """Erase only the documented observer insertions for an exact AST audit."""

    def visit_Assign(self, node):
        """Remove only observer construction, activation and report assignments."""
        target = ast.unparse(node.targets[0])
        if target in ("observer", "observer.active", "phase_profile"):
            return None
        if (
            target == "env"
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "ProfiledEnv"
        ):
            return None
        return self.generic_visit(node)

    def visit_Try(self, node):
        """Unwrap only the measured-region observer deactivation guard."""
        if (
            len(node.finalbody) == 1
            and isinstance(node.finalbody[0], ast.Assign)
            and ast.unparse(node.finalbody[0].targets[0]) == "observer.active"
        ):
            return [self.visit(child) for child in node.body]
        return self.generic_visit(node)

    def visit_Call(self, node):
        """Restore the exact original callable and forwarded arguments."""
        if ast.unparse(node.func) == "observer.call":
            node.func = node.args[1]
            node.args = node.args[2:]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        """Remove only the additional observer and provenance report fields."""
        remove = {
            "phase_profile",
            "instrumentation",
            "reference_script_sha256",
            "phase_helper_sha256",
        }
        pairs = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not (isinstance(key, ast.Constant) and key.value in remove)
        ]
        node.keys = [key for key, _ in pairs]
        node.values = [value for _, value in pairs]
        return self.generic_visit(node)


class ReferenceFidelityTests(unittest.TestCase):
    """Statically prove the training copy changes only declared observations."""

    def test_train_and_train_body_match_after_removing_observation(self):
        """Compare original training ASTs after erasing declared observations."""
        original = ast.parse(profile_ppo.REFERENCE_SCRIPT.read_text())
        observed = ast.parse(Path(profile_ppo.__file__).read_text())
        for name in ("train", "_train"):
            left = next(
                n
                for n in original.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            right = next(
                n
                for n in observed.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            right = RemoveObservation().visit(right)
            self.assertEqual(ast.dump(left), ast.dump(right), name)

    def test_original_helpers_and_reference_budget(self):
        """Check helper identity and the unchanged reference work budget."""
        for name in (
            "parse_args",
            "load_config",
            "load_package",
            "fingerprint",
            "sha256",
            "write_result",
        ):
            self.assertIs(
                getattr(profile_ppo, name),
                getattr(profile_ppo._reference, name),
            )
        cfg = profile_ppo.load_config(
            profile_ppo._reference.DEFAULT_CONFIG, 100
        )
        self.assertEqual(cfg["num_threads"], 2)
        self.assertEqual(cfg["training_num"], 20)
        self.assertEqual(cfg["updates"] * cfg["steps_per_collect"], 256000)
        self.assertEqual(cfg["updates"] * cfg["steps_per_collect"] // 20, 12800)
        self.assertEqual(
            cfg["updates"]
            * cfg["steps_per_collect"]
            * cfg["repeat_per_collect"]
            // cfg["batch_size"],
            8000,
        )


if __name__ == "__main__":
    unittest.main()
