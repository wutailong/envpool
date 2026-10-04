# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Dependency-free aggregate observers; no arrays, RNG use or runtime imports."""

import math
import time

PARENTS = {
    "policy_train": None,
    "collect": None,
    "ppo_update": None,
    "buffer_reset": None,
    "env_step": "collect",
    "env_reset": "collect",
}
TOP_LEVEL = tuple(name for name, parent in PARENTS.items() if parent is None)
# Only the float perf_counter -> integer ns wall envelope gets this tolerance.
# Nested intervals use the same integer clock and must be exactly nonnegative.
WALL_CLOCK_TOLERANCE_NS = 1000


class PhaseClock:
    """Observe synchronous nested calls; pass clock=None for count-only tests.

    Inactive calls do not read the clock, update counts, retain arguments/results,
    or alter exceptions. Active calls retain only aggregate integers. Measuring
    adds overhead: phase-exclusive residuals include observer and Python overhead.
    This is intentionally single-threaded, like the reference training driver.
    """

    def __init__(self, clock=time.perf_counter_ns):
        """Create an inactive observer with an injectable monotonic ns clock."""
        self.clock = clock
        self.active = False
        self._stack = []
        self._invalid_nesting = False
        self._negative_interval = False
        self._negative_exclusive = False
        self._stats = {
            name: {
                "calls": 0,
                "exceptions": 0,
                "inclusive_ns": 0,
                "direct_child_ns": 0,
            }
            for name in PARENTS
        }

    def call(self, name, function, *args, **kwargs):
        """Forward the exact arguments and returned object; re-raise unchanged."""
        if not self.active:
            return function(*args, **kwargs)
        stats = self._stats[name]
        parent = self._stack[-1][0] if self._stack else None
        if parent != PARENTS[name]:
            self._invalid_nesting = True
        stats["calls"] += 1
        frame = [name, 0]
        started = self.clock() if self.clock is not None else None
        self._stack.append(frame)
        try:
            return function(*args, **kwargs)
        except BaseException:
            stats["exceptions"] += 1
            raise
        finally:
            ended = self.clock() if self.clock is not None else None
            self._stack.pop()
            if self.clock is not None:
                duration = ended - started
                self._negative_interval |= duration < 0
                self._negative_exclusive |= duration < frame[1]
                stats["inclusive_ns"] += duration
                stats["direct_child_ns"] += frame[1]
                if self._stack:
                    self._stack[-1][1] += duration

    def report(
        self,
        training_seconds=None,
        expected_updates=None,
        expected_env_step_calls=None,
    ):
        """Validate nesting and budgets and build a disjoint timing partition.

        Env step/reset are already inside collect. Never sum inclusive collect
        and inclusive env time. The partition replaces inclusive collect with
        collect-exclusive plus env step/reset; all other top phases are disjoint.
        """
        if self.active or self._stack:
            raise ValueError("report requires an inactive, finished observer")
        if self._invalid_nesting:
            raise ValueError("unexpected phase nesting")
        if self._negative_interval or self._negative_exclusive:
            raise ValueError("negative integer-clock interval or residual")
        if expected_updates is not None and any(
            self._stats[name]["calls"] != expected_updates for name in TOP_LEVEL
        ):
            raise ValueError("top-level counts differ from measured updates")
        if expected_env_step_calls is not None and (
            self._stats["env_step"]["calls"] != expected_env_step_calls
        ):
            raise ValueError("EnvPool step calls differ from the fixed budget")
        timed = self.clock is not None
        phases = {}
        for name, stats in self._stats.items():
            exclusive = stats["inclusive_ns"] - stats["direct_child_ns"]
            phases[name] = {
                "calls": stats["calls"],
                "exceptions": stats["exceptions"],
                "parent": PARENTS[name],
                "inclusive_ns": stats["inclusive_ns"] if timed else None,
                "direct_child_ns": stats["direct_child_ns"] if timed else None,
                "exclusive_ns": exclusive if timed else None,
            }
        partition = None
        raw_other = None
        wall_ns = None
        if timed:
            if training_seconds is None or not (
                math.isfinite(training_seconds) and training_seconds > 0
            ):
                raise ValueError(
                    "a positive finite whole-training time is required"
                )
            env_ns = sum(
                phases[name]["inclusive_ns"]
                for name in ("env_step", "env_reset")
            )
            if phases["collect"]["direct_child_ns"] != env_ns:
                raise ValueError(
                    "collect children do not equal EnvPool call totals"
                )
            wall_ns = round(training_seconds * 1_000_000_000)
            top_ns = sum(phases[name]["inclusive_ns"] for name in TOP_LEVEL)
            raw_other = wall_ns - top_ns
            if raw_other < -WALL_CLOCK_TOLERANCE_NS:
                raise ValueError("phase totals exceed whole-training wall time")
            partition = {
                "policy_train_ns": phases["policy_train"]["exclusive_ns"],
                "collect_excluding_env_calls_ns": phases["collect"][
                    "exclusive_ns"
                ],
                "env_step_in_collect_ns": phases["env_step"]["inclusive_ns"],
                "env_reset_in_collect_ns": phases["env_reset"]["inclusive_ns"],
                "ppo_update_ns": phases["ppo_update"]["exclusive_ns"],
                "buffer_reset_ns": phases["buffer_reset"]["exclusive_ns"],
                "loop_other_ns": max(0, raw_other),
            }
        return {
            "mode": "timed" if timed else "count-only",
            "phases": phases,
            "disjoint_partition_ns": partition,
            "training_wall_ns": wall_ns,
            "raw_loop_other_ns": raw_other,
            "validation": {
                "valid": True,
                "integer_interval_tolerance_ns": 0,
                "wall_envelope_tolerance_ns": WALL_CLOCK_TOLERANCE_NS,
                "wall_roundoff_clamped_ns": max(0, -(raw_other or 0)),
                "expected_updates": expected_updates,
                "expected_env_step_calls": expected_env_step_calls,
                "explanation": "integer nesting is exact; only the float wall envelope permits 1 microsecond roundoff",
            },
        }


class ProfiledEnv:
    """Forward the reference synchronous Collector's EnvPool interface.

    No argument/result copies, default injection, array inspection or RNG calls.
    Attribute reads/writes/deletes and len are delegated. Like RecordedEnv in the
    parity harness, proxy identity/type and arbitrary special methods differ;
    this is not a general-purpose transparent proxy for every Python protocol.
    """

    def __init__(self, env, observer):
        """Retain only the existing environment and observer references."""
        object.__setattr__(self, "_profile_env", env)
        object.__setattr__(self, "_profile_observer", observer)

    def __len__(self):
        """Preserve vector environment size."""
        return len(self._profile_env)

    def __getattr__(self, name):
        """Delegate ordinary attributes without copying their values."""
        return getattr(self._profile_env, name)

    def __setattr__(self, name, value):
        """Keep ordinary writes on the underlying environment."""
        setattr(self._profile_env, name, value)

    def __delattr__(self, name):
        """Keep ordinary deletions on the underlying environment."""
        delattr(self._profile_env, name)

    def step(self, *args, **kwargs):
        """Time only the exact forwarded synchronous step call."""
        return self._profile_observer.call(
            "env_step", self._profile_env.step, *args, **kwargs
        )

    def reset(self, *args, **kwargs):
        """Time only the exact forwarded reset, including subset resets."""
        return self._profile_observer.call(
            "env_reset", self._profile_env.reset, *args, **kwargs
        )
