#!/usr/bin/env python3
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

"""One positive CartPole send-profile sample or exact-output validation record."""

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import weakref
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SOURCE / "benchmark/core_runtime"))
from check_rollouts import flatten
from runtime import load_runtime

spec = importlib.util.spec_from_file_location(
    "send_profile_helpers",
    SOURCE / "benchmark/core_runtime/binding_cost/probe.py",
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
CASES = {"s1": (20, 1), "s4": (20, 4), "m1": (256, 1), "m4": (256, 4)}
PHASES = (
    "setup",
    "set_action_slices_prior_owner_release_and_sync_accounting",
    "enqueue",
)


def digest(p):
    """Return the identity of an explicitly selected input file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_snapshot(stats, mode, calls, n, threads):
    """Check the measured schema and positive-case counter invariants."""
    if mode == "baseline":
        assert stats is None
        return
    active = mode in ("coarse", "fine")
    fine = mode == "fine"
    assert (
        stats["schema_version"] == 2
        and stats["enabled"] == active
        and stats["fine_enabled"] == fine
    )
    assert (
        stats["clock_errors"]
        == stats["invalid_interval_calls"]
        == stats["incomplete_calls"]
        == stats["untimed_completed_calls"]
        == 0
    )
    assert stats["valid"]
    assert (
        stats["started_calls"]
        == stats["completed_calls"]
        == stats["timed_calls"]
        == (calls if active else 0)
    )
    assert stats["timed_envs"] == (n * calls if active else 0)
    assert set(stats["phases"]) == set(PHASES)
    for value in stats["phases"].values():
        assert set(value) == {"wall_ns", "caller_cpu_ns"} and all(
            isinstance(v, int) and v >= 0 for v in value.values()
        )
        if not active:
            assert set(value.values()) == {0}
    child = stats["fine_enqueue"]
    assert (
        child["started_calls"]
        == child["completed_calls"]
        == child["timed_calls"]
        == (calls if fine else 0)
    )
    assert child["timed_actions"] == (n * calls if fine else 0)
    assert (
        child["clock_errors"]
        == child["invalid_interval_calls"]
        == child["incomplete_calls"]
        == child["untimed_completed_calls"]
        == 0
        and child["valid"]
    )
    assert set(child["phases"]) == {
        "producer_guard_and_reservation",
        "slot_fill",
        "work_permit_signal",
        "producer_guard_release",
    }
    assert 0 <= child["work_permit_posix_post_requests"] <= calls * threads
    if not fine:
        assert child["work_permit_posix_post_requests"] == 0
    for value in child["phases"].values():
        assert set(value) == {"wall_ns", "caller_cpu_ns"} and all(
            isinstance(v, int) and v >= 0 for v in value.values()
        )
        if not fine:
            assert set(value.values()) == {0}
    for clock in ["wall_ns", "caller_cpu_ns"]:
        assert (
            sum(v[clock] for v in child["phases"].values())
            <= stats["phases"]["enqueue"][clock]
        )
    assert stats["legacy_enqueue_wall_seconds_since_reset"] >= 0


def main():
    """Validate positive rollouts or collect one explicitly authorized sample."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--native-sha256", required=True)
    p.add_argument(
        "--mode", choices=["baseline", "off", "coarse", "fine"], required=True
    )
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--calls", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--save-rollout", type=Path)
    p.add_argument("--label", default="")
    p.add_argument(
        "--timing-authorized",
        action="store_true",
        help="Explicitly permit one timed sample; omit for positive rollout validation",
    )
    a = p.parse_args()
    assert a.calls > 0 and a.warmup >= 0
    if not a.save_rollout and not a.timing_authorized:
        p.error(
            "timing requires --timing-authorized; validation requires --save-rollout"
        )
    a.root = a.root.expanduser().resolve(strict=True)
    if a.save_rollout:
        a.save_rollout = a.save_rollout.expanduser().resolve()
        assert a.save_rollout.suffix == ".npz"
        assert not a.save_rollout.is_relative_to(SOURCE)
        assert not a.save_rollout.is_relative_to(a.root)
        assert not a.save_rollout.exists() and a.save_rollout.parent.is_dir()
    np, envpool = load_runtime(str(a.root))
    native = importlib.import_module(
        "envpool.classic_control.classic_control_envpool"
    )
    native_path = Path(native.__file__).resolve()
    assert digest(native_path) == a.native_sha256
    assert (
        Path(envpool.__file__)
        .resolve()
        .is_relative_to(a.root.resolve() / "envpool")
    )
    configure = getattr(native, "_diagnostic_send_profile_configure", None)
    snapshot = getattr(native, "_diagnostic_send_profile_snapshot", None)
    assert (configure is None) == (a.mode == "baseline") and (
        snapshot is None
    ) == (a.mode == "baseline")
    n, threads = CASES[a.case]
    before_pool = set(h.task_snapshot())
    env = envpool.make_gymnasium(
        "CartPole-v1", num_envs=n, batch_size=n, num_threads=threads, seed=42
    )
    observer = weakref.ref(env)
    action = np.zeros(
        (n,) + env.action_space.shape, dtype=env.action_space.dtype
    )
    step = None
    record = None
    try:
        assert not env.is_async
        if configure:
            configure(env, a.mode in ("coarse", "fine"), True, a.mode == "fine")
        initial = env.reset()
        if a.save_rollout:
            trace = {
                f"initial{k}": v.copy() for k, v in flatten(initial).items()
            }
            if configure:
                configure(
                    env, a.mode in ("coarse", "fine"), True, a.mode == "fine"
                )
            for index in range(32):
                current = ((np.arange(n) + index) % 2).astype(
                    env.action_space.dtype
                )
                before = current.tobytes()
                value = env.step(current)
                assert current.tobytes() == before
                for key, array in flatten(value).items():
                    trace[f"step{index}{key}"] = array.copy()
            stats = snapshot(env) if snapshot else None
            check_snapshot(stats, a.mode, 32, n, threads)
            np.savez_compressed(a.save_rollout, **trace)
            record = {
                "kind": "positive-validation",
                "mode": a.mode,
                "case": a.case,
                "calls": 32,
                "arrays": len(trace),
                "native_sha256": a.native_sha256,
                "profile": stats,
                "output": str(a.save_rollout),
                "input_buffers_unchanged": True,
            }
            initial = current = value = trace = None
        else:
            initial = None
            step = env.step
            for _ in range(a.warmup):
                step(action)
            if configure:
                configure(
                    env, a.mode in ("coarse", "fine"), True, a.mode == "fine"
                )
            input_hash = hashlib.sha256(action.tobytes()).hexdigest()
            sched_before = h.task_snapshot()
            p0 = time.process_time_ns()
            t0 = time.thread_time_ns()
            w0 = time.perf_counter_ns()
            for _ in range(a.calls):
                step(action)
            w1 = time.perf_counter_ns()
            t1 = time.thread_time_ns()
            p1 = time.process_time_ns()
            sched_after = h.task_snapshot()
            stats = snapshot(env) if snapshot else None
            check_snapshot(stats, a.mode, a.calls, n, threads)
            assert hashlib.sha256(action.tobytes()).hexdigest() == input_hash
            if a.mode in ("coarse", "fine"):
                assert (
                    sum(v["wall_ns"] for v in stats["phases"].values())
                    <= w1 - w0
                )
                assert (
                    sum(v["caller_cpu_ns"] for v in stats["phases"].values())
                    <= t1 - t0
                )
            record = {
                "kind": "timing",
                "mode": a.mode,
                "case": a.case,
                "num_envs": n,
                "batch_size": n,
                "threads": threads,
                "seed": 42,
                "vector_calls": a.calls,
                "env_steps": a.calls * n,
                "warmup": a.warmup,
                "wall_ns": w1 - w0,
                "caller_cpu_ns": t1 - t0,
                "process_cpu_ns": p1 - p0,
                "env_steps_per_s": a.calls * n * 1e9 / (w1 - w0),
                "profile": stats,
                "scheduler": h.task_delta(sched_before, sched_after),
                "input_buffers_unchanged": True,
                "input_sha256": input_hash,
                "label": a.label,
            }
        wrappers = {
            name: {
                "path": str(Path(sys.modules[name].__file__).resolve()),
                "sha256": digest(sys.modules[name].__file__),
            }
            for name in [
                "envpool.python.envpool",
                "envpool.python.gymnasium_envpool",
                "envpool.python.protocol",
            ]
        }
        record.update(
            native_path=str(native_path),
            native_sha256=digest(native_path),
            wrapper_modules=wrappers,
            driver_sha256=digest(__file__),
            helper_sha256=digest(h.__file__),
            package_path=str(Path(envpool.__file__).resolve()),
            python=sys.version,
            numpy=np.__version__,
            affinity=sorted(os.sched_getaffinity(0)),
            profile_boundary="Setup; SetAction+slice+prior-owner release+sync accounting; enqueue with existing timer bookkeeping. Top-level recording/local destruction/outer binding-GIL work excluded from coarse phases; paired clock reads have skew. Fine queue recording is outside its fine phases but inside coarse enqueue and legacy timers. Fine queue phases nest inside enqueue and must not be added again. POSIX release requests are observed under instrumentation, not actual kernel wake syscalls. No overhead subtraction.",
        )
        assert record["native_sha256"] == a.native_sha256
    finally:
        step = action = None
        env.close()
        env = None
        gc.collect()
    record["pool_released"] = observer() is None
    record["new_live_tids_after_cleanup"] = sorted(
        set(h.task_snapshot()) - before_pool, key=int
    )
    assert record["pool_released"] and not record["new_live_tids_after_cleanup"]
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
