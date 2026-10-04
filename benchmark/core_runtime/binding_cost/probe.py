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

"""Bounded, measurement-only CartPole binding probe. One process, one sample."""

# Preserve the measured helper's function bodies and validation operations.
# These local style exemptions avoid refactoring the original loops/hooks.
# ruff: noqa: B043, B905, D102, D103, D107, RUF007
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import resource
import sys
import threading
import time
import weakref
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
PHASES = ("_from", "_check_action", "_send", "_recv", "_to")
CASES = {"small": (20, 1), "medium": (256, 4)}
MODES = ("public", "raw", "profile", "profile-no-clock")
MISSING = object()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def load_selected_runtime(args):
    """Reuse the audited loader without putting the source checkout on sys.path."""
    source = args.source.resolve(strict=True)
    helper = source / "benchmark/core_runtime/runtime.py"
    spec = importlib.util.spec_from_file_location(
        "measurement_runtime_loader", helper
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "envpool" in sys.modules:
        raise RuntimeError("each runtime requires a fresh interpreter")
    np, envpool = module.load_runtime(
        str(args.envpool_root) if args.envpool_root else None
    )
    return np, envpool, helper


def runtime_record(args, np, envpool, env, helper):
    native = {}
    for cls in type(env).__mro__:
        module = sys.modules.get(cls.__module__)
        path = getattr(module, "__file__", None)
        if path and any(
            str(path).endswith(s)
            for s in importlib.machinery.EXTENSION_SUFFIXES
        ):
            native[cls.__module__] = file_record(path)
    if not native:
        raise RuntimeError("could not resolve native module from pool MRO")
    if len(native) != 1:
        raise RuntimeError(
            f"expected one CartPole native module, found: {native}"
        )
    native_record = next(iter(native.values()))
    native_path = native_record["path"]
    actual_sha256 = native_record["sha256"]
    if actual_sha256 != args.native_sha256:
        raise RuntimeError(
            f"native SHA-256 mismatch for {native_path}: "
            f"expected {args.native_sha256}, got {actual_sha256}"
        )
    wrappers = {}
    for name in ("envpool.python.envpool", "envpool.python.gymnasium_envpool"):
        module = sys.modules.get(name)
        if module is not None:
            wrappers[name] = file_record(module.__file__)
    return {
        "variant": args.variant,
        "runtime_selection": "explicit-root"
        if args.envpool_root
        else "interpreter",
        "requested_envpool_root": str(args.envpool_root.resolve())
        if args.envpool_root
        else None,
        "package": file_record(envpool.__file__),
        "wrapper_modules": wrappers,
        "native_modules": native,
        "expected_native_sha256": args.native_sha256,
        "native_sha256_verified": True,
        "runtime_loader": file_record(helper),
        "driver": file_record(__file__),
        "source_root": str(args.source.resolve()),
        "python_executable": sys.executable,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "reported_logical_cpu_count": os.cpu_count(),
        "inherited_calling_thread_affinity": sorted(os.sched_getaffinity(0)),
        "numpy_version": np.__version__,
        "numpy_path": str(Path(np.__file__).resolve()),
        "envpool_version": getattr(envpool, "__version__", None),
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "LD_PRELOAD",
            )
        },
    }


def task_snapshot():
    rows = {}
    for task in sorted(
        Path("/proc/self/task").iterdir(), key=lambda p: int(p.name)
    ):
        try:
            status = dict(
                line.split(":", 1)
                for line in (task / "status").read_text().splitlines()
            )
            rows[task.name] = {
                "name": status["Name"].strip(),
                "cpus_allowed_list": status["Cpus_allowed_list"].strip(),
                "schedstat": list(
                    map(int, (task / "schedstat").read_text().split())
                ),
                "voluntary_context_switches": int(
                    status["voluntary_ctxt_switches"]
                ),
                "involuntary_context_switches": int(
                    status["nonvoluntary_ctxt_switches"]
                ),
            }
        except (OSError, KeyError, ValueError) as exc:
            rows[task.name] = {"unavailable": type(exc).__name__}
    return rows


def task_delta(before, after):
    rows = {}
    for tid in sorted(before.keys() & after.keys(), key=int):
        a, b = before[tid], after[tid]
        if "unavailable" in a or "unavailable" in b:
            rows[tid] = {"unavailable": True}
            continue
        sd = [y - x for x, y in zip(a["schedstat"], b["schedstat"])]
        rows[tid] = {
            "run_ns": sd[0],
            "runnable_wait_ns": sd[1],
            "timeslices": sd[2],
            "voluntary_context_switches": b["voluntary_context_switches"]
            - a["voluntary_context_switches"],
            "involuntary_context_switches": b["involuntary_context_switches"]
            - a["involuntary_context_switches"],
            "cpus_allowed_list_before": a["cpus_allowed_list"],
            "cpus_allowed_list_after": b["cpus_allowed_list"],
        }
    return {
        "per_tid": rows,
        "thread_set_stable": before.keys() == after.keys(),
        "new_tids": sorted(after.keys() - before.keys(), key=int),
        "exited_tids": sorted(before.keys() - after.keys(), key=int),
        "total_run_ns": sum(x.get("run_ns", 0) for x in rows.values()),
        "total_runnable_wait_ns": sum(
            x.get("runnable_wait_ns", 0) for x in rows.values()
        ),
        "total_timeslices": sum(x.get("timeslices", 0) for x in rows.values()),
    }


def system_snapshot():
    result = {"advisory_only": True}
    try:
        result["loadavg"] = list(os.getloadavg())
        result["proc_stat_cpu_ticks"] = {
            parts[0]: list(map(int, parts[1:]))
            for line in Path("/proc/stat").read_text().splitlines()
            if (parts := line.split()) and parts[0].startswith("cpu")
        }
        result["cpu_ticks_per_second"] = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError) as exc:
        result["proc_stat_unavailable"] = type(exc).__name__
    try:
        mhz = {}
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            fields = dict(
                line.split(":", 1) for line in block.splitlines() if ":" in line
            )
            fields = {k.strip(): v.strip() for k, v in fields.items()}
            if "processor" in fields and "cpu MHz" in fields:
                mhz[fields["processor"]] = float(fields["cpu MHz"])
        result["cpuinfo_mhz"] = mhz
    except (OSError, ValueError) as exc:
        result["cpuinfo_unavailable"] = type(exc).__name__
    return result


def system_delta(before, after):
    rows = {}
    a, b = (
        before.get("proc_stat_cpu_ticks", {}),
        after.get("proc_stat_cpu_ticks", {}),
    )
    for cpu in a.keys() & b.keys():
        d = [y - x for x, y in zip(a[cpu], b[cpu])]
        total = sum(d[:8])  # guest and guest_nice are already included above.
        idle_iowait = sum(d[3:5])
        rows[cpu] = {
            "ticks": d,
            "total_ticks": total,
            "busy_excluding_idle_iowait_ticks": total - idle_iowait,
            "steal_ticks": d[7] if len(d) > 7 else None,
        }
    return {"advisory_only": True, "per_cpu": rows}


def parse_mask(value):
    result = set()
    for piece in value.split(","):
        ends = piece.split("-")
        result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def affinity_check(args, threads, pre_pool_tids, after_warmup):
    caller = str(threading.get_native_id())
    new_tids = sorted(after_warmup.keys() - pre_pool_tids, key=int)
    expected = list(range(1, 1 + threads))
    actual = {
        tid: after_warmup[tid].get("cpus_allowed_list") for tid in new_tids
    }
    masks = {
        tid: parse_mask(mask) if mask is not None else set()
        for tid, mask in actual.items()
    }
    # The stock-buffer allocator is created before the stepping workers and is
    # not pinned by thread_affinity_offset. Classify from actual masks, not TID
    # ordering or pthread's ignored return status.
    pinned = {
        tid: mask
        for tid, mask in masks.items()
        if len(mask) == 1 and next(iter(mask)) in expected
    }
    other = {tid: actual[tid] for tid in new_tids if tid not in pinned}
    verified = (
        len(pinned) == threads
        and sorted(tuple(sorted(m)) for m in pinned.values())
        == [(c,) for c in expected]
        and after_warmup.get(caller, {}).get("cpus_allowed_list") == "0"
    )
    return {
        "requested": args.pin,
        "caller_tid": caller,
        "caller_cpu_requested": 0 if args.pin else None,
        "worker_offset_requested": 1 if args.pin else -1,
        "new_pool_tids": new_tids,
        "new_pool_tid_masks": actual,
        "verified_pinned_worker_tid_masks": {tid: actual[tid] for tid in pinned}
        if args.pin
        else None,
        "other_new_pool_tid_masks": other if args.pin else actual,
        "pin_verified_from_proc_status": verified if args.pin else None,
        "note": "Actual masks read after warmup; the core pthread affinity return is not trusted. The stock-buffer allocator and any other helper remain separately reported. Logical CPU placement does not establish physical-core isolation.",
    }


class PhaseProfiler:
    """Instance-only leaf hooks. close() removes every bound-pool reference."""

    def __init__(self, env, clocks=True):
        self.env = env
        self.clocks = clocks
        self.saved = {}
        self.originals = {}
        self.active = None
        self.reset()

    def reset(self):
        self.counts = {name: 0 for name in PHASES}
        self.totals = {
            name: {k: 0 for k in ("wall_ns", "process_cpu_ns", "thread_cpu_ns")}
            for name in PHASES
        }

    def install(self):
        try:
            for name in PHASES:
                self.saved[name] = self.env.__dict__.get(name, MISSING)
                original = getattr(self.env, name)
                self.originals[name] = original
                setattr(self.env, name, self.wrap(name, original))
        except BaseException:
            self.close()
            raise
        return self

    def wrap(self, name, original):
        def wrapped(*args, **kwargs):
            if self.active is not None:
                raise RuntimeError(
                    f"phases are not disjoint: {self.active} -> {name}"
                )
            self.active = name
            try:
                if self.clocks:
                    p0 = time.process_time_ns()
                    t0 = time.thread_time_ns()
                    w0 = time.perf_counter_ns()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        w1 = time.perf_counter_ns()
                        t1 = time.thread_time_ns()
                        p1 = time.process_time_ns()
                        total = self.totals[name]
                        total["wall_ns"] += w1 - w0
                        total["thread_cpu_ns"] += t1 - t0
                        total["process_cpu_ns"] += p1 - p0
                return original(*args, **kwargs)
            finally:
                self.counts[name] += 1
                self.active = None

        return wrapped

    def summary(self, calls):
        if self.counts != dict.fromkeys(PHASES, calls):
            raise AssertionError(
                f"unexpected public call flow: {self.counts}, expected {calls} each"
            )
        return {
            "clocks_enabled": self.clocks,
            "counts": self.counts.copy(),
            "phase_totals": copy.deepcopy(self.totals) if self.clocks else None,
            "note": "Hooks preserve env.step -> send/recv. Five phases are non-nested. Uninstrumented recv bookkeeping and hook/clock costs remain in outer totals. Worker CPU may overlap native calls; process CPU is not exclusive ownership attribution. No overhead subtraction.",
        }

    def close(self):
        env = self.env
        if env is None:
            return
        try:
            for name, previous in self.saved.items():
                if previous is MISSING:
                    if name in env.__dict__:
                        delattr(env, name)
                else:
                    setattr(env, name, previous)
        finally:
            self.saved.clear()
            self.originals.clear()
            self.env = None
            self.active = None


def make_pool(args, np, envpool, case):
    n, threads = CASES[case]
    pool = envpool.make_gymnasium(
        "CartPole-v1",
        num_envs=n,
        batch_size=n,
        num_threads=threads,
        seed=42,
        thread_affinity_offset=1 if args.pin else -1,
    )
    if pool.is_async:
        raise AssertionError("this probe is synchronous only")
    action = np.zeros(
        (n,) + pool.action_space.shape, dtype=pool.action_space.dtype
    )
    return pool, action


def action_digest(arrays):
    return [
        {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "writeable": bool(a.flags.writeable),
            "c_contiguous": bool(a.flags.c_contiguous),
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
        }
        for a in arrays
    ]


def runner(env, action, mode):
    if mode == "raw":
        converted = env._from(action)
        env._check_action(converted)
        send, recv = env._send, env._recv

        def step():
            send(converted)
            return recv()

        return step, converted
    public_step = env.step

    def step():
        return public_step(action)

    return step, None


def run_sample(args, np, envpool, helper):
    n, threads = CASES[args.case]
    original_affinity = set(os.sched_getaffinity(0))
    before_pool = task_snapshot()
    env = action = step = converted = profiler = None
    try:
        env, action = make_pool(args, np, envpool, args.case)
        identity = runtime_record(args, np, envpool, env, helper)
        env.reset()
        if args.mode.startswith("profile"):
            profiler = PhaseProfiler(
                env, clocks=args.mode == "profile"
            ).install()
        step, converted = runner(env, action, args.mode)
        inputs = [action] + (converted if converted is not None else [])
        input_before = action_digest(inputs)
        # Workers are already created. Only the calling thread is pinned here.
        if args.pin:
            os.sched_setaffinity(0, {0})
        for _ in range(args.warmup):
            step()
        affinity_snapshot = task_snapshot()
        affinity = affinity_check(
            args, threads, before_pool.keys(), affinity_snapshot
        )
        if args.pin and not affinity["pin_verified_from_proc_status"]:
            raise RuntimeError(
                "requested pinning not verified: " + json.dumps(affinity)
            )
        if profiler:
            profiler.reset()
        system_before = system_snapshot() if args.system_diagnostics else None
        sched_before = task_snapshot()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        started_unix_ns = time.time_ns()
        p0 = time.process_time_ns()
        t0 = time.thread_time_ns()
        w0 = time.perf_counter_ns()
        for _ in range(args.calls):
            step()
        w1 = time.perf_counter_ns()
        t1 = time.thread_time_ns()
        p1 = time.process_time_ns()
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        sched_after = task_snapshot()
        system_after = system_snapshot() if args.system_diagnostics else None
        input_after = action_digest(inputs)
        if input_after != input_before:
            raise AssertionError("action buffers changed during the run")
        phase = profiler.summary(args.calls) if profiler else None
        wall_ns, process_ns, thread_ns = w1 - w0, p1 - p0, t1 - t0
        return {
            "schema": "envpool-binding-phase-probe-v1",
            "kind": "timed_sample",
            "label": args.label,
            "mode": args.mode,
            "pid": os.getpid(),
            "sample_started_unix_ns": started_unix_ns,
            "runtime": identity,
            "case": args.case,
            "env": "CartPole-v1",
            "num_envs": n,
            "batch_size": n,
            "threads": threads,
            "async": False,
            "seed": 42,
            "vector_calls": args.calls,
            "env_steps": n * args.calls,
            "warmup_excluded_vector_calls": args.warmup,
            "wall_ns": wall_ns,
            "process_cpu_ns": process_ns,
            "calling_thread_cpu_ns": thread_ns,
            "us_per_vector_call": wall_ns / args.calls / 1000,
            "env_steps_per_second": n * args.calls * 1e9 / wall_ns,
            "phase_profile": phase,
            "input_buffers_before": input_before,
            "input_buffers_unchanged": True,
            "mode_boundary": (
                "Diagnostic bypass: one _from and _check_action preparation excluded, then repeated _send(preconverted), _recv; output wrapping and public recv metadata bookkeeping excluded. Buffers remain writable ordinary ndarrays but are never mutated or replaced during the run. This is not a behavior-preserving production change."
                if args.mode == "raw"
                else "Actual public env.step(action); preallocated zero action, public conversion/check/send/recv/wrapping retained."
            ),
            "outer_clock_boundary": "process CPU, caller thread CPU, wall start; exactly calls vector steps; wall, caller thread CPU, process CPU end. Outer CPU clocks include nested clock overhead; no phase clocks in public/raw modes.",
            "affinity": affinity,
            "per_tid_after_warmup": affinity_snapshot,
            "per_tid_before": sched_before,
            "per_tid_after": sched_after,
            "scheduler_delta": task_delta(sched_before, sched_after),
            "process_usage_delta": {
                key: getattr(usage_after, key) - getattr(usage_before, key)
                for key in ("ru_nvcsw", "ru_nivcsw", "ru_minflt", "ru_majflt")
            },
            "peak_rss_kb": usage_after.ru_maxrss,
            "diagnostic_boundary": "All /proc, resource, hash, affinity and system diagnostics are outside the timed loop. Scheduler/resource snapshots bracket slightly wider intervals than the outer clocks.",
            "system_before": system_before,
            "system_after": system_after,
            "system_delta": system_delta(system_before, system_after)
            if system_before
            else None,
        }
    finally:
        if profiler:
            profiler.close()
        step = converted = profiler = action = None
        # Input arrays do not retain the pool, but clear their list as well.
        if "inputs" in locals():
            inputs.clear()
        if env is not None:
            env.close()
        env = None
        gc.collect()
        if args.pin:
            os.sched_setaffinity(0, original_affinity)


def freeze(np, obj):
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: freeze(np, v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(freeze(np, v) for v in obj)
    if isinstance(obj, list):
        return [freeze(np, v) for v in obj]
    return copy.deepcopy(obj)


def assert_exact(np, a, b, path="root"):
    if type(a) is not type(b):
        raise AssertionError(f"{path}: types differ {type(a)} != {type(b)}")
    if isinstance(a, np.ndarray):
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or a.tobytes(order="C") != b.tobytes(order="C")
        ):
            raise AssertionError(f"{path}: array dtype/shape/bytes differ")
    elif isinstance(a, dict):
        if list(a) != list(b):
            raise AssertionError(f"{path}: dict keys/order differ")
        for key in a:
            assert_exact(np, a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: sequence lengths differ")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_exact(np, x, y, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"{path}: scalar values differ")


def validation_rollout(args, np, envpool, helper, case, mode):
    env = action = step = converted = profiler = original_recv = (
        capture_recv
    ) = None
    saved_recv = MISSING
    states = []
    outputs = []
    baseline_tids = set(task_snapshot())
    try:
        env, action = make_pool(args, np, envpool, case)
        identity = runtime_record(args, np, envpool, env, helper)
        pool_ref = weakref.ref(env)
        metadata = freeze(
            np,
            {
                "config": env.config,
                "metadata": env.metadata,
                "state_keys": env._state_keys,
                "action_keys": env._action_keys,
            },
        )
        saved_recv = env.__dict__.get("_recv", MISSING)
        original_recv = env._recv

        def capture_recv():
            state = original_recv()
            states.append(freeze(np, state))
            return state

        env._recv = capture_recv
        outputs.append(freeze(np, env.reset()))
        if mode.startswith("profile"):
            # Correctness-only validation: install the same hooks with clocks off.
            profiler = PhaseProfiler(env, clocks=False).install()
        step, converted = runner(env, action, mode)
        input_before = action_digest([action] + (converted or []))
        for _ in range(32):
            out = step()
            if mode == "raw":
                # Validation-only adapter, never executed in a timed raw run.
                out = env._to(out, False, True)
            outputs.append(freeze(np, out))
        if profiler:
            profiler.summary(32)
        if action_digest([action] + (converted or [])) != input_before:
            raise AssertionError("validation action buffers mutated")
        terminal_count = 0
        autoreset_count = 0
        for current, following in zip(outputs[1:-1], outputs[2:]):
            done = current[2] | current[3]
            terminal_count += int(done.sum())
            if np.any(done):
                if not np.all(following[4]["elapsed_step"][done] == 0):
                    raise AssertionError(
                        "terminal slot was not auto-reset on the next vector call"
                    )
                autoreset_count += int(done.sum())
        if terminal_count == 0 or autoreset_count == 0:
            raise AssertionError(
                "32-step validation did not cover termination plus auto-reset"
            )
        result = {
            "metadata": metadata,
            "outputs": outputs,
            "native_states": states,
        }
        audit = {
            "mode": mode,
            "steps": 32,
            "reset_calls": 1,
            "terminal_slots_followed_by_autoreset": autoreset_count,
            "native_recv_states_compared": len(states),
            "all_action_buffers_unchanged": True,
            "no_clocks_read": True,
            "runtime": identity,
        }
    finally:
        if profiler:
            profiler.close()
        if env is not None and original_recv is not None:
            if saved_recv is MISSING:
                if "_recv" in env.__dict__:
                    delattr(env, "_recv")
            else:
                env._recv = saved_recv
        profiler = step = converted = action = original_recv = capture_recv = (
            saved_recv
        ) = None
        if env is not None:
            env.close()
        env = None
        gc.collect()
    audit["pool_released"] = pool_ref() is None
    audit["new_live_tids_after_cleanup"] = sorted(
        set(task_snapshot()) - baseline_tids, key=int
    )
    if not audit["pool_released"] or audit["new_live_tids_after_cleanup"]:
        raise AssertionError("validation retained a pool or its worker threads")
    return result, audit


def validate(args, np, envpool, helper):
    cases = [args.case] if args.case else list(CASES)
    rows = []
    for case in cases:
        reference = None
        audits = []
        for mode in MODES:
            result, audit = validation_rollout(
                args, np, envpool, helper, case, mode
            )
            if reference is None:
                reference = result
            else:
                assert_exact(np, reference, result, path=f"{case}/{mode}")
            audits.append(audit)
        rows.append({
            "case": case,
            "num_envs": CASES[case][0],
            "threads": CASES[case][1],
            "exact_public_outputs_native_arrays_and_metadata": True,
            "audits": audits,
        })
    return {
        "schema": "envpool-binding-phase-probe-v1",
        "kind": "untimed_validation",
        "variant": args.variant,
        "seed": 42,
        "steps_per_fresh_pool": 32,
        "all_checks_passed": True,
        "cases": rows,
        "note": "Fresh sequential matched-seed pools. Every public output leaf, native recv array, dtype, shape, byte pattern, dict key order, scalar, and pool metadata is exact across modes. Raw public adaptation is validation-only. Instrumentation uses the same hooks with clocks disabled. No performance evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=SOURCE_ROOT,
        type=Path,
        help="Source root containing benchmark/core_runtime/runtime.py (default: this repository)",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Descriptive runtime label; never used for selection",
    )
    parser.add_argument(
        "--envpool-root",
        type=Path,
        help="Root containing envpool/; omit to use this interpreter's installed runtime",
    )
    parser.add_argument(
        "--native-sha256",
        required=True,
        help="Expected SHA-256 of the loaded CartPole native module; checked before warmup or validation",
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--mode", choices=MODES, default="public")
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument(
        "--pin",
        action="store_true",
        help="Workers offset 1, then caller CPU 0; require verified actual masks",
    )
    parser.add_argument("--system-diagnostics", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--timing-authorized",
        action="store_true",
        help="Affirm explicitly approved, serialized timing",
    )
    args = parser.parse_args()
    if args.calls <= 0 or args.warmup < 0:
        parser.error("calls must be positive and warmup nonnegative")
    args.native_sha256 = args.native_sha256.lower()
    if len(args.native_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.native_sha256
    ):
        parser.error("--native-sha256 must contain 64 hexadecimal characters")
    if not args.validate_only and (not args.timing_authorized or not args.case):
        parser.error(
            "timing requires --timing-authorized and --case; coordinate before running"
        )
    if args.validate_only and args.pin:
        parser.error(
            "validation is unpinned; validate runtime equality separately from timing affinity"
        )
    if args.pin:
        required = {0} | set(range(1, CASES[args.case][1] + 1))
        if not required <= set(os.sched_getaffinity(0)):
            parser.error(
                "requested caller/worker CPUs exceed actual inherited affinity"
            )
    np, envpool, helper = load_selected_runtime(args)
    result = (
        validate(args, np, envpool, helper)
        if args.validate_only
        else run_sample(args, np, envpool, helper)
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
