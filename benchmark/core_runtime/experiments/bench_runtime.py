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
"""Measure one public-API throughput sample in the selected Python process."""

import argparse
import hashlib
import importlib.machinery
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import VARIANTS, load_runtime, positive_int


def runtime_identity(env):
    """Fingerprint the actual native pool and imported wrappers after timing."""
    names = {
        "envpool.python.envpool",
        "envpool.python.gymnasium_envpool",
        "envpool.python.protocol",
    }
    for cls in type(env).__mro__:
        module = sys.modules.get(cls.__module__)
        path = getattr(module, "__file__", "") or ""
        if any(
            path.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES
        ):
            names.add(cls.__module__)
    if len(names) == 3:
        raise RuntimeError("Could not identify the native pool module")
    records = {}
    for name in sorted(names):
        path = Path(sys.modules[name].__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[name] = {"path": str(path), "sha256": digest}
    return records


def scheduler_stats():
    """Read Linux per-thread runnable-time counters outside the timed loop."""
    rows = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            rows[task.name] = tuple(
                map(int, (task / "schedstat").read_text().split())
            )
        except (FileNotFoundError, PermissionError):
            continue
    return rows


def main() -> None:
    """Time preallocated zero actions after an untimed warmup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--envpool-root", help="Parent of the envpool package")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--n", type=positive_int, default=256)
    parser.add_argument("--batch", type=positive_int)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--caller-cpu", type=int, default=0)
    parser.add_argument("--worker-offset", type=int, default=1)
    args = parser.parse_args()
    batch = args.batch or args.n
    if batch > args.n or args.warmup < 0 or args.rep < 0:
        parser.error("require batch <= n, warmup >= 0, and rep >= 0")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.pin:
        if not sys.platform.startswith("linux"):
            parser.error("--pin requires Linux CPU affinity support")
        required = {args.caller_cpu} | set(
            range(args.worker_offset, args.worker_offset + args.threads)
        )
        if not required <= set(os.sched_getaffinity(0)):
            parser.error(
                "requested caller/worker CPUs are outside CPU affinity"
            )

    np, envpool = load_runtime(args.envpool_root)
    async_mode = batch != args.n
    env = envpool.make_gymnasium(
        args.env,
        num_envs=args.n,
        batch_size=batch,
        num_threads=args.threads,
        seed=args.seed,
        thread_affinity_offset=args.worker_offset if args.pin else -1,
    )
    action = np.zeros(
        (batch,) + env.action_space.shape, dtype=env.action_space.dtype
    )
    if async_mode:
        env.async_reset()
        ids = env.recv()[-1]["env_id"]

        def step():
            nonlocal ids
            env.send(action, ids)
            ids = env.recv()[-1]["env_id"]

    else:
        env.reset()

        def step():
            env.step(action)

    # Set caller affinity after creating workers, just as in the recorded runs.
    if args.pin:
        os.sched_setaffinity(0, {args.caller_cpu})
    for _ in range(args.warmup):
        step()
    sched_before = scheduler_stats()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    cpu = time.process_time()
    count = 0
    while True:
        for _ in range(128):
            step()
        count += 128
        elapsed = time.perf_counter() - start
        if elapsed >= args.seconds:
            break
    cpu_seconds = time.process_time() - cpu
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    sched_after = scheduler_stats()
    common = sched_before.keys() & sched_after.keys()
    sched_delta = [
        sum(sched_after[t][i] - sched_before[t][i] for t in common)
        for i in range(3)
    ]
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_kb = peak_rss / 1024 if sys.platform == "darwin" else peak_rss
    except ImportError:
        rss_kb = None
    identity = runtime_identity(env)
    print(
        json.dumps({
            "runtime_modules": identity,
            "runtime_hashes": {
                name: value["sha256"] for name, value in identity.items()
            },
            "scheduler_run_seconds": sched_delta[0] / 1e9,
            "scheduler_wait_seconds": sched_delta[1] / 1e9,
            "scheduler_timeslices": sched_delta[2],
            "scheduler_threads": len(common),
            "scheduler_thread_set_stable": sched_before.keys()
            == sched_after.keys(),
            "voluntary_context_switches": usage_after.ru_nvcsw
            - usage_before.ru_nvcsw,
            "involuntary_context_switches": usage_after.ru_nivcsw
            - usage_before.ru_nivcsw,
            "variant": args.variant,
            "env": args.env,
            "num_envs": args.n,
            "batch_size": batch,
            "threads": args.threads,
            "async": async_mode,
            "rep": args.rep,
            "pin": args.pin,
            "seconds": elapsed,
            "cpu_seconds": cpu_seconds,
            "vector_calls": count,
            "env_steps": count * batch,
            "env_steps_per_s": count * batch / elapsed,
            "vector_calls_per_s": count / elapsed,
            "us_per_call": elapsed / count * 1e6,
            "rss_kb": rss_kb,
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "warmup": args.warmup,
            "requested_seconds": args.seconds,
            "seed": args.seed,
            "worker_affinity_offset": args.worker_offset if args.pin else -1,
            "runtime_selection": "explicit-root"
            if args.envpool_root
            else "interpreter",
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
