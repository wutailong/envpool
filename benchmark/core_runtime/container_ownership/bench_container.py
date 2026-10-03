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
"""Measure normal Dummy Container conversion, including typed-backing cleanup.

This uses the real native Dummy pool and its ordinary Python _recv conversion;
it does not bypass conversion, retain historical outputs, or force collection.
The baseline already releases payloads on this successful conversion path.
This is an overhead comparison, not a leak test or an expected speedup.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import VARIANTS, load_runtime

NUM_ENVS = 64
THREADS = 4
SEED = 42
WARMUP = 400
SECONDS = 3.0


def scheduler_stats():
    """Read Linux per-thread runnable counters outside the timed loop."""
    rows = {}
    task_root = Path("/proc/self/task")
    if not task_root.is_dir():
        raise RuntimeError("this scheduler-instrumented benchmark needs Linux")
    for task in task_root.iterdir():
        try:
            rows[task.name] = tuple(
                map(int, (task / "schedstat").read_text().split())
            )
        except (FileNotFoundError, PermissionError):
            continue
    if not rows:
        raise RuntimeError("no readable /proc/self/task/*/schedstat counters")
    return rows


def sha256(path):
    """Fingerprint the imported native module outside the measured interval."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class DummyLoop:
    """Route actions by actual returned IDs while discarding old observations."""

    def __init__(self, env, list_action):
        """Cache field indices and reuse the fixed-size list_action buffer."""
        self.env = env
        self.state_keys = tuple(env._state_keys)
        action_keys = tuple(env._action_keys)
        sources = {
            "env_id": "info:env_id",
            "players.env_id": "info:players.env_id",
            "players.action": "info:players.id",
            "players.id": "info:players.id",
        }
        if set(action_keys) != {*sources, "list_action"}:
            raise ValueError("unexpected Dummy action schema")
        self.mapping = tuple(
            (action_keys.index(target), self.state_keys.index(source))
            for target, source in sources.items()
        )
        self.env_id_index = self.state_keys.index("info:env_id")
        self.action = [None] * len(action_keys)
        self.action[action_keys.index("list_action")] = list_action
        # The caller has already requested reset. _recv executes the production
        # Container-to-object-array converter, including nested NumPy arrays.
        self.state = env._recv()

    def step(self):
        """Return environment response count, not player count or vector calls."""
        for target, source in self.mapping:
            self.action[target] = self.state[source]
        self.env._send(self.action)
        self.state = self.env._recv()
        return len(self.state[self.env_id_index])

    def validate(self, np, max_num_players):
        """Check synchronous response shape and real conversion outside timing."""
        state = dict(zip(self.state_keys, self.state, strict=True))
        env_ids = state["info:env_id"]
        player_env_ids = state["info:players.env_id"]
        player_ids = state["info:players.id"]
        dynamic = state["obs:dyn"]
        np.testing.assert_array_equal(np.sort(env_ids), np.arange(NUM_ENVS))
        if not (
            len(player_env_ids) == len(player_ids) == len(dynamic)
            and NUM_ENVS <= len(player_ids) <= NUM_ENVS * max_num_players
            and dynamic.dtype == np.dtype(object)
            and dynamic.ndim == 1
        ):
            raise ValueError("invalid Dummy player/Container response")
        if not np.all(np.isin(player_env_ids, env_ids)):
            raise ValueError("player-to-environment IDs do not match responses")
        for env_id, value in zip(player_env_ids, dynamic, strict=True):
            if not isinstance(value, np.ndarray) or value.shape != (
                int(env_id) + 1,
                10,
            ):
                raise ValueError("Dummy Container was not converted normally")


def main():
    """Time one synchronous N=B=64, T=4 sample from an explicit runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--envpool-root", required=True)
    parser.add_argument(
        "--max-num-players", type=int, choices=(1, 4), required=True
    )
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 0 or args.rep < 0:
        parser.error("require warmup >= 0 and rep >= 0")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    np, _ = load_runtime(args.envpool_root)
    root = Path(args.envpool_root).expanduser().resolve()
    module = importlib.import_module("envpool.dummy.dummy_envpool")
    selected_binary = Path(module.__file__).absolute()
    if not selected_binary.is_relative_to(root / "envpool"):
        raise RuntimeError("Dummy import did not use --envpool-root")
    # Frozen runtimes may share native files through read-only symlinks.
    binary = selected_binary.resolve()
    binary_hash = sha256(binary)
    conf = dict(
        zip(
            module._DummyEnvSpec._config_keys,
            module._DummyEnvSpec._default_config_values,
            strict=True,
        )
    )
    conf.update(
        num_envs=NUM_ENVS,
        batch_size=NUM_ENVS,
        num_threads=THREADS,
        max_num_players=args.max_num_players,
        seed=SEED,
        thread_affinity_offset=-1,
        state_num=10,
        action_num=6,
    )
    spec = module._DummyEnvSpec(
        tuple(conf[key] for key in module._DummyEnvSpec._config_keys)
    )
    env = module._DummyEnvPool(spec)
    list_action = np.zeros((NUM_ENVS, 6), dtype=np.float64)
    env._reset(np.arange(NUM_ENVS, dtype=np.int32))
    loop = DummyLoop(env, list_action)
    loop.validate(np, args.max_num_players)
    for _ in range(args.warmup):
        loop.step()
    loop.validate(np, args.max_num_players)

    sched_before = scheduler_stats()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    cpu = time.process_time()
    calls = 0
    responses = 0
    step = loop.step
    while True:
        for _ in range(128):
            responses += step()
        calls += 128
        elapsed = time.perf_counter() - start
        if elapsed >= args.seconds:
            break
    cpu_seconds = time.process_time() - cpu
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    sched_after = scheduler_stats()

    if responses != calls * NUM_ENVS:
        raise RuntimeError("synchronous response count differs from N=B=64")
    loop.validate(np, args.max_num_players)
    if sha256(binary) != binary_hash:
        raise RuntimeError("imported Dummy binary changed during measurement")
    common = sched_before.keys() & sched_after.keys()
    sched_delta = [
        sum(sched_after[t][i] - sched_before[t][i] for t in common)
        for i in range(3)
    ]
    print(
        json.dumps({
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
            "env": f"DummyPlayers{args.max_num_players}",
            "num_envs": NUM_ENVS,
            "batch_size": NUM_ENVS,
            "threads": THREADS,
            "max_num_players": args.max_num_players,
            "state_num": conf["state_num"],
            "action_num": conf["action_num"],
            "async": False,
            "rep": args.rep,
            "pin": False,
            "seconds": elapsed,
            "cpu_seconds": cpu_seconds,
            "vector_calls": calls,
            "env_steps": responses,
            "environment_responses": responses,
            "env_steps_per_s": responses / elapsed,
            "vector_calls_per_s": calls / elapsed,
            "us_per_call": elapsed / calls * 1e6,
            "affinity": sorted(os.sched_getaffinity(0)),
            "warmup": args.warmup,
            "requested_seconds": args.seconds,
            "seed": SEED,
            "worker_affinity_offset": -1,
            "runtime_selection": "explicit-root",
            "runtime_root": str(root),
            "binary_path": str(binary),
            "binary_sha256": binary_hash,
            "api": "_DummyEnvPool._send/_recv (normal Python conversion)",
            "response_unit": "environment response, including auto-reset",
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
