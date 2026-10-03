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
"""Compare original/candidate EnvPool outputs byte-for-byte in separate processes.

Example:
  python compare_families.py --candidate-runtime /path/to/runtime --output results

The current Python's installed envpool is the baseline unless --original-runtime
is supplied. Per-environment action streams depend only on seed, ID, and sequence
index. Async arrival order is normalized by (environment ID, per-ID output index).
Every observation, reward, flag, and nested info array is compared with dtype,
shape, and exact bytes. Earlier arrays are retained across later calls and close.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

TASKS = [
    ("Catch-v0", {}),
    ("FrozenLake-v1", {}),
    ("FrozenLake8x8-v1", {}),
    ("Taxi-v3", {}),
    ("Taxi-v4", {}),
    ("Taxi-v4", {"is_rainy": True, "rainy_probability": 0.5}),
    (
        "Taxi-v4",
        {
            "is_rainy": True,
            "rainy_probability": 0.5,
            "fickle_passenger": True,
            "fickle_probability": 0.7,
        },
    ),
    ("NChain-v0", {}),
    ("CliffWalking-v0", {}),
    ("CliffWalkingSlippery-v1", {}),
    ("Blackjack-v1", {}),
    ("MiniGrid-Empty-5x5-v0", {}),
    ("MiniGrid-Empty-Random-6x6-v0", {}),
    ("MiniGrid-DoorKey-8x8-v0", {}),
    ("MiniGrid-Dynamic-Obstacles-Random-6x6-v0", {}),
    ("MiniGrid-LavaCrossingS9N2-v0", {}),
    ("MiniGrid-MemoryS9-v0", {}),
    ("MiniGrid-MultiRoom-N2-S4-v0", {}),
    ("BabyAI-GoToRedBallGrey-v0", {}),
]
CONFIGS = [
    {"mode": "sync", "num_envs": 16, "batch_size": 16, "num_threads": 1},
    {"mode": "sync", "num_envs": 32, "batch_size": 32, "num_threads": 4},
    {"mode": "async", "num_envs": 32, "batch_size": 8, "num_threads": 4},
]


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=""):
    """Flatten numeric public outputs into named arrays."""
    import numpy as np

    if isinstance(value, dict):
        return {
            k2: v2
            for k in sorted(value)
            for k2, v2 in flatten(value[k], prefix + "/" + str(k)).items()
        }
    if isinstance(value, (tuple, list)):
        return {
            k2: v2
            for i, v in enumerate(value)
            for k2, v2 in flatten(v, prefix + "/" + str(i)).items()
        }
    array = np.asarray(value)
    assert not array.dtype.hasobject, (
        prefix,
        "object arrays need a value serializer",
    )
    return {prefix: array}


def same(a, b) -> bool:
    """Check exact dtype, shape and byte equality."""
    return (
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    )


def semantic(result):
    """Normalize reset and step output tuples."""
    if len(result) == 2:
        return {"obs": result[0], "info": result[1]}
    return dict(
        zip(
            ("obs", "reward", "terminated", "truncated", "info"),
            result,
            strict=True,
        )
    )


def actions(env, ids, steps, seed):
    """Choose actions from per-environment deterministic counters."""
    import numpy as np

    # Pseudorandom-looking integer sequence with no dependency on arrival order.
    x = (ids.astype(np.uint64) + 1) * np.uint64(0x9E3779B1)
    x = x ^ ((steps.astype(np.uint64) + seed) * np.uint64(0x85EBCA77))
    x ^= x >> np.uint64(13)
    return (x % env.action_space.n).astype(env.action_space.dtype)


def run_worker(args):
    """Record one isolated runtime with all configured cases."""
    if args.runtime:
        sys.path.insert(0, str(Path(args.runtime).resolve()))
    import numpy as np

    import envpool

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    evidence = {
        "variant": args.worker,
        "envpool_module": envpool.__file__,
        "native_modules": {},
        "steps": args.steps,
        "cases": [],
    }
    for family in ("toy_text", "minigrid"):
        module = importlib.import_module(f"envpool.{family}.{family}_envpool")
        evidence["native_modules"][family] = {
            "path": module.__file__,
            "sha256": sha256(Path(module.__file__)),
        }
    for task_i, (task, extra) in enumerate(TASKS):
        for config in CONFIGS:
            tag = f"{task_i:02}_{task}_{config['mode']}_n{config['num_envs']}_b{config['batch_size']}_t{config['num_threads']}"
            kw = {k: v for k, v in config.items() if k != "mode"}
            seed = 71
            env = envpool.make_gymnasium(
                task, seed=seed, max_episode_steps=43, **kw, **extra
            )
            n = config["num_envs"]
            counts = np.zeros(n, dtype=np.int64)
            streams = [[] for _ in range(n)]
            held = []
            retained_checks = 0

            def retain(result, held=held):
                if len(held) < 8:
                    held.append((
                        result,
                        {
                            k: v.copy()
                            for k, v in flatten(semantic(result)).items()
                        },
                    ))

            def verify_held(held=held, task=task, config=config):
                nonlocal retained_checks
                for old, saved in held:
                    current = flatten(semantic(old))
                    assert current.keys() == saved.keys()
                    for k in saved:
                        assert same(current[k], saved[k]), (
                            "retained output changed",
                            task,
                            config,
                            k,
                        )
                        retained_checks += 1

            def record(
                result, tag=tag, counts=counts, streams=streams, retain=retain
            ):
                flat = flatten(semantic(result))
                ids = result[-1]["env_id"]
                for k, v in flat.items():
                    assert v.ndim and v.shape[0] == len(ids), (tag, k, v.shape)
                for row, env_id in enumerate(ids):
                    if counts[env_id] < args.steps:
                        streams[env_id].append({
                            k: v[row].copy() for k, v in flat.items()
                        })
                    counts[env_id] += 1
                retain(result)
                return ids

            arrays = {}
            if config["mode"] == "sync":
                result = env.reset()
                arrays.update({
                    "reset" + k: v.copy()
                    for k, v in flatten(semantic(result)).items()
                })
                retain(result)
                ids = np.arange(n, dtype=np.int32)
                for step in range(args.steps):
                    result = env.step(actions(env, ids, counts[ids], seed))
                    assert np.array_equal(result[-1]["env_id"], ids)
                    record(result)
                    if step % 47 == 0:
                        verify_held()
            else:
                env.async_reset()
                calls = 0
                while counts.min() < args.steps:
                    result = env.recv()
                    ids = record(result)
                    if counts.min() >= args.steps:
                        break
                    env.send(actions(env, ids, counts[ids] - 1, seed), ids)
                    calls += 1
                    if calls % 47 == 0:
                        verify_held()
                assert calls < args.steps * n * 4
            keys = streams[0][0].keys()
            for k in keys:
                arrays["stream" + k] = np.stack([
                    np.stack([row[k] for row in stream]) for stream in streams
                ])
            env.close()
            del env, result
            gc.collect()
            verify_held()
            path = out / (tag + ".npz")
            np.savez_compressed(path, **arrays)
            record_data = {
                "tag": tag,
                "task_id": task,
                "options": extra,
                "config": config,
                "seed": seed,
                "max_episode_steps": 43,
                "per_environment_outputs": args.steps,
                "environment_outputs": n * args.steps,
                "additional_sync_reset_outputs": n
                if config["mode"] == "sync"
                else 0,
                "arrays": len(arrays),
                "bytes": sum(v.nbytes for v in arrays.values()),
                "retained_batches": len(held),
                "retained_array_checks": retained_checks,
                "passed": True,
            }
            evidence["cases"].append(record_data)
            print(json.dumps(record_data), flush=True)
            (out / "report.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
    # Partial/reset/shuffled synchronous IDs, including single-environment writes.
    for task in ("Taxi-v4", "MiniGrid-DoorKey-8x8-v0"):
        env = envpool.make_gymnasium(
            task,
            num_envs=16,
            num_threads=4,
            seed=91,
            max_episode_steps=31,
            **(
                {"render_mode": "rgb_array"}
                if task.startswith("MiniGrid")
                else {}
            ),
        )
        arrays, held = {}, []
        for phase, ids in enumerate((
            np.array([7, 1, 15, 3], np.int32),
            np.array([8], np.int32),
            np.arange(15, -1, -1, dtype=np.int32),
        )):
            result = env.reset(ids)
            assert np.array_equal(result[-1]["env_id"], ids)
            held.append((
                result,
                {k: v.copy() for k, v in flatten(semantic(result)).items()},
            ))
            for k, v in flatten(semantic(result)).items():
                arrays[f"{phase}/reset" + k] = v.copy()
            for step in range(96):
                result = env.step(
                    actions(env, ids, np.full(len(ids), step), 91), ids
                )
                assert np.array_equal(result[-1]["env_id"], ids)
                for k, v in flatten(semantic(result)).items():
                    arrays[f"{phase}/{step}" + k] = v.copy()
                if task.startswith("MiniGrid") and step in (0, 31, 63, 95):
                    # Render uses the targeted ID order and is compared exactly.
                    arrays[f"{phase}/{step}/render"] = env.render(
                        env_ids=ids
                    ).copy()
        env.close()
        del env, result
        gc.collect()
        for old, saved in held:
            for k, v in flatten(semantic(old)).items():
                assert same(v, saved[k]), ("partial retained", task, k)
        tag = task + "_partial_shuffled_render"
        np.savez_compressed(out / (tag + ".npz"), **arrays)
        evidence["cases"].append({
            "tag": tag,
            "task_id": task,
            "config": {
                "mode": "partial_sync",
                "num_envs": 16,
                "num_threads": 4,
            },
            "seed": 91,
            "max_episode_steps": 31,
            "subset_ids": [[7, 1, 15, 3], [8], list(range(15, -1, -1))],
            "steps_per_subset": 96,
            "environment_outputs": (4 + 1 + 16) * 97,
            "retained_batches": 3,
            "arrays": len(arrays),
            "bytes": sum(v.nbytes for v in arrays.values()),
            "passed": True,
        })
        print(tag, "passed", flush=True)
    (out / "report.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    """Run the command-line build or comparison workflow."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-runtime", type=Path)
    p.add_argument("--original-runtime", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--compare-only", action="store_true")
    p.add_argument("--worker", choices=("original", "candidate"))
    p.add_argument("--runtime", type=Path)
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.worker:
        run_worker(a)
        return
    if not a.candidate_runtime and not a.compare_only:
        p.error("--candidate-runtime is required")
    a.output.mkdir(parents=True, exist_ok=True)
    for variant, runtime in (
        ()
        if a.compare_only
        else (
            ("original", a.original_runtime),
            ("candidate", a.candidate_runtime),
        )
    ):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            variant,
            "--output",
            str(a.output / variant),
            "--steps",
            str(a.steps),
        ]
        if runtime:
            cmd.extend(("--runtime", str(runtime.resolve())))
        with (a.output / (variant + ".log")).open("w") as log:
            subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        print(variant, "rollouts passed", flush=True)
    import numpy as np

    original_files = sorted((a.output / "original").glob("*.npz"))
    candidate_files = sorted((a.output / "candidate").glob("*.npz"))
    assert original_files, "No rollout files found"
    assert [p.name for p in original_files] == [p.name for p in candidate_files]
    report = {
        "passed": True,
        "comparison": "identical keys, dtype, shape, and ndarray.tobytes(); async keyed by env_id and per-ID output index",
        "cases": [],
        "total_arrays": 0,
        "total_bytes": 0,
        "mismatches": [],
    }
    for original, candidate in zip(
        original_files, candidate_files, strict=True
    ):
        left, right = np.load(original), np.load(candidate)
        assert set(left.files) == set(right.files), original.name
        digest = hashlib.sha256()
        size = 0
        for key in sorted(left.files):
            x, y = left[key], right[key]
            if not same(x, y):
                report["mismatches"].append({
                    "case": original.stem,
                    "field": key,
                    "original_shape": x.shape,
                    "candidate_shape": y.shape,
                })
            digest.update(key.encode())
            digest.update(str((x.dtype.str, x.shape)).encode())
            digest.update(x.tobytes())
            size += x.nbytes
        report["cases"].append({
            "tag": original.stem,
            "arrays": len(left.files),
            "bytes": size,
            "terminated_outputs": int(left["stream/terminated"].sum())
            if "stream/terminated" in left.files
            else None,
            "truncated_outputs": int(left["stream/truncated"].sum())
            if "stream/truncated" in left.files
            else None,
            "original_content_sha256": digest.hexdigest(),
            "passed": not any(
                m["case"] == original.stem for m in report["mismatches"]
            ),
        })
        report["total_arrays"] += len(left.files)
        report["total_bytes"] += size
    report["passed"] = not report["mismatches"]
    (a.output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "cases"}),
        flush=True,
    )
    assert report["passed"], "Byte comparison failed; inspect comparison.json"


if __name__ == "__main__":
    main()
