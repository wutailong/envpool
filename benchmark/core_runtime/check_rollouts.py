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
"""Record and compare deterministic streams independently of async arrival order."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

from runtime import VARIANTS, load_runtime


def flatten(value, prefix=""):
    """Flatten nested Gymnasium output into named NumPy arrays."""
    import numpy as np

    if isinstance(value, dict):
        return {
            key: item
            for name in sorted(value)
            for key, item in flatten(value[name], prefix + "." + name).items()
        }
    if isinstance(value, tuple):
        return {
            key: item
            for index, child in enumerate(value)
            for key, item in flatten(child, prefix + "." + str(index)).items()
        }
    return {prefix: np.asarray(value)}


def actions(env, ids, steps):
    """Derive deterministic changing actions from environment ID and step count."""
    import numpy as np

    if hasattr(env.action_space, "n"):
        return ((ids + steps) % env.action_space.n).astype(
            env.action_space.dtype
        )
    return (
        np.sin(
            ids[:, None] * 0.7
            + steps[:, None] * 0.3
            + np.arange(env.action_space.shape[0]) * 0.2
        )
        * 0.4
    ).astype(env.action_space.dtype)


def arrays_equal(left, right) -> bool:
    """Compare shape, dtype, and exact bytes, including NaNs and signed zeros."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def record(args) -> None:
    """Capture full streams, retained outputs, and shuffled/subset reset paths."""
    np, envpool = load_runtime(args.envpool_root)
    outputs = {}
    report = []
    for name in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "HalfCheetah-v4"]:
        for count, batch, threads in [(20, 20, 1), (32, 32, 4), (32, 8, 4)]:
            env = envpool.make_gymnasium(
                name,
                num_envs=count,
                batch_size=batch,
                num_threads=threads,
                seed=71,
                max_episode_steps=79,
            )
            counts = np.zeros(count, dtype=np.int64)
            limit = 240
            streams = [[] for _ in range(count)]
            held = []
            env.async_reset()
            result = env.recv()
            while True:
                flat = flatten(result)
                ids = result[-1]["env_id"]
                rows = len(ids)
                for index, env_id in enumerate(ids):
                    if counts[env_id] < limit:
                        streams[env_id].append({
                            key: np.array(value[index], copy=True)
                            for key, value in flat.items()
                            if value.shape and value.shape[0] == rows
                        })
                    counts[env_id] += 1
                if len(held) < 5:
                    held.append((
                        result,
                        {k: v.copy() for k, v in flat.items()},
                    ))
                if counts.min() >= limit:
                    break
                env.send(actions(env, ids, counts[ids]), ids)
                if counts.min() % 31 == 0:
                    gc.collect()
                result = env.recv()
            del result, env
            gc.collect()
            for old, saved in held:
                for key, value in flatten(old).items():
                    if not arrays_equal(value, saved[key]):
                        raise AssertionError(("retained", name, key))
            tag = f"{name}_n{count}_b{batch}_t{threads}"
            for key in streams[0][0]:
                outputs[tag + key] = np.stack([
                    np.stack([row[key] for row in stream]) for stream in streams
                ])
            report.append({
                "config": tag,
                "env_outputs": count * limit,
                "retained_batches": len(held),
            })
            print(tag, "passed", flush=True)
    for name in ["CartPole-v1", "HalfCheetah-v4"]:
        env = envpool.make_gymnasium(name, num_envs=16, num_threads=4, seed=91)
        subsets = [
            np.array([7, 1, 15, 3], np.int32),
            np.array([8], np.int32),
            np.arange(15, -1, -1, dtype=np.int32),
        ]
        for phase, ids in enumerate(subsets):
            result = env.reset(ids)
            if not np.array_equal(result[-1]["env_id"], ids):
                raise AssertionError("reset did not retain requested ID order")
            for step in range(100):
                result = env.step(
                    actions(env, ids, np.full(len(ids), step)), ids
                )
                if not np.array_equal(result[-1]["env_id"], ids):
                    raise AssertionError(
                        "step did not retain requested ID order"
                    )
                for key, value in flatten(result).items():
                    outputs[f"{name}_partial{phase}_{step}" + key] = (
                        value.copy()
                    )
        del env
    # Exclusive creation avoids silently replacing a previous correctness run.
    with args.out.open("xb") as output:
        np.savez_compressed(output, **outputs)
    print(
        json.dumps({
            "variant": args.variant,
            "cases": report,
            "array_count": len(outputs),
            "sha256": hashlib.sha256(
                b"".join(
                    key.encode() + value.tobytes()
                    for key, value in sorted(outputs.items())
                )
            ).hexdigest(),
        }),
        flush=True,
    )


def compare(reference: Path, candidates: list[Path]) -> None:
    """Fail unless every candidate archive matches all reference arrays exactly."""
    import numpy as np

    with np.load(reference, allow_pickle=False) as expected:
        for candidate in candidates:
            with np.load(candidate, allow_pickle=False) as actual:
                if set(expected.files) != set(actual.files):
                    raise AssertionError("rollout array keys differ")
                mismatches = [
                    key
                    for key in expected.files
                    if not arrays_equal(expected[key], actual[key])
                ]
                if mismatches:
                    raise AssertionError(
                        f"{len(mismatches)} bitwise mismatches: {mismatches[:10]}"
                    )
                print(
                    json.dumps({"arrays": len(expected.files), "mismatches": 0})
                )


def main() -> None:
    """Select recording or local archive comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    recording = modes.add_parser("record", help="Create one build's archive")
    recording.add_argument("--variant", choices=VARIANTS, required=True)
    recording.add_argument(
        "--envpool-root", help="Parent of the envpool package"
    )
    recording.add_argument("--out", type=Path, required=True)
    comparison = modes.add_parser("compare", help="Compare exact array bytes")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "record":
        if args.out.exists():
            parser.error("--out already exists; select a fresh archive")
        record(args)
    else:
        compare(args.reference, args.candidates)


if __name__ == "__main__":
    main()
