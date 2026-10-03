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
"""Record exact Box2D Container output parity without large rollout archives.

Run each runtime in a separate process with the same Python/dependencies:
  python check_box2d.py record --variant baseline --envpool-root BASE --out a.json
  python check_box2d.py record --variant candidate --envpool-root NEXT --out b.json
  python check_box2d.py compare a.json b.json

Both Box2D extensions MUST be built with ENVPOOL_TEST: the public production
build omits the Container<float> diagnostic fields. No runtime option enables
them. The recorder fails rather than silently passing without these fields.

Each case performs one reset and 64 full-batch step calls, including next-step
autoresets, with two environments, two threads, seeds 0/42 and a 13-step limit.
All public outputs and input actions are retained as dtype/shape/SHA-256 records.
Object arrays are recursively serialized by value, never by pointer bytes.
First inner arrays from every Container field survive subsequent calls, close,
pool deletion and garbage collection, with unchanged contents checked at end.
These checks establish output/lifetime parity, not native leak freedom.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import load_runtime

SCHEMA = "envpool-box2d-container-parity-v1"
TASKS = {
    "CarRacing-v3": ("track", "road_poly", "road_color"),
    "BipedalWalker-v3": ("path4",),
}
SEEDS = (0, 42)
CONFIG = {
    "num_envs": 2,
    "batch_size": 2,
    "num_threads": 2,
    "max_episode_steps": 13,
}
STEPS = 64
ACTION_SEED_OFFSET = 1729


def json_bytes(value) -> bytes:
    """Encode JSON consistently, without platform-dependent whitespace."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: bytes) -> str:
    """Return a SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def canonicalize(np, value, counts=None):
    """Preserve every nested value, hashing contiguous numeric bytes exactly.

    Object-array hashes cover canonical child records because raw object bytes
    contain process-specific addresses. Numeric bytes preserve signed zeros and
    NaN payloads; neither floating-point JSON nor tolerances are used.
    """
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("output mappings must have string keys")
        return {
            "kind": "dict",
            "items": {
                key: canonicalize(np, value[key], counts)
                for key in sorted(value)
            },
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [canonicalize(np, item, counts) for item in value],
        }
    if value is None or isinstance(value, str):
        return {"kind": "literal", "value": value}
    if not isinstance(value, (np.ndarray, np.generic, bool, int, float)):
        raise TypeError(f"unsupported output type: {type(value).__name__}")
    array = np.asarray(value)
    result = {
        "kind": "array",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.dtype.fields is not None:
        # dtype.str alone describes structured arrays only as opaque voids.
        result["dtype_descr"] = json.loads(json_bytes(array.dtype.descr))
    if array.dtype.hasobject:
        if array.dtype != np.dtype(object):
            raise TypeError("structured object arrays are not supported")
        result["kind"] = "object_array"
        result["items"] = [
            canonicalize(np, item, counts) for item in array.flat
        ]
        result["sha256"] = digest(json_bytes(result["items"]))
        if counts is not None:
            counts["object_arrays"] += 1
            counts["object_elements"] += int(array.size)
    else:
        raw = np.ascontiguousarray(array).tobytes(order="C")
        result["sha256"] = digest(raw)
        result["nbytes"] = len(raw)
        if counts is not None:
            counts["numeric_arrays"] += 1
            counts["numeric_bytes"] += len(raw)
    return result


def semantic(result):
    """Name all fields of the public Gymnasium reset or step tuple."""
    if len(result) == 2:
        return {"obs": result[0], "info": result[1]}
    return dict(
        zip(
            ("obs", "reward", "terminated", "truncated", "info"),
            result,
            strict=True,
        )
    )


def check_info(np, info, task: str) -> int:
    """Require ordered full batches and real Container<float> inner arrays."""
    if not np.array_equal(info["env_id"], np.arange(CONFIG["num_envs"])):
        raise AssertionError(f"{task}: full-batch environment order changed")
    total = 0
    for key in TASKS[task]:
        if key not in info:
            raise RuntimeError(
                f"{task}: missing info.{key}; rebuild the Box2D extension "
                "with ENVPOOL_TEST enabled in both runtimes"
            )
        outer = info[key]
        if (
            not isinstance(outer, np.ndarray)
            or outer.dtype != np.dtype(object)
            or outer.shape != (CONFIG["num_envs"],)
        ):
            raise AssertionError(f"{task}: info.{key} is not a Container batch")
        for inner in outer.flat:
            if (
                not isinstance(inner, np.ndarray)
                or inner.dtype != np.dtype(np.float32)
                or inner.size == 0
            ):
                raise AssertionError(
                    f"{task}: info.{key} lacks a nonempty float32 inner array"
                )
            total += 1
    return total


def action_stream(np, space, seed: int):
    """Precompute changing finite actions from an explicit, independent RNG."""
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise AssertionError("Box2D action bounds must be finite")
    rng = np.random.Generator(np.random.PCG64(ACTION_SEED_OFFSET + seed))
    shape = (STEPS, CONFIG["num_envs"], *space.shape)
    actions = (low + rng.random(shape) * (high - low)).astype(space.dtype)
    if not np.all(np.isfinite(actions)) or not all(
        space.contains(row) for batch in actions for row in batch
    ):
        raise AssertionError("generated actions fall outside the action space")
    return actions


def record_case(np, envpool, task: str, seed: int):
    """Record a bounded rollout and verify independently retained payloads."""
    env = envpool.make_gymnasium(task, seed=seed, **CONFIG)
    counts = {
        "reset_calls": 1,
        "step_calls": STEPS,
        "records": STEPS + 1,
        "environment_responses": (STEPS + 1) * CONFIG["num_envs"],
        "numeric_arrays": 0,
        "numeric_bytes": 0,
        "object_arrays": 0,
        "object_elements": 0,
        "container_inner_arrays": 0,
    }
    held = []
    outer_refs = []
    records = []
    terminated_counts = np.zeros(CONFIG["num_envs"], dtype=np.int64)
    truncated_counts = np.zeros_like(terminated_counts)
    autoreset_counts = np.zeros_like(terminated_counts)
    previous_done = np.zeros(CONFIG["num_envs"], dtype=np.bool_)
    try:
        actions = action_stream(np, env.action_space, seed)
        action_spec = canonicalize(
            np,
            {
                "low": env.action_space.low,
                "high": env.action_space.high,
            },
        )
        result = semantic(env.reset())
        counts["container_inner_arrays"] += check_info(np, result["info"], task)
        for key in TASKS[task]:
            inner = result["info"][key].flat[0]
            held.append((f"info.{key}[0]", inner, canonicalize(np, inner)))
            outer_refs.append(weakref.ref(result["info"][key]))
        del inner
        records.append({
            "call": "reset",
            "index": 0,
            "output": canonicalize(np, result, counts),
        })
        for step, action in enumerate(actions, start=1):
            result = semantic(env.step(action))
            info = result["info"]
            counts["container_inner_arrays"] += check_info(np, info, task)
            elapsed = np.asarray(info["elapsed_step"])
            if elapsed.shape != previous_done.shape:
                raise AssertionError(f"{task}: unexpected elapsed_step shape")
            autoreset = elapsed == 0
            if not np.array_equal(autoreset, previous_done):
                raise AssertionError(
                    f"{task}: next-step autoreset was not observed"
                )
            terminated = np.asarray(result["terminated"])
            truncated = np.asarray(result["truncated"])
            if (
                terminated.shape != previous_done.shape
                or truncated.shape != previous_done.shape
            ):
                raise AssertionError(f"{task}: unexpected terminal flag shape")
            terminated_counts += terminated
            truncated_counts += truncated
            autoreset_counts += autoreset
            previous_done = terminated | truncated
            records.append({
                "call": "step",
                "index": step,
                "action": canonicalize(np, action),
                "output": canonicalize(np, result, counts),
            })
        if not np.all(autoreset_counts > 0) or not np.all(truncated_counts > 0):
            raise AssertionError(
                f"{task}: rollout did not cover truncation/autoreset"
            )
        # No full output or outer object array is retained by this point.
        del result, info, elapsed, terminated, truncated
    finally:
        env.close()
    del env
    gc.collect()
    if any(ref() is not None for ref in outer_refs):
        raise AssertionError(
            f"{task}: a retained payload's outer array is live"
        )
    retained = []
    for path, inner, before in held:
        after = canonicalize(np, inner)
        if after != before:
            raise AssertionError(
                f"{task}: retained {path} changed after close/GC"
            )
        retained.append({
            "path": path,
            "array": after,
            "unchanged": True,
            "outer_released": True,
        })
    counts.update({
        "terminated_by_env": terminated_counts.tolist(),
        "truncated_by_env": truncated_counts.tolist(),
        "autoresets_by_env": autoreset_counts.tolist(),
        "retained_inner_arrays": len(retained),
        "retained_checks_after_close_gc": len(retained),
    })
    return {
        "task": task,
        "seed": seed,
        "action_spec": action_spec,
        "counts": counts,
        "retained": retained,
        "records": records,
    }


def record(args) -> None:
    """Load exactly one explicit runtime and write a reproducible JSON record."""
    np, envpool = load_runtime(args.envpool_root)
    native = importlib.import_module("envpool.box2d.box2d_envpool")
    native_path = Path(native.__file__).resolve()
    root = Path(args.envpool_root).expanduser().resolve()
    if not native_path.is_relative_to(root / "envpool"):
        raise RuntimeError("Box2D native import did not use --envpool-root")
    cases = []
    for task in TASKS:
        for seed in SEEDS:
            case = record_case(np, envpool, task, seed)
            cases.append(case)
            print(
                json.dumps({"task": task, "seed": seed, **case["counts"]}),
                flush=True,
            )
    semantic_record = {
        "config": CONFIG,
        "steps": STEPS,
        "action_rng": "numpy.random.PCG64",
        "action_seed_offset": ACTION_SEED_OFFSET,
        "cases": cases,
    }
    report = {
        "schema": SCHEMA,
        "provenance": {
            "variant": args.variant,
            "envpool_root": str(root),
            "envpool_module": str(Path(envpool.__file__).resolve()),
            "native_module": str(native_path),
            "native_sha256": digest(native_path.read_bytes()),
            "numpy_version": np.__version__,
            "python_version": sys.version,
        },
        "semantic": semantic_record,
        "semantic_sha256": digest(json_bytes(semantic_record)),
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps({
            "variant": args.variant,
            "cases": len(cases),
            "records": sum(case["counts"]["records"] for case in cases),
            "semantic_sha256": report["semantic_sha256"],
        }),
        flush=True,
    )


def read_record(path: Path):
    """Reject malformed/incomplete archives before comparing their semantics."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema"] != SCHEMA:
        raise ValueError(f"{path}: incompatible record schema")
    value = report["semantic"]
    if digest(json_bytes(value)) != report["semantic_sha256"]:
        raise ValueError(f"{path}: semantic digest does not match content")
    if value["config"] != CONFIG or value["steps"] != STEPS:
        raise ValueError(f"{path}: unexpected rollout configuration")
    identities = [(case["task"], case["seed"]) for case in value["cases"]]
    if identities != [(task, seed) for task in TASKS for seed in SEEDS]:
        raise ValueError(f"{path}: missing, duplicate, or reordered cases")
    for case in value["cases"]:
        records = case["records"]
        counts = case["counts"]
        if (
            len(records) != STEPS + 1
            or counts["records"] != len(records)
            or counts["step_calls"] != STEPS
            or counts["reset_calls"] != 1
            or counts["environment_responses"]
            != (STEPS + 1) * CONFIG["num_envs"]
            or counts["container_inner_arrays"]
            != (STEPS + 1) * CONFIG["num_envs"] * len(TASKS[case["task"]])
            or counts["retained_inner_arrays"] != len(TASKS[case["task"]])
            or counts["retained_checks_after_close_gc"] != len(case["retained"])
            or [(row["call"], row["index"]) for row in records]
            != [("reset", 0)] + [("step", step) for step in range(1, STEPS + 1)]
        ):
            raise ValueError(
                f"{path}: incomplete records or inconsistent counts"
            )
    return value


def first_difference(left, right, path="semantic") -> str | None:
    """Locate one exact mismatch without dumping a large rollout."""
    if type(left) is not type(right):
        return path + " (type)"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + " (keys)"
        for key in sorted(left):
            mismatch = first_difference(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + " (length)"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            mismatch = first_difference(a, b, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
    elif left != right:
        return path
    return None


def compare(reference: Path, candidates: list[Path]) -> None:
    """Compare every semantic record and count, excluding build provenance."""
    expected = read_record(reference)
    for candidate in candidates:
        actual = read_record(candidate)
        mismatch = first_difference(expected, actual)
        if mismatch is not None:
            raise AssertionError(f"{candidate}: exact mismatch at {mismatch}")
        print(
            json.dumps({
                "reference": str(reference),
                "candidate": str(candidate),
                "cases": len(expected["cases"]),
                "records": sum(
                    case["counts"]["records"] for case in expected["cases"]
                ),
                "mismatches": 0,
                "semantic_sha256": digest(json_bytes(expected)),
            }),
            flush=True,
        )


def main() -> None:
    """Select explicit-runtime recording or dependency-free comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    recording = modes.add_parser("record", help="Record one isolated runtime")
    recording.add_argument("--variant", required=True, help="Build label only")
    recording.add_argument(
        "--envpool-root", required=True, help="Parent of the envpool package"
    )
    recording.add_argument("--out", type=Path, required=True)
    comparison = modes.add_parser("compare", help="Compare exact JSON records")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "record":
        if args.out.exists():
            parser.error("--out already exists; select a fresh JSON archive")
        record(args)
    else:
        compare(args.reference, args.candidates)


if __name__ == "__main__":
    main()
