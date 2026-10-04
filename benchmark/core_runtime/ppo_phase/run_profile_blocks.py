# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Run one fixed 32-process PPO attribution trial, including genuine A/A pairs."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

ORDERS = ("aceghfdb", "dfhbagec", "egacdbhf", "hbdfecag")
CONDITIONS = {
    "a": ("original", "plain"),
    "b": ("original", "plain"),
    "c": ("original", "profile"),
    "d": ("original", "profile"),
    "e": ("retained", "plain"),
    "f": ("retained", "plain"),
    "g": ("retained", "profile"),
    "h": ("retained", "profile"),
}


def sha256(path):
    """Hash supplied local files, without importing any runtime."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_orders():
    """Each condition occupies every slot once; every label is early twice."""
    for order in ORDERS:
        if len(order) != 8 or set(order) != set(CONDITIONS):
            raise ValueError("invalid eight-label block")
        for first, second in ("ab", "cd", "ef", "gh"):
            if order.index(first) + order.index(second) != 7:
                raise ValueError("condition replicas must bracket block center")
    for pair in ("ab", "cd", "ef", "gh"):
        slots = Counter(
            order.index(label) for order in ORDERS for label in pair
        )
        if slots != Counter(range(8)):
            raise ValueError("condition positions are not balanced")
    for label in CONDITIONS:
        if sum(order.index(label) < 4 for order in ORDERS) != 2:
            raise ValueError("label early/late positions are not balanced")


def validate_result(data, condition, script_hash):
    """Preserve the exact reference work, identities and observation distinction."""
    cfg = data["config"]
    if any(
        cfg[key] != value
        for key, value in {
            "training_num": 20,
            "num_threads": 2,
            "updates": 100,
            "steps_per_collect": 2560,
            "repeat_per_collect": 2,
            "batch_size": 64,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        }.items()
    ):
        raise ValueError("reference PPO configuration changed")
    if any(
        data["budget"][key] != value
        for key, value in {
            "updates": 100,
            "env_steps": 256000,
            "optimizer_steps": 8000,
        }.items()
    ):
        raise ValueError("reference PPO budget changed")
    provenance = data["provenance"]
    if (
        provenance["native_sha256"] != condition["native_sha256"]
        or not provenance["expected_native_sha256_verified"]
        or provenance["script_sha256"] != script_hash
    ):
        raise ValueError("runtime or script identity mismatch")
    if ("phase_profile" in data) != (condition["mode"] == "profile"):
        raise ValueError("profile/plain observation mode mismatch")
    if data["measurement"]["warmup_updates"] != 2:
        raise ValueError("priming budget changed")


def main():
    """Refuse old outputs; save the complete plan before starting any trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--original-package", required=True, type=Path)
    parser.add_argument("--retained-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_orders()
    python = os.path.abspath(os.path.expanduser(args.python))
    folder = Path(__file__).resolve().parent
    scripts = {
        "plain": folder.parent / "experiments/bench_ppo.py",
        "profile": folder / "profile_ppo.py",
    }
    packages = {
        "original": args.original_package.resolve(),
        "retained": args.retained_package.resolve(),
    }
    conditions = {}
    for label, (runtime, mode) in CONDITIONS.items():
        package = packages[runtime]
        if not (package / "__init__.py").is_file():
            raise ValueError("package lacks __init__.py")
        native = package / "classic_control/classic_control_envpool.so"
        conditions[label] = {
            "runtime": runtime,
            "mode": mode,
            "package": str(package),
            "native_sha256": sha256(native),
            "script": str(scripts[mode]),
            "script_sha256": sha256(scripts[mode]),
        }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "purpose": "Attribute reference PPO wall time and quantify observer cost; no core changes",
        "blocks": 4,
        "samples": 32,
        "orders": list(ORDERS),
        "conditions": conditions,
        "python": python,
        "budget": {
            "updates": 100,
            "env_steps": 256000,
            "optimizer_steps": 8000,
        },
        "warmup_updates": 2,
        "counterfactual_scope": "EnvPool Python call time includes conversion and native waits; background work can affect other phases. Any Amdahl model holds the rest constant and is not a hard core bound.",
        "sampling": "One predeclared fixed trial, all runs retained, no adaptive stopping",
        "pairing": "Two replicas per condition per block; symmetric slots and condition/label position balance, not elimination of carryover or thermal noise",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    rows = []
    with (output / "samples.jsonl").open("x", buffering=1) as stream:
        for block, order in enumerate(ORDERS):
            for slot, label in enumerate(order):
                condition = conditions[label]
                name = f"block-{block}-slot-{slot}-{label}"
                target = output / (name + ".json")
                started = time.time()
                result = subprocess.run(
                    [
                        python,
                        condition["script"],
                        "--package",
                        condition["package"],
                        "--label",
                        "sample",
                        "--output",
                        str(target),
                        "--iterations",
                        "100",
                        "--warmup",
                        "2",
                        "--native-sha256",
                        condition["native_sha256"],
                    ],
                    cwd=output,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                (output / (name + ".log")).write_text(
                    result.stdout + result.stderr
                )
                if result.returncode:
                    raise RuntimeError(
                        f"PPO child failed: {name}: {result.returncode}"
                    )
                data = json.loads(target.read_text())
                validate_result(data, condition, condition["script_sha256"])
                row = {
                    "block": block,
                    "slot": slot,
                    "variant": label,
                    "runtime": condition["runtime"],
                    "mode": condition["mode"],
                    "order": list(order),
                    "started_unix": started,
                    "data": data,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(
                    f"{name} {condition['runtime']}/{condition['mode']} "
                    f"{data['timing']['training_seconds']:.3f}s",
                    flush=True,
                )
    if len({row["data"]["semantic_sha256"] for row in rows}) != 1:
        raise AssertionError("PPO semantic fingerprints differ")
    for condition in conditions.values():
        native = (
            Path(condition["package"])
            / "classic_control/classic_control_envpool.so"
        )
        if sha256(native) != condition["native_sha256"]:
            raise AssertionError("native binary changed during trial")
        if sha256(condition["script"]) != condition["script_sha256"]:
            raise AssertionError("measurement script changed during trial")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "samples": 32,
                "semantic_parity": True,
                "frozen_native_and_script_hashes_match": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
