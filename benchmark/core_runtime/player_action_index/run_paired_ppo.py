# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Initial fixed four-block PPO trial with duplicate A/A same-binary labels."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PATTERNS = ("acdb", "cabd", "bdca", "dbac")


def main():
    """Keep all 16 fresh-process runs, refusing to overwrite earlier evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    python = os.path.abspath(os.path.expanduser(args.python))
    control = args.baseline_root.resolve()
    candidate = args.candidate_root.resolve()
    roots = {label: control if label in "ab" else candidate for label in "abcd"}
    hashes = {}
    for label, root in roots.items():
        native = root / "envpool/classic_control/classic_control_envpool.so"
        hashes[label] = hashlib.sha256(native.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "reason": "Initial predeclared PPO trial with same-binary A/A controls",
        "blocks": 4,
        "samples": 16,
        "orders": list(PATTERNS),
        "roots": {key: str(value) for key, value in roots.items()},
        "native_hashes": hashes,
        "budget": "Unchanged bench_ppo.py: 256000 responses, 100 updates, 8000 optimizer steps; two discarded priming updates",
        "scope": "One initial predeclared trial; retain all 16 full-budget samples, no adaptive stopping or source changes",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    script = Path(__file__).resolve().parents[1] / "experiments/bench_ppo.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    rows = []
    with (output / "samples.jsonl").open("x", buffering=1) as stream:
        for block, pattern in enumerate(PATTERNS):
            for slot, label in enumerate(pattern):
                name = f"block-{block}-slot-{slot}-{label}"
                target = output / (name + ".json")
                started = time.time()
                result = subprocess.run(
                    [
                        python,
                        str(script),
                        "--package",
                        str(roots[label] / "envpool"),
                        "--label",
                        "sample",
                        "--output",
                        str(target),
                        "--native-sha256",
                        hashes[label],
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
                row = {
                    "block": block,
                    "slot": slot,
                    "variant": label,
                    "order": list(pattern),
                    "started_unix": started,
                    "data": data,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(
                    f"block={block} slot={slot} {label}: {data['timing']['training_seconds']:.3f}s",
                    flush=True,
                )
    if len({row["data"]["semantic_sha256"] for row in rows}) != 1:
        raise AssertionError("PPO semantic fingerprints differ")
    if (
        len({json.dumps(row["data"]["config"], sort_keys=True) for row in rows})
        != 1
    ):
        raise AssertionError("PPO configurations differ")
    for label, root in roots.items():
        native = root / "envpool/classic_control/classic_control_envpool.so"
        if hashlib.sha256(native.read_bytes()).hexdigest() != hashes[label]:
            raise AssertionError("Native binary changed during confirmation")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "samples": len(rows),
                "semantic_parity": True,
                "frozen_hashes_match": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
