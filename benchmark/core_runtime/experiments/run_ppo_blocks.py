# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Run serialized ABBA/BAAB full-budget PPO trials and check semantic parity."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


def main():
    """Use fresh child processes and require an entirely new evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--variant", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=2)
    args = parser.parse_args()
    roots = dict(item.split("=", 1) for item in args.variant)
    if len(args.variant) != 2 or len(roots) != 2 or args.blocks <= 0:
        parser.error("supply two distinct labels and a positive block count")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", label) for label in roots):
        parser.error("variant labels must be short alphanumeric identifiers")
    packages = {
        label: Path(root).expanduser().resolve() / "envpool"
        for label, root in roots.items()
    }
    hashes = {
        label: hashlib.sha256(
            (
                package / "classic_control/classic_control_envpool.so"
            ).read_bytes()
        ).hexdigest()
        for label, package in packages.items()
    }
    python = os.path.abspath(os.path.expanduser(args.python))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    labels = list(roots)
    rows = []
    script = Path(__file__).resolve().with_name("bench_ppo.py")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for block in range(args.blocks):
        order = (
            [labels[0], labels[1], labels[1], labels[0]]
            if block % 2 == 0
            else [labels[1], labels[0], labels[0], labels[1]]
        )
        for slot, label in enumerate(order):
            name = f"block-{block}-slot-{slot}-{label}"
            path = output / (name + ".json")
            cmd = [
                python,
                str(script),
                "--package",
                str(packages[label]),
                "--label",
                "sample",
                "--output",
                str(path),
                "--native-sha256",
                hashes[label],
            ]
            result = subprocess.run(
                cmd,
                env=env,
                cwd=output,
                capture_output=True,
                text=True,
                timeout=300,
            )
            (output / (name + ".log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"PPO child failed with exit {result.returncode}"
                )
            data = json.loads(path.read_text())
            rows.append({
                "block": block,
                "slot": slot,
                "variant": label,
                "order": order,
                "data": data,
            })
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
    with (output / "samples.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "semantic_parity": True,
        "blocks": args.blocks,
        "samples": len(rows),
        "variants": {},
    }
    for label in labels:
        values = [
            row["data"]["timing"]["training_seconds"]
            for row in rows
            if row["variant"] == label
        ]
        summary["variants"][label] = {
            "n": len(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
