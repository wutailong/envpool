# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Summarize the initial fixed sixteen-trial PPO comparison with A/A controls."""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from run_paired_ppo import PATTERNS
from summarize_blocks import summarize_log_differences


def summarize(directory):
    """Validate paired identities/order and report positive-as-faster log ratios."""
    raw = (directory / "samples.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    plan = json.loads((directory / "plan.json").read_text())
    if len(rows) != 16 or plan["orders"] != list(PATTERNS):
        raise ValueError("incomplete or changed fixed plan")
    if (
        plan["roots"]["a"] != plan["roots"]["b"]
        or plan["roots"]["c"] != plan["roots"]["d"]
    ):
        raise ValueError("same-binary roots differ")
    signatures = set()
    values = {}
    for index, row in enumerate(rows):
        block, slot = divmod(index, 4)
        label = PATTERNS[block][slot]
        if (row["block"], row["slot"], row["variant"], row["order"]) != (
            block,
            slot,
            label,
            list(PATTERNS[block]),
        ):
            raise ValueError("row order differs from plan")
        data = row["data"]
        seconds = data["timing"]["training_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid elapsed time")
        provenance = data["provenance"]
        if (
            provenance["native_sha256"] != plan["native_hashes"][label]
            or not provenance["expected_native_sha256_verified"]
        ):
            raise ValueError("native hash differs from plan")
        signatures.add(
            json.dumps(
                [
                    data["config"],
                    data["budget"],
                    data["semantic_sha256"],
                    provenance["script_sha256"],
                    provenance["reference_config_sha256"],
                    provenance["cpu_affinity"],
                    provenance["thread_environment"],
                    provenance["versions"],
                ],
                sort_keys=True,
            )
        )
        values[block, label] = seconds
    if len(signatures) != 1:
        raise ValueError(
            "PPO config, budget, semantics or runtime settings differ"
        )
    rng = random.Random(738)
    primary = []
    ab = []
    cd = []
    for block in range(4):
        logs = {key: math.log(values[block, key]) for key in "abcd"}
        primary.append((logs["a"] + logs["b"] - logs["c"] - logs["d"]) / 2)
        ab.append(logs["a"] - logs["b"])
        cd.append(logs["c"] - logs["d"])
    metrics = {}
    for name, labels in [("control", "ab"), ("candidate", "cd")]:
        times = [
            value for (_, label), value in values.items() if label in labels
        ]
        metrics[name] = {
            "samples": len(times),
            "seconds": times,
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
        }
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "samples": 16,
        "blocks": 4,
        "semantic_parity": True,
        "same_binary_pairs_verified": True,
        "interpretation": "Positive means faster at the identical fixed budget; four-block bootstrap is descriptive, not a significance guarantee. All sixteen predeclared trials are retained.",
        "metrics": metrics,
        "candidate_over_control_rate": summarize_log_differences(primary, rng),
        "b_over_a_same_binary_rate": summarize_log_differences(ab, rng),
        "d_over_c_same_binary_rate": summarize_log_differences(cd, rng),
    }


def main():
    """Write one fresh report from completed raw fixed-trial records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
