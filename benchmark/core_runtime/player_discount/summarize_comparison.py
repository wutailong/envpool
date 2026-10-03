# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Summarize a four-label Williams trial with two identical control labels."""

import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from summarize_blocks import summarize_log_differences, summarize_rows  # noqa: E402


def summarize(rows):
    """Pool control log rates within blocks, preserving every configuration."""
    summaries = summarize_rows(rows, "a")
    labels = ["a", "b", "direct", "fast"]
    patterns = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    groups = collections.defaultdict(list)
    for row in rows:
        if row["order"] != [labels[i] for i in patterns[row["block"] % 4]]:
            raise ValueError("unexpected Williams order or label set")
        key = (
            row["env"],
            row["num_envs"],
            row["batch_size"],
            row["threads"],
            row["pin"],
        )
        groups[key].append(row)
    rng = random.Random(738)
    output = []
    for item in summaries:
        selected = groups[tuple(item["config"])]
        effects = {
            name: []
            for name in [
                "direct_vs_control",
                "fast_vs_control",
                "fast_vs_direct",
                "b_over_a",
            ]
        }
        for block in sorted({row["block"] for row in selected}):
            current = [row for row in selected if row["block"] == block]
            if len(current) != 4 or {row["variant"] for row in current} != set(
                labels
            ):
                raise ValueError(
                    "expected exactly one sample of each label per block"
                )
            logs = {
                row["variant"]: math.log(row["env_steps_per_s"])
                for row in current
            }
            control = statistics.mean([logs["a"], logs["b"]])
            effects["direct_vs_control"].append(logs["direct"] - control)
            effects["fast_vs_control"].append(logs["fast"] - control)
            effects["fast_vs_direct"].append(logs["fast"] - logs["direct"])
            effects["b_over_a"].append(logs["b"] - logs["a"])
        output.append({
            "config": item["config"],
            "blocks": item["blocks"],
            "metrics": item["metrics"],
            "contrasts": {
                name: summarize_log_differences(values, rng)
                for name, values in effects.items()
            },
            "caveat": "Descriptive block bootstrap on a shared host; no significance or universal-neutrality guarantee.",
        })
    return output


def main():
    """Write fresh summary evidence without replacing a previous report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize([
        json.loads(line) for line in args.input.read_text().splitlines()
    ])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Summarized {len(result)} configurations")


if __name__ == "__main__":
    main()
