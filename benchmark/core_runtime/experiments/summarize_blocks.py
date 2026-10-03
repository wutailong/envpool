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
"""Summarize one paired-block diagnosis experiment without pooling runs."""

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path


def summarize(path: Path, baseline: str) -> list[dict]:
    """Keep every case and variant; bootstrap blocks only as descriptive evidence."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(row["expected_samples"] != len(rows) for row in rows):
        raise ValueError("empty or incomplete experiment")
    groups = collections.defaultdict(list)
    identities = set()
    for row in rows:
        key = (
            row["env"],
            row["num_envs"],
            row["batch_size"],
            row["threads"],
            row["pin"],
        )
        identity = (*key, row["block"], row["slot"])
        if identity in identities:
            raise ValueError("duplicate configuration/block/slot")
        identities.add(identity)
        if row["variant"] != row["order"][row["slot"]]:
            raise ValueError("sample label differs from planned order")
        if (
            not math.isfinite(row["env_steps_per_s"])
            or row["env_steps_per_s"] <= 0
        ):
            raise ValueError("invalid response rate")
        groups[key].append(row)
    output = []
    rng = random.Random(738)
    for key, samples in sorted(groups.items()):
        labels = list(dict.fromkeys(row["variant"] for row in samples))
        if baseline not in labels:
            raise ValueError("baseline absent from configuration")
        blocks = sorted({row["block"] for row in samples})
        metrics = {}
        block_means = {label: [] for label in labels}
        for block in blocks:
            selected = [row for row in samples if row["block"] == block]
            order = selected[0]["order"]
            if sorted(row["slot"] for row in selected) != list(
                range(len(order))
            ):
                raise ValueError("missing or unexpected block slots")
            if any(row["order"] != order for row in selected):
                raise ValueError("inconsistent block order")
            counts = collections.Counter(row["variant"] for row in selected)
            if set(counts) != set(labels) or len(set(counts.values())) != 1:
                raise ValueError("unbalanced block variants")
            for label in labels:
                block_means[label].append(
                    statistics.mean(
                        math.log(row["env_steps_per_s"])
                        for row in selected
                        if row["variant"] == label
                    )
                )
        for label in labels:
            values = [row for row in samples if row["variant"] == label]
            rates = [row["env_steps_per_s"] for row in values]
            metrics[label] = {
                "n": len(values),
                "median": statistics.median(rates),
                "min": min(rates),
                "max": max(rates),
                "cv": statistics.stdev(rates) / statistics.mean(rates)
                if len(rates) > 1
                else None,
                "cpu_us_per_response": statistics.median(
                    row["cpu_seconds"] / row["env_steps"] * 1e6
                    for row in values
                ),
                "runnable_wait_ns_per_response": statistics.median(
                    row["scheduler_wait_seconds"] / row["env_steps"] * 1e9
                    for row in values
                ),
                "voluntary_switches_per_1m_responses": statistics.median(
                    row["voluntary_context_switches"] / row["env_steps"] * 1e6
                    for row in values
                ),
                "scheduler_thread_sets_stable": all(
                    row["scheduler_thread_set_stable"] for row in values
                ),
            }
        contrasts = {}
        for label in labels:
            if label == baseline:
                continue
            diffs = [
                left - right
                for left, right in zip(
                    block_means[label], block_means[baseline], strict=True
                )
            ]
            bootstrap = sorted(
                math.expm1(statistics.mean(rng.choices(diffs, k=len(diffs))))
                * 100
                for _ in range(10000)
            )
            contrasts[label] = {
                "ratio_of_medians_pct": 100
                * (metrics[label]["median"] / metrics[baseline]["median"] - 1),
                "block_geomean_effect_pct": 100
                * math.expm1(statistics.mean(diffs)),
                "block_effects_pct": [
                    100 * math.expm1(value) for value in diffs
                ],
                "exploratory_block_bootstrap_95pct": [
                    bootstrap[250],
                    bootstrap[9749],
                ],
            }
        output.append({
            "config": list(key),
            "base": baseline,
            "blocks": len(blocks),
            "metrics": metrics,
            "contrasts": contrasts,
            "bootstrap_caveat": "Few blocks; descriptive uncertainty, not proof of statistical significance.",
        })
    return output


def main() -> None:
    """Print complete ranges/effects and optionally save a fresh JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    rows = summarize(args.path, args.baseline)
    for row in rows:
        print(
            f"N/B/T/pinned={row['config']}; baseline={row['base']}; blocks={row['blocks']}"
        )
        for label, value in row["metrics"].items():
            cv = "n/a" if value["cv"] is None else f"{value['cv']:.1%}"
            print(
                f"  {label}: n={value['n']} median={value['median']:,.0f} "
                f"range={value['min']:,.0f}..{value['max']:,.0f} CV={cv}"
            )
        for label, value in row["contrasts"].items():
            print(f"  {label} vs {row['base']}: " + json.dumps(value))
    print("Block bootstrap ranges are descriptive, not significance claims.")
    if args.json_out:
        with args.json_out.open("x") as output:
            json.dump(rows, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
