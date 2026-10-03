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
"""Summarize explicitly selected final JSONL samples without pooling experiments."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from runtime import VARIANTS


def summarize(paths: list[Path]) -> list[dict]:
    """Compute each configuration's medians, ranges, and sample dispersion."""
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    identities = set()
    for path in paths:
        for line in path.read_text().splitlines():
            data = json.loads(line)
            config = (
                data["env"],
                data["num_envs"],
                data["batch_size"],
                data["threads"],
                data["pin"],
            )
            variant = data["variant"]
            if variant not in VARIANTS:
                raise ValueError(f"unexpected variant: {variant}")
            identity = (*config, variant, data["rep"])
            if identity in identities:
                raise ValueError(
                    f"duplicate configuration/variant/rep: {identity}"
                )
            identities.add(identity)
            groups[config][variant].append(data)
    rows = []
    for config, variants in groups.items():
        if not {"original", "candidate"} <= variants.keys():
            raise ValueError(
                "every case requires original and candidate samples"
            )
        row = {"config": list(config), "variants": {}}
        for variant, values in variants.items():
            rates = [data["env_steps_per_s"] for data in values]
            row["variants"][variant] = {
                "median": statistics.median(rates),
                "min": min(rates),
                "max": max(rates),
                "cv": statistics.stdev(rates) / statistics.mean(rates)
                if len(rates) > 1
                else 0,
                "samples": len(rates),
                "median_us_per_call": statistics.median(
                    data["us_per_call"] for data in values
                ),
                "median_cpu_seconds_per_wall_second": statistics.median(
                    data["cpu_seconds"] / data["seconds"] for data in values
                ),
            }
        medians = {
            key: value["median"] for key, value in row["variants"].items()
        }
        row["vs_original_pct"] = (
            medians["candidate"] / medians["original"] - 1
        ) * 100
        row["vs_rebuilt_pct"] = (
            (medians["candidate"] / medians["rebuilt"] - 1) * 100
            if "rebuilt" in medians
            else None
        )
        rows.append(row)
    return rows


def dispersion(rows: list[dict]) -> str:
    """Format all variant ranges; CV is sample standard deviation / mean."""
    lines = []
    for row in rows:
        env, count, batch, threads, pin = row["config"]
        lines.append(
            f"{env} N={count}, batch={batch}, threads={threads}, "
            f"affinity={'pinned' if pin else 'default'}"
        )
        for variant, value in row["variants"].items():
            lines.append(
                f"  {variant:9s} median={value['median']:,.0f}; "
                f"range={value['min']:,.0f}..{value['max']:,.0f}; "
                f"CV={value['cv']:.1%}; n={value['samples']}; "
                f"vector-call latency={value['median_us_per_call']:.2f} us"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print a comparison and optionally write machine-readable/range reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dispersion-out", type=Path)
    args = parser.parse_args()
    rows = summarize(args.paths)
    print(
        "EnvPool core: environment responses/s (includes autoreset responses)"
    )
    print(
        "Environment N B T affinity original rebuilt candidate vs-original vs-rebuilt"
    )
    for row in rows:
        env, count, batch, threads, pin = row["config"]
        medians = {
            key: value["median"] for key, value in row["variants"].items()
        }
        rebuilt_change = row["vs_rebuilt_pct"]
        print(
            f"{env} {count} {batch} {threads} {'pinned' if pin else 'default'} "
            f"{medians['original']:,.0f} {medians.get('rebuilt', float('nan')):,.0f} "
            f"{medians['candidate']:,.0f} {row['vs_original_pct']:+.1f}% "
            + (
                f"{rebuilt_change:+.1f}%"
                if rebuilt_change is not None
                else "n/a"
            )
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    if args.dispersion_out:
        args.dispersion_out.write_text(dispersion(rows))


if __name__ == "__main__":
    main()
