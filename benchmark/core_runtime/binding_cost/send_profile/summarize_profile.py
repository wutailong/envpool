#!/usr/bin/env python3
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

"""Summarize the fixed coarse/fine blocks without collecting new samples."""

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SOURCE / "benchmark/core_runtime/experiments"))
from summarize_blocks import summarize_log_differences


def main():
    """Apply the original descriptive mathematics to one completed run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path", type=Path, help="Completed fixed-run output directory"
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh summary JSON outside the repository",
    )
    a = p.parse_args()
    path = a.path.expanduser().resolve(strict=True)
    summary_output = a.output.expanduser().resolve()
    assert (
        not summary_output.exists()
        and summary_output.parent.is_dir()
        and not summary_output.is_relative_to(SOURCE)
    )
    blob = (path / "samples.jsonl").read_bytes()
    rows = [json.loads(s) for s in blob.splitlines()]
    assert (
        len(rows)
        == json.loads((path / "complete.json").read_text())["samples"]
        == 64
    )
    plan = json.loads((path / "plan.json").read_text())
    seen = set()
    groups = collections.defaultdict(list)
    for r in rows:
        ident = (r["case"], r["block"], r["slot"])
        assert ident not in seen
        seen.add(ident)
        assert (
            r["order"] == plan["orders"][r["block"]]
            and r["label"] == r["order"][r["slot"]]
        )
        groups[r["case"]].append(r)
    rng = random.Random(739)
    result = []

    def summary(v):
        return {
            "n": len(v),
            "median": statistics.median(v),
            "min": min(v),
            "max": max(v),
            "cv_pct": statistics.stdev(v) / statistics.mean(v) * 100
            if len(v) > 1 and statistics.mean(v)
            else None,
        }

    for case, items in sorted(groups.items()):
        assert len(items) == 16
        output = {"case": case, "clocks": {}}
        for clock in ["wall_ns", "caller_cpu_ns", "process_cpu_ns"]:
            effects = {"primary": [], "b_over_a": [], "d_over_c": []}
            for b in range(4):
                m = {
                    r["label"]: -math.log(r["sample"][clock])
                    for r in items
                    if r["block"] == b
                }
                assert set(m) == set("abcd")
                effects["primary"].append(
                    (m["c"] + m["d"] - m["a"] - m["b"]) / 2
                )
                effects["b_over_a"].append(m["b"] - m["a"])
                effects["d_over_c"].append(m["d"] - m["c"])
            output["clocks"][clock] = {
                "us_per_call": {
                    name: summary([
                        r["sample"][clock] / 5000 / 1000
                        for r in items
                        if r["label"] in labs
                    ])
                    for name, labs in [("coarse", "ab"), ("fine", "cd")]
                },
                **{
                    name: summarize_log_differences(values, rng)
                    for name, values in effects.items()
                },
            }
        for mode, labs in [("coarse", "ab"), ("fine", "cd")]:
            values = collections.defaultdict(list)
            for r in items:
                if r["label"] not in labs:
                    continue
                s = r["sample"]
                p = s["profile"]
                f = p["fine_enqueue"]
                for kind, phases in [
                    ("outer", p["phases"]),
                    ("fine", f["phases"]),
                ]:
                    for name, phase in phases.items():
                        for clock in ["wall_ns", "caller_cpu_ns"]:
                            values[f"{kind}/{name}/{clock}_us"].append(
                                phase[clock] / 5000 / 1000
                            )
                            values[
                                f"{kind}/{name}/{clock}_step_share_pct"
                            ].append(phase[clock] / s[clock] * 100)
                            if kind == "fine":
                                values[
                                    f"{kind}/{name}/{clock}_enqueue_share_pct"
                                ].append(
                                    phase[clock]
                                    / p["phases"]["enqueue"][clock]
                                    * 100
                                )
                values["legacy_enqueue_us"].append(
                    p["legacy_enqueue_wall_seconds_since_reset"] * 1e6 / 5000
                )
                values["post_requests_per_call"].append(
                    f["work_permit_posix_post_requests"] / 5000
                )
                values["nested_unattributed_wall_us"].append(
                    (
                        p["phases"]["enqueue"]["wall_ns"]
                        - sum(x["wall_ns"] for x in f["phases"].values())
                    )
                    / 5000
                    / 1000
                )
            output[mode] = {k: summary(v) for k, v in values.items()}
        result.append(output)
    out = {
        "samples": 64,
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        "groups": result,
        "caveat": "Four-block descriptive intervals; fine nested boundaries include paired-clock and scheduling disturbance. Post requests are not kernel wakes. No subtraction or optimization claim.",
    }
    with summary_output.open("x") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for r in result:
        print(r["case"])
        for clock in ["wall_ns", "caller_cpu_ns", "process_cpu_ns"]:
            s = r["clocks"][clock]
            e = s["primary"]
            print(
                clock,
                "medians",
                [
                    round(s["us_per_call"][n]["median"], 3)
                    for n in ["coarse", "fine"]
                ],
                "rate effect",
                e["block_geomean_effect_pct"],
                "CI",
                e["exploratory_block_bootstrap_95pct"],
                "AA",
                [
                    s[n]["block_geomean_effect_pct"]
                    for n in ["b_over_a", "d_over_c"]
                ],
            )
        for n, v in r["fine"].items():
            if (
                n.endswith("_us")
                or "signal" in n
                or n == "post_requests_per_call"
            ):
                print(n, "median/min/max", v["median"], v["min"], v["max"])
        print("coarse legacy", r["coarse"]["legacy_enqueue_us"])


if __name__ == "__main__":
    main()
