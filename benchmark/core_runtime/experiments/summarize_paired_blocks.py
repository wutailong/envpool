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
"""Summarize paired a/b control and c/d treatment replicates against a saved plan."""

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from summarize_blocks import summarize_log_differences, summarize_rows

PATTERNS = ("acdb", "cabd", "bdca", "dbac")
CONFIG_FIELDS = ("env", "num_envs", "batch_size", "threads", "pin")
SETTINGS = (
    "warmup",
    "requested_seconds",
    "seed",
    "async",
    "affinity",
    "worker_affinity_offset",
    "runtime_selection",
)
HASH_FIELDS = (
    "binary_sha256",
    "runtime_sha256",
    "binary_hashes",
    "runtime_hashes",
)


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _plan_configs(plan: dict) -> set[tuple]:
    """Resolve the saved runner settings without guessing missing configurations."""
    cases = plan["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty or invalid planned cases")
    keys = []
    for case in cases:
        key = (case["env"], case["n"], case["batch"], case["threads"])
        if (
            not isinstance(key[0], str)
            or not key[0]
            or not all(_positive_int(value) for value in key[1:])
            or case["batch"] > case["n"]
        ):
            raise ValueError("invalid planned configuration")
        keys.append(key)
    if (
        len(set(keys)) != len(keys)
        or type(plan["case_count"]) is not int
        or plan["case_count"] != len(keys)
    ):
        raise ValueError("duplicate or mismatched planned cases")
    affinities = plan.get("affinities")
    if affinities is None:
        affinities = {
            "default": ["default"],
            "default (all allowed CPUs)": ["default"],
            "pinned": ["pinned"],
        }.get(plan.get("affinity"))
    if (
        not isinstance(affinities, list)
        or not affinities
        or any(value not in ("default", "pinned") for value in affinities)
        or len(set(affinities)) != len(affinities)
    ):
        raise ValueError("invalid planned affinities")
    if (
        not _positive_int(plan["blocks"])
        or type(plan["warmup"]) is not int
        or plan["warmup"] < 0
        or type(plan["seconds"]) not in (int, float)
        or not math.isfinite(plan["seconds"])
        or plan["seconds"] <= 0
    ):
        raise ValueError("invalid planned run settings")
    configs = {
        (*key, affinity == "pinned") for key in keys for affinity in affinities
    }
    expected = plan["blocks"] * len(configs) * 4
    if (
        type(plan["expected_samples"]) is not int
        or plan["expected_samples"] != expected
    ):
        raise ValueError(
            "planned sample count disagrees with configurations/blocks"
        )
    return configs


def _hash_value(value):
    """Normalize a SHA-256 digest or a nonempty binary-name-to-digest mapping."""
    if isinstance(value, str) and len(value) == 64:
        if all(char in "0123456789abcdefABCDEF" for char in value):
            return value.lower()
    if isinstance(value, dict) and value:
        if all(isinstance(key, str) and key for key in value):
            if all(isinstance(digest, str) for digest in value.values()):
                return {
                    key: _hash_value(digest)
                    for key, digest in sorted(value.items())
                }
    raise ValueError("invalid measured binary SHA-256 fingerprint")


def _binary_identity(samples: list[dict], hash_fields: list[str]) -> dict:
    """Check measured fingerprints only; labels and runtime paths are not proof."""
    if not hash_fields:
        return {
            "status": "unverified",
            "reason": "No measured binary hashes in rows; pair identity requires separate provenance.",
        }
    by_label = {}
    for label in "abcd":
        fingerprints = [
            {field: _hash_value(row[field]) for field in hash_fields}
            for row in samples
            if row["variant"] == label
        ]
        if any(value != fingerprints[0] for value in fingerprints):
            raise ValueError("measured binary hash changed within a label")
        by_label[label] = fingerprints[0]
    for left, right in (("a", "b"), ("c", "d")):
        if by_label[left] != by_label[right]:
            raise ValueError(
                f"same-binary pair {left}/{right} has mismatched measured hashes"
            )
    return {
        "status": "verified",
        "hash_fields": hash_fields,
        "by_label": by_label,
    }


def _validate(rows: list[dict], plan: dict, hash_field: str | None) -> dict:
    """Require every planned four-slot block and consistent measurement settings."""
    configs = _plan_configs(plan)
    if len(rows) != plan["expected_samples"]:
        raise ValueError("empty or incomplete experiment relative to plan")
    candidates = list(HASH_FIELDS)
    if hash_field and hash_field not in candidates:
        candidates.append(hash_field)
    hash_fields = [
        field for field in candidates if any(field in row for row in rows)
    ]
    if hash_field and hash_field not in hash_fields:
        raise ValueError("requested measured binary hash field is absent")
    if any(field not in row for row in rows for field in hash_fields):
        raise ValueError("incomplete measured binary hash evidence")
    groups = collections.defaultdict(list)
    identities = set()
    for row in rows:
        key = tuple(row[field] for field in CONFIG_FIELDS)
        if (
            type(row["pin"]) is not bool
            or not isinstance(row["env"], str)
            or not all(
                _positive_int(row[field]) for field in CONFIG_FIELDS[1:4]
            )
            or key not in configs
        ):
            raise ValueError("unexpected configuration relative to plan")
        if (
            type(row["block"]) is not int
            or not 0 <= row["block"] < plan["blocks"]
            or type(row["slot"]) is not int
            or not 0 <= row["slot"] < 4
            or type(row["expected_samples"]) is not int
            or row["expected_samples"] != plan["expected_samples"]
            or type(row["rep"]) is not int
            or row["rep"] != row["block"]
        ):
            raise ValueError("invalid or mismatched block metadata")
        identity = (*key, row["block"], row["slot"])
        if identity in identities:
            raise ValueError("duplicate configuration/block/slot")
        identities.add(identity)
        order = list(PATTERNS[row["block"] % 4])
        if row["order"] != order or row["variant"] != order[row["slot"]]:
            raise ValueError(
                "expected exact a/b and c/d labels in paired-replicate order"
            )
        if (
            type(row["warmup"]) is not int
            or row["warmup"] != plan["warmup"]
            or type(row["requested_seconds"]) not in (int, float)
            or row["requested_seconds"] != plan["seconds"]
            or type(row["seed"]) is not int
            or ("seed" in plan and row["seed"] != plan["seed"])
            or type(row["async"]) is not bool
            or row["async"] != (row["batch_size"] != row["num_envs"])
        ):
            raise ValueError("mismatched measurement settings")
        groups[key].append(row)
    if set(groups) != configs:
        raise ValueError("missing planned configuration")
    evidence = {}
    for key, samples in groups.items():
        if len(samples) != plan["blocks"] * 4:
            raise ValueError("missing planned blocks or slots")
        settings = {field: samples[0][field] for field in SETTINGS}
        if any(
            any(row[field] != settings[field] for field in SETTINGS)
            for row in samples
        ):
            raise ValueError(
                "mismatched measurement settings within configuration"
            )
        evidence[key] = {
            "settings": settings,
            "same_binary_pairs": _binary_identity(samples, hash_fields),
        }
    return evidence


def summarize_paired(
    path: Path, plan_path: Path, hash_field: str | None = None
) -> dict:
    """Use equal-weight block log effects; keep cases separate and raw input untouched."""
    raw = path.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
        plan = json.loads(plan_path.read_text())
        evidence = _validate(rows, plan, hash_field)
        summaries = summarize_rows(rows, "a")
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise ValueError("malformed experiment rows or plan") from error
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in CONFIG_FIELDS)].append(row)
    rng = random.Random(738)
    output = []
    for summary in summaries:
        key = tuple(summary["config"])
        samples = groups[key]
        blocks = sorted({row["block"] for row in samples})
        effects = {"treatment_vs_control": [], "b_over_a": [], "d_over_c": []}
        for block in blocks:
            logs = {
                row["variant"]: math.log(row["env_steps_per_s"])
                for row in samples
                if row["block"] == block
            }
            effects["treatment_vs_control"].append(
                statistics.mean((logs["c"], logs["d"]))
                - statistics.mean((logs["a"], logs["b"]))
            )
            effects["b_over_a"].append(logs["b"] - logs["a"])
            effects["d_over_c"].append(logs["d"] - logs["c"])
        contrasts = {
            name: {
                "block_ids": blocks,
                "block_log_effects": values,
                **summarize_log_differences(values, rng),
            }
            for name, values in effects.items()
        }
        output.append({
            "config": summary["config"],
            "blocks": summary["blocks"],
            **evidence[key],
            "metrics": summary["metrics"],
            "primary": contrasts["treatment_vs_control"],
            "aa_diagnostics": {
                name: contrasts[name] for name in ("b_over_a", "d_over_c")
            },
        })
    return {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "samples": len(rows),
        "plan": plan,
        "control_labels": ["a", "b"],
        "treatment_labels": ["c", "d"],
        "primary_estimand": "mean_blocks(mean(log(c), log(d)) - mean(log(a), log(b)))",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 738,
        "bootstrap_caveat": "Few blocks; descriptive uncertainty, not proof of statistical significance. Each interval resamples complete blocks.",
        "configurations": output,
    }


def main() -> None:
    """Write a fresh JSON summary, refusing to replace input or previous results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument(
        "--binary-hash-field",
        help="Measured SHA-256 field, containing a digest or binary-name-to-digest mapping",
    )
    args = parser.parse_args()
    if args.json_out.exists():
        parser.error("--json-out must be a new file")
    result = summarize_paired(args.path, args.plan, args.binary_hash_field)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.json_out.open("x") as output:
        output.write(encoded)
    print(
        f"Wrote {len(result['configurations'])} configurations; input preserved."
    )


if __name__ == "__main__":
    main()
