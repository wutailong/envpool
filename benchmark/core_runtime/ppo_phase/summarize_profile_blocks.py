# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Validate and describe a complete fixed PPO phase/observer-cost trial."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
from pathlib import Path

from phase_clock import PARENTS, TOP_LEVEL, WALL_CLOCK_TOLERANCE_NS
from run_profile_blocks import (
    CONDITIONS,
    ORDERS,
    validate_orders,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]


def load_helper(name, path):
    """Load existing dependency-free helpers without importing native runtimes."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize_log_differences = load_helper(
    "_ppo_block_summary", ROOT / "experiments/summarize_blocks.py"
).summarize_log_differences
fingerprint = load_helper(
    "_ppo_reference_fingerprint", ROOT / "experiments/bench_ppo.py"
).fingerprint
BUDGET = {"updates": 100, "env_steps": 256000, "optimizer_steps": 8000}


def require(condition, message):
    """Reject inconsistent evidence rather than silently dropping observations."""
    if not condition:
        raise ValueError(message)


def digest(value):
    """Require a normalized SHA-256 identity."""
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "invalid SHA-256 identity",
    )
    return value


def positive(value):
    """Require finite positive numeric wall/rate/CPU measurements."""
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "timing/rate must be positive and finite",
    )
    return value


def integer(value, minimum=0):
    """Keep counts and integer-clock intervals exact, rejecting booleans."""
    require(
        type(value) is int and value >= minimum, "invalid integer count/time"
    )
    return value


def validate_profile(data):
    """Reconstruct disjoint accounting, including the reported wall roundoff."""
    profile = data["phase_profile"]
    require(profile["mode"] == "timed", "profile is not timed")
    phases = profile["phases"]
    require(set(phases) == set(PARENTS), "unexpected phase set")
    for name, phase in phases.items():
        calls = integer(phase["calls"])
        require(phase["parent"] == PARENTS[name], "invalid phase nesting")
        require(integer(phase["exceptions"]) == 0, "phase raised an exception")
        inclusive = integer(phase["inclusive_ns"])
        children = integer(phase["direct_child_ns"])
        exclusive = integer(phase["exclusive_ns"])
        require(inclusive - children == exclusive, "exclusive time mismatch")
        require(
            inclusive > 0 if calls else inclusive == 0, "call/time mismatch"
        )
        if name != "collect":
            require(children == 0, "unexpected nested child timing")
        if name in TOP_LEVEL:
            require(calls == 100, "top-level call count mismatch")
    require(phases["env_step"]["calls"] == 12800, "env_step count mismatch")
    resets = phases["env_reset"]["calls"]
    episodes = data["budget"]["train_episodes"]
    require(
        math.ceil(episodes / 20) <= resets <= min(episodes, 12800),
        "episode/subset env_reset count mismatch",
    )
    env_ns = sum(
        phases[name]["inclusive_ns"] for name in ("env_step", "env_reset")
    )
    require(
        phases["collect"]["direct_child_ns"] == env_ns,
        "collect child sum mismatch",
    )
    wall = integer(profile["training_wall_ns"], 1)
    require(
        wall == round(data["timing"]["training_seconds"] * 1e9),
        "wall clock mismatch",
    )
    raw_other = wall - sum(phases[name]["inclusive_ns"] for name in TOP_LEVEL)
    require(profile["raw_loop_other_ns"] == raw_other, "loop residual mismatch")
    require(
        raw_other >= -WALL_CLOCK_TOLERANCE_NS, "phases exceed wall envelope"
    )
    validation = profile["validation"]
    require(validation["valid"] is True, "profile validation failed")
    for key, expected in {
        "integer_interval_tolerance_ns": 0,
        "wall_envelope_tolerance_ns": WALL_CLOCK_TOLERANCE_NS,
        "wall_roundoff_clamped_ns": max(0, -raw_other),
        "expected_updates": 100,
        "expected_env_step_calls": 12800,
    }.items():
        require(
            integer(validation[key]) == expected,
            f"invalid profile validation: {key}",
        )
    partition = {
        "policy_train_ns": phases["policy_train"]["exclusive_ns"],
        "collect_excluding_env_calls_ns": phases["collect"]["exclusive_ns"],
        "env_step_in_collect_ns": phases["env_step"]["inclusive_ns"],
        "env_reset_in_collect_ns": phases["env_reset"]["inclusive_ns"],
        "ppo_update_ns": phases["ppo_update"]["exclusive_ns"],
        "buffer_reset_ns": phases["buffer_reset"]["exclusive_ns"],
        "loop_other_ns": max(0, raw_other),
    }
    require(
        profile["disjoint_partition_ns"] == partition,
        "disjoint partition mismatch",
    )
    require(
        sum(partition.values()) - validation["wall_roundoff_clamped_ns"]
        == wall,
        "partition double counts or omits wall time",
    )
    require(env_ns < wall, "modeled env-call region leaves no other time")
    instrument = data["instrumentation"]
    require(
        instrument["schema_version"] == 1, "instrumentation schema mismatch"
    )
    require(
        instrument["clock"] == "time.perf_counter_ns", "unexpected phase clock"
    )
    positive(instrument["clock_resolution_seconds"])
    require(
        instrument["semantic_hashes_include_phase_data"] is False,
        "phase data contaminates semantics",
    )
    require(
        instrument["saves_weights_or_rollout_arrays"] is False,
        "unexpected saved arrays",
    )
    return {name: phase["calls"] for name, phase in phases.items()}


def validate_trial(plan, rows, complete):
    """Check the exact 32-row schedule, frozen identities and reference work."""
    validate_orders()
    require(plan["blocks"] == 4 and plan["samples"] == 32, "wrong trial size")
    require(plan["orders"] == list(ORDERS), "unplanned block orders")
    require(
        plan["budget"] == BUDGET and plan["warmup_updates"] == 2,
        "plan budget mismatch",
    )
    require(
        set(plan["conditions"]) == set(CONDITIONS), "plan conditions mismatch"
    )
    require(complete["samples"] == 32, "incomplete completion marker")
    require(complete["semantic_parity"] is True, "completion parity failed")
    require(
        complete["frozen_native_and_script_hashes_match"] is True,
        "completion identities failed",
    )
    require(len(rows) == 32, "missing or extra trial rows")
    natives, scripts = {}, {}
    for label, (runtime, mode) in CONDITIONS.items():
        condition = plan["conditions"][label]
        require(
            (condition["runtime"], condition["mode"]) == (runtime, mode),
            "condition meaning changed",
        )
        native = digest(condition["native_sha256"])
        script = digest(condition["script_sha256"])
        native_identity = (native, condition["package"])
        script_identity = (script, condition["script"])
        require(
            natives.setdefault(runtime, native_identity) == native_identity,
            "same-runtime native/package mismatch",
        )
        require(
            scripts.setdefault(mode, script_identity) == script_identity,
            "same-mode script mismatch",
        )
    reference_config = json.loads((ROOT / "ppo/config.json").read_text())
    reference, counts, semantic = None, None, None
    last_started = 0
    for index, row in enumerate(rows):
        block, slot = divmod(index, 8)
        label = ORDERS[block][slot]
        condition = plan["conditions"][label]
        require(
            (row["block"], row["slot"], row["variant"], row["order"])
            == (block, slot, label, list(ORDERS[block])),
            "rows are incomplete, duplicated or out of planned order",
        )
        require(
            (row["runtime"], row["mode"]) == CONDITIONS[label],
            "row condition mismatch",
        )
        started = positive(row["started_unix"])
        require(started >= last_started, "run start order mismatch")
        last_started = started
        data = row["data"]
        require(
            data["schema_version"] == 1 and data["label"] == "sample",
            "result schema/label mismatch",
        )
        validate_result(data, condition, condition["script_sha256"])
        require(
            data["config"] == reference_config,
            "reference configuration mismatch",
        )
        require(
            set(data["budget"]) == {*BUDGET, "train_episodes"},
            "budget schema mismatch",
        )
        integer(data["budget"]["train_episodes"])
        require(
            data["measurement"]["fresh_seeded_run_after_warmup"] is True,
            "fresh-run isolation failed",
        )
        timing = data["timing"]
        require(
            set(timing)
            == {
                "training_seconds",
                "process_cpu_seconds",
                "env_steps_per_second",
                "optimizer_steps_per_second",
            },
            "timing schema mismatch",
        )
        for value in timing.values():
            positive(value)
        for key, work in (
            ("env_steps_per_second", 256000),
            ("optimizer_steps_per_second", 8000),
        ):
            require(
                math.isclose(
                    timing[key],
                    work / timing["training_seconds"],
                    rel_tol=1e-12,
                ),
                "rate/work mismatch",
            )
        provenance = data["provenance"]
        for flag in (
            "package_path_verified",
            "native_path_verified",
            "expected_native_sha256_verified",
            "torch_deterministic",
        ):
            require(provenance[flag] is True, f"unverified provenance: {flag}")
        require(
            provenance["native_module"]
            == "envpool.classic_control.classic_control_envpool",
            "native module mismatch",
        )
        require(
            provenance["torch_threads"]
            == provenance["torch_interop_threads"]
            == 1,
            "Torch threads mismatch",
        )
        require(
            provenance["thread_environment"]
            == dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                ),
                "1",
            ),
            "thread environment mismatch",
        )
        require(
            set(provenance["versions"])
            == {"envpool", "torch", "tianshou", "numpy", "gymnasium", "numba"}
            and all(
                isinstance(value, str) and value
                for value in provenance["versions"].values()
            ),
            "version schema mismatch",
        )
        require(
            provenance["versions"]["tianshou"] == "0.5.1",
            "reference Tianshou version mismatch",
        )
        affinity = provenance["cpu_affinity"]
        require(
            affinity is None
            or (
                isinstance(affinity, list)
                and affinity
                and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
                and affinity == sorted(set(affinity))
            ),
            "invalid CPU affinity",
        )
        digest(provenance["reference_config_sha256"])
        shared = {
            key: provenance[key]
            for key in (
                "reference_config_sha256",
                "python_version",
                "versions",
                "cpu_affinity",
                "thread_environment",
                "torch_threads",
                "torch_interop_threads",
                "torch_deterministic",
            )
        }
        shared.update(
            config=data["config"],
            budget=data["budget"],
            measurement=data["measurement"],
        )
        if reference is None:
            reference = shared
        require(
            shared == reference,
            "config/budget/version/affinity/thread inconsistency",
        )
        identity = digest(data["semantic_sha256"])
        require(
            fingerprint(data["semantic_fingerprints"]) == identity,
            "semantic fingerprint digest mismatch",
        )
        require(
            data["semantic_fingerprints"]["config"]
            == fingerprint(data["config"]),
            "semantic config mismatch",
        )
        if semantic is None:
            semantic = identity
        require(identity == semantic, "semantic hashes differ across trial")
        if row["mode"] == "profile":
            require(
                provenance["reference_script_sha256"] == scripts["plain"][0],
                "profile reference-script hash mismatch",
            )
            current = (
                validate_profile(data),
                digest(provenance["phase_helper_sha256"]),
            )
            if counts is None:
                counts = current
            require(
                current == counts, "profile call counts/helper hashes differ"
            )
        else:
            require(
                "instrumentation" not in data,
                "plain run includes instrumentation",
            )
    return semantic


def spread(values):
    """Describe run-level values without treating phases or calls as replicates."""
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def profile_summary(rows):
    """Describe eight profiled runs per runtime, each divided by its own wall."""
    output = {}
    for runtime in ("original", "retained"):
        selected = [
            row
            for row in rows
            if row["runtime"] == runtime and row["mode"] == "profile"
        ]
        raw = []
        for row in selected:
            data = row["data"]
            wall = data["timing"]["training_seconds"]
            phase_seconds = {
                key.removesuffix("_ns"): value / 1e9
                for key, value in data["phase_profile"][
                    "disjoint_partition_ns"
                ].items()
            }
            shares = {key: value / wall for key, value in phase_seconds.items()}
            env_seconds = (
                phase_seconds["env_step_in_collect"]
                + phase_seconds["env_reset_in_collect"]
            )
            env_share = env_seconds / wall
            raw.append({
                "block": row["block"],
                "slot": row["slot"],
                "variant": row["variant"],
                "training_seconds": wall,
                "process_cpu_seconds": data["timing"]["process_cpu_seconds"],
                "process_cpu_per_wall": data["timing"]["process_cpu_seconds"]
                / wall,
                "phase_seconds": phase_seconds,
                "own_wall_shares": shares,
                "env_call_seconds": env_seconds,
                "env_call_share": env_share,
                "conditional_2x_env_call_total_speedup": 1
                / (1 - env_share / 2),
                "conditional_zero_cost_env_call_total_speedup": 1
                / (1 - env_share),
                "exact_counts": {
                    key: phase["calls"]
                    for key, phase in data["phase_profile"]["phases"].items()
                },
                "train_episodes": data["budget"]["train_episodes"],
                "wall_roundoff_clamped_ns": data["phase_profile"]["validation"][
                    "wall_roundoff_clamped_ns"
                ],
            })
        metrics = (
            "training_seconds",
            "process_cpu_seconds",
            "process_cpu_per_wall",
            "env_call_seconds",
            "env_call_share",
            "conditional_2x_env_call_total_speedup",
            "conditional_zero_cost_env_call_total_speedup",
        )
        output[runtime] = {
            "runs": 8,
            "context": {
                key: spread([run[key] for run in raw]) for key in metrics
            },
            "phases": {
                key: {
                    "seconds": spread([
                        run["phase_seconds"][key] for run in raw
                    ]),
                    "own_wall_share": spread([
                        run["own_wall_shares"][key] for run in raw
                    ]),
                }
                for key in raw[0]["phase_seconds"]
            },
            "exact_counts_each_run": raw[0]["exact_counts"],
            "raw_run_metrics": raw,
        }
    return output


def summarize_trial(plan, rows, complete):
    """Use four complete-block contrasts and retain all original observations."""
    semantic = validate_trial(plan, rows, complete)
    logs = [
        {
            row["variant"]: math.log(row["data"]["timing"]["training_seconds"])
            for row in rows[block * 8 : (block + 1) * 8]
        }
        for block in range(4)
    ]
    definitions = {
        "native_plain_rate": ("ab", "ef", "positive means retained is faster"),
        "native_profile_rate_descriptive": (
            "cd",
            "gh",
            "positive means retained is faster; profiled descriptive contrast only",
        ),
        "original_observer_time_overhead": (
            "cd",
            "ab",
            "positive means profiling is slower",
        ),
        "retained_observer_time_overhead": (
            "gh",
            "ef",
            "positive means profiling is slower",
        ),
        "aa_b_over_a_rate": ("a", "b", "positive means b is faster than a"),
        "aa_d_over_c_rate": ("c", "d", "positive means d is faster than c"),
        "aa_f_over_e_rate": ("e", "f", "positive means f is faster than e"),
        "aa_h_over_g_rate": ("g", "h", "positive means h is faster than g"),
    }
    contrasts = {}
    for name, (left, right, meaning) in definitions.items():
        diffs = [
            statistics.mean(block[label] for label in left)
            - statistics.mean(block[label] for label in right)
            for block in logs
        ]
        contrasts[name] = {
            "interpretation": meaning,
            "formula": f"mean(log seconds of {left}) - mean(log seconds of {right})",
            "block_log_contrasts": diffs,
            **summarize_log_differences(diffs, random.Random(738)),
        }
    return {
        "schema_version": 1,
        "validation": {
            "valid": True,
            "samples": 32,
            "profile_samples": 16,
            "semantic_sha256": semantic,
        },
        "uncertainty": {
            "units": "four complete blocks",
            "blocks": 4,
            "bootstrap_resamples": 10000,
            "seed": 738,
            "caveat": "Exploratory percentile block bootstrap with only four blocks; not significance proof. No call/update pooling. Position balance does not eliminate carryover or thermal drift.",
        },
        "contrasts": contrasts,
        "profiles": profile_summary(rows),
        "interpretation": [
            "Absolute phase seconds and own-run wall shares include observer cost; do not mechanically subtract total overhead from individual phases.",
            "The disjoint partition replaces inclusive collect with collect excluding env calls plus env_step and env_reset. Inclusive collect must not be added again.",
            "Modeled total speedups hold all other time unchanged and are NOT a hard core ceiling. The env-call region includes Python adaptation, conversion, native waiting and observer overhead; background workers can affect other phases.",
            "Process CPU/whole wall is contextual aggregate process utilization, not a disjoint phase or per-core efficiency measure. Negative observer and A/A estimates are retained.",
        ],
        "raw_plan": plan,
        "raw_completion": complete,
        "raw_rows": rows,
    }


def summarize(directory):
    """Read one immutable byte snapshot and hash the complete source evidence."""
    snapshots = {
        name: (Path(directory) / name).read_bytes()
        for name in ("plan.json", "samples.jsonl", "complete.json")
    }
    rows = [
        json.loads(line) for line in snapshots["samples.jsonl"].splitlines()
    ]
    try:
        result = summarize_trial(
            json.loads(snapshots["plan.json"]),
            rows,
            json.loads(snapshots["complete.json"]),
        )
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(
            f"missing or malformed trial schema: {error}"
        ) from error
    result["source_sha256"] = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in snapshots.items()
    }
    return result


def main():
    """Write a new JSON summary only after the complete fixed trial validates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("summary output exists; choose a new file")
    result = summarize(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "validation": result["validation"],
                "contrasts": result["contrasts"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
