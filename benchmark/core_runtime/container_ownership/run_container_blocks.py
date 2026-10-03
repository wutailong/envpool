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
"""Run the fixed 64-process Container-overhead trial and save its analysis plan.

Example: python run_container_blocks.py --python /path/to/python
    --baseline-root /path/to/baseline --candidate-root /path/to/candidate
    --out /path/to/container.jsonl

Each root must contain an importable envpool package with its Dummy extension.
The output and default OUTPUT.plan.json feed experiments/summarize_paired_blocks.py.
Normal conversion releases baseline payloads, so these timings do not test leaks.
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

from bench_container import NUM_ENVS, SECONDS, SEED, THREADS, WARMUP

BLOCKS = 8
PATTERNS = ("acdb", "cabd", "bdca", "dbac")
CASE_ORDER_SEED = 20261003


def make_plan(roots, python):
    """Describe the fixed, balanced trial in the existing paired-summary schema."""
    cases = [
        {
            "env": f"DummyPlayers{players}",
            "n": NUM_ENVS,
            "batch": NUM_ENVS,
            "threads": THREADS,
            "max_num_players": players,
        }
        for players in (1, 4)
    ]
    return {
        "purpose": "Detect overhead of Container-output ownership cleanup",
        "blocks": BLOCKS,
        "seconds": SECONDS,
        "warmup": WARMUP,
        "seed": SEED,
        "affinities": ["default"],
        "case_count": len(cases),
        "cases": cases,
        "expected_samples": BLOCKS * len(cases) * 4,
        "paired_replicates": True,
        "orders": list(PATTERNS),
        "case_order_seed": CASE_ORDER_SEED,
        "roots": roots,
        "python": python,
        "response_unit": "environment response, including auto-reset",
        "interpretation": "Normal conversion throughput only; no RSS/leak claim",
    }


def commands(plan):
    """Yield each serial fresh-process command with its exact block metadata."""
    rng = random.Random(plan["case_order_seed"])
    bench = Path(__file__).with_name("bench_container.py").resolve()
    for block in range(plan["blocks"]):
        cases = list(plan["cases"])
        rng.shuffle(cases)
        order = list(PATTERNS[block % len(PATTERNS)])
        for case in cases:
            for slot, label in enumerate(order):
                command = [
                    plan["python"],
                    str(bench),
                    "--variant",
                    "original" if label in "ab" else "candidate",
                    "--envpool-root",
                    plan["roots"][label],
                    "--max-num-players",
                    str(case["max_num_players"]),
                    "--seconds",
                    str(plan["seconds"]),
                    "--warmup",
                    str(plan["warmup"]),
                    "--rep",
                    str(block),
                ]
                yield {
                    "block": block,
                    "slot": slot,
                    "label": label,
                    "order": order,
                    "env": case["env"],
                    "command": command,
                }


def validate_sample(row, item, plan):
    """Reject a child result that differs from the requested measurement."""
    case = next(case for case in plan["cases"] if case["env"] == item["env"])
    expected = {
        "env": case["env"],
        "num_envs": case["n"],
        "batch_size": case["batch"],
        "threads": case["threads"],
        "max_num_players": case["max_num_players"],
        "warmup": plan["warmup"],
        "requested_seconds": plan["seconds"],
        "seed": plan["seed"],
        "rep": item["block"],
        "pin": False,
        "async": False,
        "runtime_selection": "explicit-root",
        "runtime_root": plan["roots"][item["label"]],
    }
    if any(row[key] != value for key, value in expected.items()):
        raise RuntimeError(
            "child returned a different measurement configuration"
        )
    if not (
        row["environment_responses"]
        == row["env_steps"]
        == row["vector_calls"] * case["batch"]
    ):
        raise RuntimeError(
            "child returned inconsistent environment-response counts"
        )


def main():
    """Save a predeclared plan, then run all subprocesses serially or dry-run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Do not resolve an interpreter symlink: its venv-relative path matters.
    python = os.path.abspath(args.python.expanduser())
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        parser.error("--python must name an executable interpreter")
    baseline = args.baseline_root.expanduser().resolve()
    candidate = args.candidate_root.expanduser().resolve()
    for root in (baseline, candidate):
        if not (root / "envpool/__init__.py").is_file():
            parser.error(f"runtime root lacks envpool/__init__.py: {root}")
    roots = {
        label: str(baseline if label in "ab" else candidate) for label in "abcd"
    }
    plan = make_plan(roots, python)
    if args.dry_run:
        print(json.dumps({"plan": plan}))
        for item in commands(plan):
            print(json.dumps(item))
        return

    output = args.out.expanduser().resolve()
    plan_path = (
        args.plan_out.expanduser().resolve()
        if args.plan_out
        else output.with_name(output.name + ".plan.json")
    )
    if output == plan_path or output.exists() or plan_path.exists():
        parser.error("output and plan must be distinct, new files")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    # A failed run leaves its plan and partial raw data for diagnosis. The
    # existing summarizer rejects incomplete trials instead of pooling them.
    with plan_path.open("x") as destination:
        json.dump(plan, destination, indent=2)
        destination.write("\n")
    hashes = {}
    with (
        tempfile.TemporaryDirectory(prefix="envpool-container-") as cwd,
        output.open("x", buffering=1) as destination,
    ):
        for item in commands(plan):
            started = time.time()
            result = subprocess.run(
                item["command"],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result.returncode:
                raise RuntimeError((
                    item,
                    result.returncode,
                    result.stdout,
                    result.stderr,
                ))
            row = json.loads(result.stdout.strip().splitlines()[-1])
            validate_sample(row, item, plan)
            root = roots[item["label"]]
            measured_hash = row["binary_sha256"]
            if hashes.setdefault(root, measured_hash) != measured_hash:
                raise RuntimeError("Dummy binary changed within a runtime root")
            row.update(
                variant=item["label"],
                block=item["block"],
                slot=item["slot"],
                order=item["order"],
                started_unix=started,
                expected_samples=plan["expected_samples"],
            )
            destination.write(json.dumps(row) + "\n")
            print(
                f"block={item['block']} slot={item['slot']} {item['env']} "
                f"{item['label']} {row['env_steps_per_s']:.0f} resp/s "
                f"CPU={row['cpu_seconds'] / row['seconds']:.2f} "
                f"sched_wait={row['scheduler_wait_seconds']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
