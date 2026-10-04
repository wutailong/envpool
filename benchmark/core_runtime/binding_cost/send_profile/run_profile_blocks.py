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

"""Fixed same-binary coarse/fine enqueue attribution; no adaptive extension."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3]
DRIVER = HERE / "probe_send.py"
HELPER = SOURCE / "benchmark/core_runtime/binding_cost/probe.py"
ORDERS = ["acdb", "cabd", "bdca", "dbac"]
CASES = ["s1", "s4", "m1", "m4"]


def digest(p):
    """Return the identity of an explicitly selected input file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    """Run exactly the prespecified 64 samples using a frozen runtime."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Already-built diagnostic runtime package root",
    )
    p.add_argument(
        "--native-sha256",
        required=True,
        help="Independently recorded expected native SHA-256",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh output directory outside the repository",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Existing coarse-s1.json and fine-s1.json positive-validation records from this packaged probe",
    )
    p.add_argument(
        "--project-source",
        type=Path,
        required=True,
        help="Frozen diagnostic project source snapshot used to build the selected runtime",
    )
    p.add_argument(
        "--concurrentqueue-source",
        type=Path,
        required=True,
        help="Frozen diagnostic concurrentqueue source snapshot used for that build",
    )
    p.add_argument(
        "--timing-authorized",
        action="store_true",
        help="Explicitly permit exactly the fixed 64 samples",
    )
    a = p.parse_args()
    if not a.timing_authorized:
        p.error("timing requires --timing-authorized")
    runtime = a.root.expanduser().resolve(strict=True)
    output = a.output.expanduser().resolve()
    validation_dir = a.validation_dir.expanduser().resolve(strict=True)
    project_source = a.project_source.expanduser().resolve(strict=True)
    dependency_source = a.concurrentqueue_source.expanduser().resolve(
        strict=True
    )
    assert (
        not output.exists()
        and output.parent.is_dir()
        and not output.is_relative_to(SOURCE)
    )
    assert not any(
        output.is_relative_to(root)
        for root in (runtime, validation_dir, project_source, dependency_source)
    )
    assert project_source.is_dir() and dependency_source.is_dir()
    assert (project_source / "envpool/core/async_envpool.h").is_file() and (
        dependency_source / "lightweightsemaphore.h"
    ).is_file()
    expected = {
        mode: json.loads((validation_dir / f"{mode}-s1.json").read_text())
        for mode in ["coarse", "fine"]
    }
    for mode, row in expected.items():
        assert (
            row["kind"] == "positive-validation"
            and row["mode"] == mode
            and row["case"] == "s1"
            and row["calls"] == 32
        )
        assert (
            row["native_sha256"] == a.native_sha256
            and row["driver_sha256"] == digest(DRIVER)
            and row["helper_sha256"] == digest(HELPER)
        )
        assert (
            row["pool_released"]
            and not row["new_live_tids_after_cleanup"]
            and row["input_buffers_unchanged"]
        )
    assert (
        digest(runtime / "envpool/classic_control/classic_control_envpool.so")
        == a.native_sha256
    )
    hashes = {
        mode: {
            name: item["sha256"]
            for name, item in row["wrapper_modules"].items()
        }
        for mode, row in expected.items()
    }
    assert (
        hashes["coarse"] == hashes["fine"]
        and expected["coarse"]["native_sha256"]
        == expected["fine"]["native_sha256"]
    )
    plan = {
        "purpose": "Nested enqueue attribution, not a production optimization",
        "blocks": 4,
        "orders": ORDERS,
        "cases": CASES,
        "calls": 5000,
        "warmup": 400,
        "seed": 42,
        "case_order_seed": 2026100417,
        "expected_samples": 64,
        "settings": "Default scheduling, same native binary, serialized fresh processes, no Python phase hooks",
        "gate": "Report signal-stage share with caller CPU support and fine-versus-coarse observer disturbance (not v2 OFF overhead). No semaphore replacement or full candidate suite from these measurements alone. No added samples to seek a favorable result.",
        "limitations": [
            "Fine phase sums nest inside parent enqueue; never sum twice",
            "Middle checkpoints extend producer guard occupancy and can alter spin/park and release-request distribution",
            "POSIX post requests are not actual retries, futex calls, or kernel wake events",
            "Fine versus coarse includes code path, clock and counter overhead; no subtraction correction",
            "Coarse legacy enqueue preserves original queue operations; fine legacy enqueue includes fine clocks and recording",
        ],
        "native_sha256": expected["coarse"]["native_sha256"],
        "wrapper_hashes": hashes["coarse"],
        "driver_sha256": digest(DRIVER),
        "helper_sha256": expected["coarse"]["helper_sha256"],
        "runner_sha256": digest(__file__),
    }
    output.mkdir()
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    roots = {"project": project_source, "concurrentqueue": dependency_source}
    source_files = {
        f"{name}/{p.relative_to(root)}": p
        for name, root in roots.items()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    source_hashes = {name: digest(p) for name, p in source_files.items()}
    (output / "source-before.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n"
    )
    env = os.environ.copy()
    for name in ["PYTHONPATH", "LD_PRELOAD", "ENVPOOL_ASSETS_PATH"]:
        env.pop(name, None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    rng = random.Random(plan["case_order_seed"])
    count = 0
    with (output / "samples.jsonl").open("x") as stream:
        for block in range(4):
            cases = CASES.copy()
            rng.shuffle(cases)
            for case in cases:
                for slot, label in enumerate(ORDERS[block]):
                    mode = "coarse" if label in "ab" else "fine"
                    cmd = [
                        sys.executable,
                        "-B",
                        str(DRIVER),
                        "--root",
                        str(runtime),
                        "--native-sha256",
                        plan["native_sha256"],
                        "--mode",
                        mode,
                        "--case",
                        case,
                        "--calls",
                        "5000",
                        "--warmup",
                        "400",
                        "--label",
                        label,
                        "--timing-authorized",
                    ]
                    t = time.time_ns()
                    r = subprocess.run(
                        cmd,
                        cwd=output,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    if r.returncode:
                        (output / f"failure-{count}.txt").write_text(
                            r.stdout + r.stderr
                        )
                        raise RuntimeError((count, r.returncode))
                    row = json.loads(r.stdout)
                    assert row["native_sha256"] == plan["native_sha256"]
                    assert {
                        k: v["sha256"]
                        for k, v in row["wrapper_modules"].items()
                    } == hashes[mode]
                    assert (
                        row["driver_sha256"] == plan["driver_sha256"]
                        and row["helper_sha256"] == plan["helper_sha256"]
                    )
                    assert (
                        row["vector_calls"] == 5000
                        and row["warmup"] == 400
                        and row["input_buffers_unchanged"]
                        and row["pool_released"]
                        and not row["new_live_tids_after_cleanup"]
                    )
                    assert row["scheduler"]["thread_set_stable"]
                    item = {
                        "block": block,
                        "comparison": "coarse_to_fine",
                        "case": case,
                        "mode": mode,
                        "label": label,
                        "slot": slot,
                        "order": ORDERS[block],
                        "started_unix_ns": t,
                        "sample": row,
                        "stderr": r.stderr,
                    }
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    count += 1
                print(f"block {block} {case} complete", flush=True)
    assert (
        count == 64
        and digest(DRIVER) == plan["driver_sha256"]
        and digest(__file__) == plan["runner_sha256"]
    )
    assert source_hashes == {
        f"{name}/{p.relative_to(root)}": digest(p)
        for name, root in roots.items()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    assert (
        digest(runtime / "envpool/classic_control/classic_control_envpool.so")
        == plan["native_sha256"]
    )
    for name, value in hashes["coarse"].items():
        assert (
            digest(runtime / Path(*name.split(".")).with_suffix(".py")) == value
        )
    (output / "complete.json").write_text(
        json.dumps({
            "samples": count,
            "source_native_wrapper_identities_unchanged": True,
            "no_added_samples": True,
        })
        + "\n"
    )
    print("Fixed 64-sample nested enqueue study complete", flush=True)


if __name__ == "__main__":
    main()
