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
"""Run serial, interleaved, independent-process throughput comparisons."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from runtime import VARIANTS, positive_int

MATRICES = {
    "classic": [
        ("CartPole-v1", 1, 1, 1),
        ("CartPole-v1", 20, 20, 1),
        ("CartPole-v1", 64, 64, 1),
        ("CartPole-v1", 256, 256, 1),
        ("CartPole-v1", 256, 256, 4),
        ("CartPole-v1", 1024, 1024, 8),
        ("CartPole-v1", 256, 64, 4),
        ("CartPole-v1", 1024, 256, 8),
    ],
    "heavy": [
        ("HalfCheetah-v4", 20, 20, 1),
        ("HalfCheetah-v4", 64, 64, 4),
        ("HalfCheetah-v4", 256, 64, 4),
        ("HalfCheetah-v4", 256, 256, 8),
    ],
    "confirm": [
        ("CartPole-v1", 20, 20, 1),
        ("CartPole-v1", 256, 256, 1),
        ("CartPole-v1", 256, 256, 4),
        ("CartPole-v1", 1024, 256, 8),
        ("HalfCheetah-v4", 64, 64, 4),
        ("HalfCheetah-v4", 256, 64, 4),
    ],
}


def main() -> None:
    """Launch each selected build without inheriting source-tree import paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Default Python path")
    for variant in VARIANTS:
        parser.add_argument(f"--{variant}-python", help="Override Python path")
        parser.add_argument(
            f"--{variant}-root", help="Parent of this build's envpool package"
        )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matrix", choices=MATRICES, default="classic")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=VARIANTS
    )
    parser.add_argument("--reps", type=positive_int, default=5)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--caller-cpu", type=int, default=0)
    parser.add_argument("--worker-offset", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("duplicate --variants would duplicate samples")
    if args.warmup < 0 or any(
        not math.isfinite(value) or value <= 0
        for value in (args.seconds, args.timeout)
    ):
        parser.error("require warmup >= 0 and finite positive seconds/timeout")
    if args.out.exists() and not args.dry_run:
        parser.error("--out already exists; select a fresh file per experiment")
    interpreters = {
        variant: str(
            Path(getattr(args, f"{variant}_python") or args.python)
            .expanduser()
            .resolve()
        )
        for variant in args.variants
    }
    roots = {
        variant: str(Path(root).expanduser().resolve())
        for variant in args.variants
        if (root := getattr(args, f"{variant}_root"))
    }
    script = Path(__file__).resolve().with_name("bench_core.py")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    # Assets, if needed, must be installed or explicitly configured by the user.
    with tempfile.TemporaryDirectory(prefix="envpool-benchmark-") as cwd:
        output = None if args.dry_run else args.out.open("x", buffering=1)
        try:
            for rep in range(args.reps):
                for name, count, batch, threads in MATRICES[args.matrix]:
                    order = (
                        args.variants if rep % 2 == 0 else args.variants[::-1]
                    )
                    for variant in order:
                        command = [
                            interpreters[variant],
                            str(script),
                            "--variant",
                            variant,
                            "--env",
                            name,
                            "--n",
                            str(count),
                            "--batch",
                            str(batch),
                            "--threads",
                            str(threads),
                            "--seconds",
                            str(args.seconds),
                            "--warmup",
                            str(args.warmup),
                            "--rep",
                            str(rep),
                        ]
                        if variant in roots:
                            command += ["--envpool-root", roots[variant]]
                        if args.pin:
                            command += [
                                "--pin",
                                "--caller-cpu",
                                str(args.caller_cpu),
                                "--worker-offset",
                                str(args.worker_offset),
                            ]
                        if args.dry_run:
                            print(json.dumps(command))
                            continue
                        result = subprocess.run(
                            command,
                            env=environment,
                            cwd=cwd,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                            check=False,
                        )
                        if result.returncode:
                            raise RuntimeError(
                                f"{variant} {name} failed: {result.stderr}"
                            )
                        line = result.stdout.strip().splitlines()[-1]
                        data = json.loads(line)
                        output.write(line + "\n")
                        print(
                            f"{name} N={count} B={batch} T={threads} "
                            f"rep={rep} {variant}: "
                            f"{data['env_steps_per_s']:,.0f} responses/s",
                            flush=True,
                        )
        finally:
            if output is not None:
                output.close()


if __name__ == "__main__":
    main()
