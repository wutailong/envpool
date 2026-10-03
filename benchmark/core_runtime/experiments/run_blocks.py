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
"""Serial order-balanced frozen-runtime diagnosis for the two reported cases."""

import argparse
import json
import os
import pathlib
import random
import subprocess
import tempfile
import time

p = argparse.ArgumentParser()
p.add_argument("--python", required=True)
p.add_argument("--variant", action="append", required=True)
p.add_argument("--out", required=True)
p.add_argument(
    "--cases", required=True, help="JSON list of env/n/batch/threads objects"
)
p.add_argument("--blocks", type=int, default=4)
p.add_argument("--seconds", type=float, default=4)
p.add_argument("--affinities", nargs="+", default=["default", "pinned"])
p.add_argument("--warmup", type=int, default=400)
p.add_argument("--dry-run", action="store_true")
p.add_argument(
    "--paired-replicates",
    action="store_true",
    help="Four labels: first two share control, last two share treatment; ABBA/BAAB by treatment",
)
a = p.parse_args()
cases_spec = json.loads(pathlib.Path(a.cases).read_text())
assert cases_spec and len({
    (c["env"], c["n"], c["batch"], c["threads"]) for c in cases_spec
}) == len(cases_spec)
roots = dict(v.split("=", 1) for v in a.variant)
assert len(roots) in (2, 4)
roots = {k: str(pathlib.Path(v).resolve()) for k, v in roots.items()}
labels = list(roots)
if a.paired_replicates:
    assert len(labels) == 4
    assert roots[labels[0]] == roots[labels[1]]
    assert roots[labels[2]] == roots[labels[3]]
assert all(
    (pathlib.Path(v) / "envpool/__init__.py").exists() for v in roots.values()
)
assert (
    a.blocks > 0
    and a.seconds > 0
    and a.warmup >= 0
    and set(a.affinities) <= {"default", "pinned"}
)
python = os.path.abspath(os.path.expanduser(a.python))
bench = pathlib.Path(__file__).with_name("bench_runtime.py").resolve()
rng = random.Random(20261003)
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env.update(
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
with tempfile.TemporaryDirectory(prefix="envpool-diagnosis-") as cwd:
    out = None if a.dry_run else open(a.out, "x", buffering=1)
    for block in range(a.blocks):
        cases = [
            (c["env"], c["n"], c["batch"], c["threads"], pin)
            for c in cases_spec
            for pin in a.affinities
        ]
        rng.shuffle(cases)
        for env_id, n, b, t, pin in cases:
            if len(labels) == 2:
                order = (
                    [labels[0], labels[1], labels[1], labels[0]]
                    if block % 2 == 0
                    else [labels[1], labels[0], labels[0], labels[1]]
                )
            elif a.paired_replicates:
                patterns = [
                    [0, 2, 3, 1],
                    [2, 0, 1, 3],
                    [1, 3, 2, 0],
                    [3, 1, 0, 2],
                ]
                order = [labels[i] for i in patterns[block % 4]]
            else:
                patterns = [
                    [0, 1, 3, 2],
                    [1, 2, 0, 3],
                    [2, 3, 1, 0],
                    [3, 0, 2, 1],
                ]
                order = [labels[i] for i in patterns[block % 4]]
            for slot, label in enumerate(order):
                cmd = [
                    python,
                    str(bench),
                    "--variant",
                    "candidate",
                    "--envpool-root",
                    roots[label],
                    "--env",
                    env_id,
                    "--n",
                    str(n),
                    "--batch",
                    str(b),
                    "--threads",
                    str(t),
                    "--seconds",
                    str(a.seconds),
                    "--warmup",
                    str(a.warmup),
                    "--rep",
                    str(block),
                ]
                if pin == "pinned":
                    cmd += ["--pin"]
                if a.dry_run:
                    print(
                        json.dumps({
                            "block": block,
                            "slot": slot,
                            "label": label,
                            "command": cmd,
                        })
                    )
                    continue
                started = time.time()
                r = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                if r.returncode:
                    raise RuntimeError((r.returncode, r.stdout, r.stderr))
                row = json.loads(r.stdout.strip().splitlines()[-1])
                row.update(
                    variant=label,
                    block=block,
                    slot=slot,
                    order=order,
                    started_unix=started,
                    expected_samples=a.blocks
                    * len(a.affinities)
                    * len(cases_spec)
                    * 4,
                )
                out.write(json.dumps(row) + "\n")
                print(
                    f"block={block} slot={slot} {pin} {env_id} N/B/T={n}/{b}/{t} {label} {row['env_steps_per_s']:.0f} resp/s CPU={row['cpu_seconds'] / row['seconds']:.2f} sched_wait={row['scheduler_wait_seconds']:.2f}s",
                    flush=True,
                )
    if out:
        out.close()
