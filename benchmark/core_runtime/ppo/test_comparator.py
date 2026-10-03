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
"""Check that the parity comparator rejects injected metric/array/tensor drift."""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from verify_ppo_parity import compare, write_json


def main():
    """Verify that controlled metric, array, and tensor changes are rejected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    candidate = results / "candidate"
    controls = []
    for kind in ("metric", "array", "tensor"):
        with tempfile.TemporaryDirectory(
            prefix="negative-", dir=results
        ) as tmp:
            tmp = Path(tmp)
            for path in candidate.iterdir():
                (tmp / path.name).symlink_to(
                    path, target_is_directory=path.is_dir()
                )
            if kind == "metric":
                target = tmp / "metrics.jsonl"
                rows = [
                    json.loads(line) for line in target.read_text().splitlines()
                ]
                rows[0]["losses"]["loss"][0] += 1e-9
                target.unlink()
                target.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            elif kind == "array":
                target = tmp / "train-trace.npz"
                with np.load(target, allow_pickle=False) as loaded:
                    arrays = dict(loaded)
                value = arrays["step/obs_next"]
                value.flat[0] = np.nextafter(value.flat[0], np.float32(np.inf))
                target.unlink()
                np.savez_compressed(target, **arrays)
            else:
                # Materialize only one checkpoint; keep the other local files read-only.
                target = tmp / "checkpoints"
                target.unlink()
                target.mkdir()
                for path in (candidate / "checkpoints").glob("*.pt"):
                    (target / path.name).symlink_to(path)
                target = target / "update-100.pt"
                state = torch.load(
                    target, map_location="cpu", weights_only=False
                )
                value = next(iter(state["optim"]["state"].values()))["exp_avg"]
                value.view(-1)[0] = torch.nextafter(
                    value.view(-1)[0], torch.tensor(float("inf"))
                )
                target.unlink()
                torch.save(state, target)
            report = results / f"negative-{kind}.json"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    compare(
                        argparse.Namespace(
                            original=results / "original",
                            candidate=tmp,
                            output=report,
                        )
                    )
            except SystemExit as failure:
                assert failure.code == 1
            else:
                raise AssertionError(
                    f"Comparator failed to detect injected {kind} drift"
                )
            outcome = json.loads(report.read_text())
            assert not outcome["passed"] and outcome["failures"]
            controls.append({
                "injected_difference": kind,
                "detected": True,
                "failures": outcome["failures"],
            })
    write_json(
        results / "comparator-tests.json",
        {"passed": True, "negative_controls": controls},
    )
    print(json.dumps(controls, indent=2))


if __name__ == "__main__":
    main()
