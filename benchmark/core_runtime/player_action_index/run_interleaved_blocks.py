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
"""Run one scattered-player case as 32 serial, fresh-process paired samples.

The inherited runner retains its CLI, exclusive file creation, clean child
working directory, subprocess checks, and stable native-hash checks. Only plan
construction, the benchmark script path, and extra row validation are adapted.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "container_ownership")
)

import run_container_blocks as inherited
from bench_interleaved import CASE_NAME, ENV_SEEDS, PLAYER_ORDER

_make_plan = inherited.make_plan
_commands = inherited.commands
_validate_sample = inherited.validate_sample


def make_plan(roots, python):
    """Keep the paired-analysis schema and one synchronized stress case."""
    plan = _make_plan(roots, python)
    case = next(case for case in plan["cases"] if case["max_num_players"] == 4)
    case.update(env=CASE_NAME, player_order=PLAYER_ORDER, env_seed=ENV_SEEDS)
    plan.update(
        purpose="Measure ParseAction with interleaved player action rows",
        cases=[case],
        case_count=1,
        expected_samples=plan["blocks"] * 4,
        interpretation=(
            "Synchronized equal-player stress workload; normal Container "
            "conversion and per-step indexed copies in both variants. "
            "Keep separate from normal-seed or grouped-player samples."
        ),
    )
    return plan


def commands(plan):
    """Reuse exact balanced serial ordering, changing only the child driver."""
    bench = Path(__file__).with_name("bench_interleaved.py").resolve()
    for item in _commands(plan):
        item["command"][1] = str(bench)
        yield item


def validate_sample(row, item, plan):
    """Check inherited configuration, synchronized seeds, and warmup scatter."""
    _validate_sample(row, item, plan)
    expected_variant = "original" if item["label"] in "ab" else "candidate"
    if (
        row["player_order"] != PLAYER_ORDER
        or row["env_seed"] != ENV_SEEDS
        or row["variant"] != expected_variant
        or row["warmup_player_counts"] != [1, 2, 3]
        or not 0 < row["scattered_warmup_steps"] < plan["warmup"]
    ):
        raise RuntimeError("child did not validate synchronized interleaving")


def main():
    """Install scoped hooks while delegating all subprocess/file handling."""
    with (
        patch.object(inherited, "__doc__", __doc__),
        patch.object(inherited, "make_plan", make_plan),
        patch.object(inherited, "commands", commands),
        patch.object(inherited, "validate_sample", validate_sample),
    ):
        inherited.main()


if __name__ == "__main__":
    main()
