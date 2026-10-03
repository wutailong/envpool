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
"""Prepare an isolated header overlay, or run its test with a parent timeout.

prepare --source-root SOURCE --overlay-root NEW_DIRECTORY
    Copies the actual queue header and inserts one hook at the exact matched
    admission boundary. Compile admission_boundary_test.cc with -I NEW_DIRECTORY
    before -I SOURCE, plus the existing dependency include/link flags. The
    overlay must be used only for this standalone test translation unit.

run --timeout-seconds 30 /absolute/path/to/admission_boundary_test
    A stuck gate, dequeue, or join fails with exit 124. The parent kills and
    reaps the entire test process, rather than detaching live consumer threads.
    Run sanitizer variants the same way, with their normal environment options.

The test gates execution rather than relying on sleep or scheduler timing.
This tests queue admission only; the separate gated Dummy Reset regression
checks that pool destruction waits for active environment work.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HEADER = Path("envpool/core/action_buffer_queue.h")
ANCHOR = """  bool DequeueOrStop(ActionSlice* action, const std::atomic<int>& stop) {
    while (!sem_.wait()) {
    }
    if (stop.load(std::memory_order_acquire) != 0) {
      return false;
    }
    *action = DequeueReady();
    return true;
  }
"""
HOOK = "    ::shutdown_admission_test::AfterAdmission();\n"


def prepare(source_root: Path, overlay_root: Path) -> int:
    """Create one anchored, isolated header copy without touching production."""
    source = (source_root / HEADER).resolve(strict=True)
    if overlay_root.exists() or overlay_root.is_symlink():
        raise FileExistsError("the overlay requires a new directory")
    overlay_root = overlay_root.resolve()
    destination = overlay_root / HEADER
    if destination.resolve() == source:
        raise ValueError("the overlay must not overwrite the production header")
    original = source.read_bytes()
    anchor = ANCHOR.encode()
    if original.count(anchor) != 1:
        raise ValueError("expected exactly one unchanged DequeueOrStop anchor")
    instrumented = original.replace(
        anchor,
        ANCHOR.replace(
            "    *action = DequeueReady();\n",
            HOOK + "    *action = DequeueReady();\n",
        ).encode(),
        1,
    )
    # Fail closed on reuse, including a preexisting directory or symlink.
    overlay_root.mkdir(parents=True, exist_ok=False)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(instrumented)
    print(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "overlay_header": str(destination),
                "overlay_sha256": hashlib.sha256(instrumented).hexdigest(),
                "inserted_hook": HOOK.strip(),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    """Prepare the overlay or run its standalone child with a finite timeout."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--overlay-root", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--timeout-seconds", type=float, default=30)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "prepare":
        return prepare(args.source_root, args.overlay_root)
    if (
        not args.command
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("run requires a command and a finite, positive timeout")
    try:
        result = subprocess.run(
            args.command, timeout=args.timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired:
        print(
            "FAIL: admission boundary test exceeded parent timeout",
            file=sys.stderr,
        )
        return 124
    return (
        result.returncode if result.returncode >= 0 else 128 - result.returncode
    )


if __name__ == "__main__":
    sys.exit(main())
