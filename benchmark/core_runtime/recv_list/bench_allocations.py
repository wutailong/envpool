# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Emit allocation audit JSON to stdout; keep redirected output outside source."""

import argparse
import json
import platform
import sys

import numpy as np
import recv_list_probe as probe


def main():
    """Compare both conversion paths and emit bounded allocation counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    self_test = probe.interception_self_test()
    if self_test["calls"] != 1 or self_test["requested_bytes"] != 37:
        raise RuntimeError("operator-new interception was not verified")
    report = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "module": probe.__file__,
        "scope": "calling-thread ordinary C++ new/new[] requested bytes",
        "source_construction_counted": False,
        "includes_final_python_list_conversion": True,
        "timing_benchmark": False,
        "interception_self_test": self_test,
        "cases": [],
    }
    for fields in (1, 4, 8, 16, 32):
        records = {
            mode: probe.measure(mode, fields, args.iterations)
            for mode in ("legacy", "direct")
        }
        old, new = records["legacy"], records["direct"]
        records["calls_saved_per_recv"] = (
            old["calls"] - new["calls"]
        ) / args.iterations
        records["requested_bytes_saved_per_recv"] = (
            old["requested_bytes"] - new["requested_bytes"]
        ) / args.iterations
        records["checksums_equal"] = old["checksum"] == new["checksum"]
        if not all(record["clean"] for record in (old, new)):
            raise RuntimeError("unreclaimed or untracked conversion allocation")
        if not records["checksums_equal"]:
            raise RuntimeError("conversion checksum mismatch")
        report["cases"].append(records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
