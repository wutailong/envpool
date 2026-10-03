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
"""Serial, isolated rebuild of ToyText/MiniGrid against changed EnvPool core.

Reuse only unchanged external Bazel libraries. Inspect saved dependency files and
recompile every affected EnvPool translation unit, including archive members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run the command-line build or comparison workflow."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline-source", type=Path, required=True)
    p.add_argument("--bazel-execroot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=("toy_text", "minigrid"), required=True)
    a = p.parse_args()
    src, baseline, ex, out = [
        x.resolve()
        for x in (a.source, a.baseline_source, a.bazel_execroot, a.output)
    ]
    out.mkdir(parents=True, exist_ok=True)
    ext, binary = ex / "external", ex / "bazel-out/k8-opt/bin"
    family = binary / "envpool" / a.family
    changed = [
        str(h.relative_to(src))
        for h in sorted((src / "envpool/core").glob("*.h"))
        if h.read_bytes() != (baseline / h.relative_to(src)).read_bytes()
    ]
    includes = [
        src,
        ext / "+repositories+com_google_absl",
        ext / "+repositories+concurrentqueue",
        ext / "+repositories+threadpool",
        ext / "+repositories+pybind11/include",
        ext
        / "rules_python++python+python_3_12_x86_64-unknown-linux-gnu/include/python3.12",
        binary
        / "external/+repositories+openxla_ffi_headers/_virtual_includes/headers",
        next(
            (binary / "external/+repositories+opencv/opencv/include").glob(
                "opencv*"
            )
        ),
    ]
    flags = [
        "/usr/bin/g++",
        "-std=c++17",
        "-O3",
        "-g0",
        "-DNDEBUG",
        "-fPIC",
        "-fvisibility=hidden",
        "-fno-omit-frame-pointer",
        "-ffunction-sections",
        "-fdata-sections",
        "-fstack-protector",
        "-U_FORTIFY_SOURCE",
        "-D_FORTIFY_SOURCE=1",
        "-pthread",
    ]
    for i in includes:
        flags.extend(("-I", str(i)))
    evidence = {
        "family": a.family,
        "source": str(src),
        "baseline_source": str(baseline),
        "changed_headers": {h: sha256(src / h) for h in changed},
        "dependency_audit": [],
        "commands": [],
        "reused_external_libraries": {},
        "outputs": {},
    }
    replacements = {}

    def run(command: list[str]) -> None:
        print("RUN", " ".join(command), flush=True)
        start = time.time()
        subprocess.run(command, check=True, cwd=out)
        evidence["commands"].append({
            "argv": command,
            "seconds": round(time.time() - start, 3),
        })
        (out / "build-evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )

    for dependency in sorted((family / "_objs").glob("*/*.d")):
        tokens = dependency.read_text().replace("\\\n", " ").split()
        old_object = tokens[0].rstrip(":")
        source_rel = next(
            v
            for v in tokens[1:]
            if v.startswith("envpool/") and v.endswith(".cc")
        )
        affected = sorted(set(changed).intersection(tokens))
        row = {
            "dependency_file": str(dependency),
            "dependency_sha256": sha256(dependency),
            "source": source_rel,
            "source_sha256": sha256(src / source_rel),
            "changed_dependencies": affected,
            "rebuilt": bool(affected),
        }
        evidence["dependency_audit"].append(row)
        if affected:
            obj = out / Path(old_object).name
            dep = out / (obj.name + ".d")
            run(
                flags
                + [
                    "-MMD",
                    "-MF",
                    str(dep),
                    "-c",
                    str(src / source_rel),
                    "-o",
                    str(obj),
                ]
            )
            replacements[old_object] = str(obj)
            # Newly generated dependency evidence must reference candidate core.
            assert any(str(src / h) in dep.read_text() for h in affected)
            row["new_dependency_file"] = str(dep)
    for params in sorted(family.glob("*.a-0.params")):
        args = params.read_text().splitlines()
        archive = out / Path(args[1]).name
        members = [replacements.get(v, str(ex / v)) for v in args[2:]]
        run(["/usr/bin/ar", args[0], str(archive), *members])
        replacements[args[1]] = str(archive)
    name = a.family + "_envpool.so"
    params = (family / (name + "-0.params")).read_text().splitlines()
    so = out / name
    command = ["/usr/bin/g++"]
    skip = False
    for v in params:
        if skip:
            skip = False
            continue
        if v == "-o":
            command.extend(("-o", str(so)))
            skip = True
        elif v in replacements:
            command.append(replacements[v])
        elif v.startswith(("bazel-out/", "external/")):
            file = ex / v
            if v.endswith(".a"):
                assert "/external/" in v, f"Unaccounted EnvPool archive: {v}"
                evidence["reused_external_libraries"][str(file)] = sha256(file)
            command.append(str(file))
        else:
            command.append(v)
    run(command)
    evidence["outputs"][name] = sha256(so)
    evidence["translation_units_rebuilt"] = sum(
        row["rebuilt"] for row in evidence["dependency_audit"]
    )
    (out / "build-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        json.dumps({
            "family": a.family,
            "rebuilt": evidence["translation_units_rebuilt"],
            "sha256": evidence["outputs"][name],
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
