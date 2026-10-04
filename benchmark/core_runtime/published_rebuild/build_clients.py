# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Rebuild three public EnvPool units, reusing read-only external libraries.

This focused rebuild is not a clean Bazel/dependency rebuild. Source must be a
clean checkout of the pinned public commit. Source deps/env.sh before invoking
this script when it supplies system includes: the environment is inherited.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

COMMIT = "ce1c47f238a069454f732a70089d1b9857dcc6de"
TREE = "7521d1b10ca2f53d6098d0c8d24667a6c5f59358"
TEMPLATES = Path(
    "benchmark/core_runtime/state_tuple/results/build-commands.json"
)
CLIENTS = {
    "classic": "classic/classic_control_envpool.commands.json",
    "dummy": "candidate-dummy/commands.json",
}
OLD_ARCHIVE = (
    "$BAZEL_EXECROOT/bazel-out/k8-opt/bin/envpool/classic_control/"
    "libclassic_control_env.pic.a"
)
SEARCH_ENV = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "LD_LIBRARY_PATH",
)


def file_record(path):
    """Hash an existing file, retaining its spelling and resolved location."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "resolved": str(path.resolve(strict=True)),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def check_project_path(path, source):
    """Reject project includes and symlinks referring to another checkout."""
    path = Path(path)
    resolved = path.resolve()
    for candidate in (path, resolved):
        project_path = "envpool" in candidate.parts or (
            candidate.is_dir() and (candidate / "envpool").is_dir()
        )
        if project_path and not resolved.is_relative_to(source):
            raise ValueError(f"stale EnvPool path: {path} -> {resolved}")


def remap(command, roots):
    """Expand only audited placeholders; reject unexpanded variables."""
    result = []
    for token in command:
        if not isinstance(token, str):
            raise ValueError("command arguments must be strings")
        for name, root in roots.items():
            token = token.replace(f"${name}", str(root))
        if "$" in token:
            raise ValueError(f"unresolved placeholder: {token}")
        result.append(token)
    return result


def source_identity(source):
    """Require the exact, clean public source revision and tree."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    identity = {
        "path": str(source),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git(
            "status", "--porcelain", "--untracked-files=all"
        ),
    }
    if (identity["commit"], identity["tree"]) != (COMMIT, TREE):
        raise ValueError("source does not match the pinned public commit/tree")
    if identity["status_porcelain"]:
        raise ValueError("source must be clean, including untracked files")
    return identity


def executable(name):
    """Resolve a required executable without running it."""
    path = shutil.which(str(name))
    if path is None:
        raise FileNotFoundError(f"required executable is missing: {name}")
    return str(Path(path).absolute())


def build_plan(source, execroot, output, compiler=None):
    """Validate inputs and construct a serial plan without creating outputs."""
    source, execroot, output = (
        Path(path).resolve() for path in (source, execroot, output)
    )
    if output.exists():
        raise FileExistsError(f"output must be fresh: {output}")
    if any(output.is_relative_to(root) for root in (source, execroot)):
        raise ValueError("output must be outside source and Bazel execroot")
    if not source.is_dir() or not execroot.is_dir():
        raise FileNotFoundError("source and Bazel execroot must exist")
    templates = json.loads((source / TEMPLATES).read_text())
    roots = {
        "CANDIDATE_SOURCE": source,
        "BAZEL_EXECROOT": execroot,
        "OUTPUT_DIR": output,
    }
    opencv_root = execroot / (
        "bazel-out/k8-opt/bin/external/+repositories+opencv/opencv/include"
    )
    opencv = next(
        (
            opencv_root / version
            for version in ("opencv5", "opencv4")
            if (opencv_root / version / "opencv2/opencv.hpp").is_file()
        ),
        None,
    )
    if opencv is None:
        raise FileNotFoundError(f"missing OpenCV 5/4 headers: {opencv_root}")
    compiler = executable(compiler or templates[CLIENTS["classic"]][0][0])
    archive = output / "build/classic/libclassic_control_env.pic.a"
    units, links, externals = {}, {}, set()
    for name, key in CLIENTS.items():
        commands = templates[key]
        if len(commands) != 2:
            raise ValueError(f"expected one compile and one link: {key}")
        compile_command, link_command = (list(cmd) for cmd in commands)
        if link_command.count(OLD_ARCHIVE) != 1:
            raise ValueError(f"expected exactly one render archive: {key}")
        link_command[link_command.index(OLD_ARCHIVE)] = str(archive)
        compile_command, link_command = (
            remap(cmd, roots) for cmd in (compile_command, link_command)
        )
        compile_command[0] = link_command[0] = compiler
        compile_command.extend(["-I", str(opencv)])
        units[name] = compile_command
        links[name] = link_command
    render = units["classic"].copy()
    render[render.index("-c") + 1] = str(
        source / "envpool/classic_control/render_utils.cc"
    )
    render[render.index("-o") + 1] = str(
        output / "build/classic/render_utils.pic.o"
    )
    units["render_utils"] = render
    for name, command in units.items():
        command.extend(["-MD", "-MF", str(output / f"{name}.d"), "-MT", name])
        if command.count("-c") != 1 or command.count("-o") != 1:
            raise ValueError("expected a single source and output per unit")
        source_file = Path(command[command.index("-c") + 1])
        if not source_file.resolve().is_relative_to(source):
            raise ValueError(
                f"source unit escapes fresh checkout: {source_file}"
            )
        for index, token in enumerate(command[1:], 1):
            if token.startswith("/"):
                path = Path(token)
                if command[index - 1] in ("-o", "-MF"):
                    if not path.is_relative_to(output):
                        raise ValueError(
                            f"compile output escapes output: {path}"
                        )
                    continue
                if not path.exists():
                    raise FileNotFoundError(path)
                check_project_path(path, source)
                if not (
                    path.is_relative_to(source) or path.is_relative_to(execroot)
                ):
                    raise ValueError(f"unrecognized template input: {path}")
    objects = {command[command.index("-o") + 1] for command in units.values()}
    generated = objects | {str(archive)}
    for name, command in links.items():
        if command.count("-o") != 1:
            raise ValueError("expected one output per link")
        for index, token in enumerate(command[1:], 1):
            if not token.startswith("/"):
                continue
            path = Path(token)
            if command[index - 1] == "-o":
                if not path.is_relative_to(output):
                    raise ValueError(f"link output escapes output: {path}")
            elif token not in generated:
                check_project_path(path, source)
                if path.resolve().is_relative_to(source):
                    raise ValueError(
                        f"project link input must be rebuilt: {path}"
                    )
                relative = path.relative_to(execroot)
                if "external" not in relative.parts or path.suffix != ".a":
                    raise ValueError(f"non-external link input: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                externals.add(str(path))
        command.append(f"-Wl,-Map,{output / (name + '.map')}")
    environment = {name: os.environ.get(name) for name in SEARCH_ENV}
    for value in environment.values():
        if value is not None:
            for path in value.split(os.pathsep):
                if not path or not Path(path).is_absolute():
                    raise ValueError("inherited search paths must be absolute")
                if not Path(path).is_dir():
                    raise FileNotFoundError(path)
                check_project_path(Path(path), source)
    return {
        "source": str(source),
        "bazel_execroot": str(execroot),
        "output": str(output),
        "compiler": compiler,
        "compile": units,
        "archive": [
            executable("ar"),
            "rcs",
            str(archive),
            render[render.index("-o") + 1],
        ],
        "link": links,
        "external_libraries": sorted(externals),
        "inherited_search_environment": environment,
    }


def dependencies(depfile, source, required_source):
    """Read GCC -MD evidence, rejecting headers from stale project trees."""
    text = Path(depfile).read_text().replace("\\\n", " ")
    _, separator, prerequisites = text.partition(":")
    if not separator:
        raise ValueError(f"invalid dependency file: {depfile}")
    paths = sorted(set(shlex.split(prerequisites)))
    if str(required_source) not in paths:
        raise ValueError(
            f"source missing from dependency file: {required_source}"
        )
    records = []
    for name in paths:
        path = Path(name)
        if not path.is_absolute():
            raise ValueError(f"non-absolute compiler dependency: {path}")
        check_project_path(path, source)
        record = file_record(path)
        record["project_source"] = path.resolve().is_relative_to(source)
        records.append(record)
    return records


def rebuild(plan):
    """Execute the three units serially and retain commands and provenance."""
    source, output = Path(plan["source"]), Path(plan["output"])
    identity = source_identity(source)
    manifest = {
        "scope": "Focused client rebuild; no clean Bazel/dependency rebuild",
        "status": "running",
        "plan": plan,
        "source": identity,
        "template": file_record(source / TEMPLATES),
        "helper": file_record(Path(__file__).resolve()),
        "compiler": file_record(plan["compiler"]),
        "compiler_version": subprocess.check_output(
            [plan["compiler"], "--version"], text=True
        ),
        "archiver": file_record(plan["archive"][0]),
        "external_libraries": [
            file_record(path) for path in plan["external_libraries"]
        ],
        "dependencies": {},
        "executed": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    for command in plan["compile"].values():
        Path(command[command.index("-o") + 1]).parent.mkdir(
            parents=True, exist_ok=True
        )

    def save():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    def run(name, command):
        print(f"{name}: {shlex.join(command)}", flush=True)
        with (output / f"{name}.log").open("x") as log:
            subprocess.run(
                command,
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest["executed"].append(name)
        save()

    save()
    try:
        for name, command in plan["compile"].items():
            run(f"compile-{name}", command)
            manifest["dependencies"][name] = dependencies(
                output / f"{name}.d",
                source,
                Path(command[command.index("-c") + 1]),
            )
        run("archive-render", plan["archive"])
        for name, command in plan["link"].items():
            run(f"link-{name}", command)
        if identity != source_identity(source):
            raise ValueError("source identity changed during rebuild")
        if manifest["external_libraries"] != [
            file_record(path) for path in plan["external_libraries"]
        ]:
            raise ValueError("reused external libraries changed during rebuild")
        manifest["artifacts"] = [
            file_record(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.suffix in (".o", ".a", ".so", ".d", ".map")
        ]
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()


def main():
    """Require explicit fresh-source, read-only dependency, and output roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bazel-execroot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler")
    args = parser.parse_args()
    rebuild(
        build_plan(args.source, args.bazel_execroot, args.output, args.compiler)
    )


if __name__ == "__main__":
    main()
