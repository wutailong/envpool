# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Synthetic rebuild-plan and provenance tests; never invoke native tools."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_clients as build


class BuildClientsTest(unittest.TestCase):
    """Exercise fresh-source isolation without compiling or linking."""

    def setUp(self):
        """Create the minimum audited-template-shaped source and externals."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.execroot = self.root / "execroot"
        self.output = self.root / "output"
        self.external = (
            self.execroot / "bazel-out/k8-opt/bin/external/dependency/libtest.a"
        )
        self.opencv = self.execroot / (
            "bazel-out/k8-opt/bin/external/+repositories+opencv/opencv/include/opencv5"
        )
        self.write(self.external, "external archive")
        self.write(self.opencv / "opencv2/opencv.hpp", "OpenCV header")
        self.header = self.source / "envpool/core/async_envpool.h"
        self.write(self.header, "core header")
        self.templates = {}
        for name, key in build.CLIENTS.items():
            unit = {
                "classic": "envpool/classic_control/classic_control.cc",
                "dummy": "envpool/dummy/dummy_envpool.cc",
            }[name]
            self.write(self.source / unit, "source")
            self.templates[key] = [
                [
                    "c++",
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                    "-I",
                    "$CANDIDATE_SOURCE",
                    "-c",
                    f"$CANDIDATE_SOURCE/{unit}",
                    "-o",
                    f"$OUTPUT_DIR/build/{name}.o",
                ],
                [
                    "c++",
                    "-shared",
                    "-o",
                    f"$OUTPUT_DIR/build/{name}.so",
                    f"$OUTPUT_DIR/build/{name}.o",
                    build.OLD_ARCHIVE,
                    "$BAZEL_EXECROOT/bazel-out/k8-opt/bin/external/dependency/libtest.a",
                ],
            ]
        self.write(
            self.source / "envpool/classic_control/render_utils.cc",
            "render source",
        )
        self.write_templates()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.tools = patch.object(
            build,
            "executable",
            side_effect=lambda name: f"/tools/{Path(name).name}",
        )
        self.tools.start()
        self.addCleanup(self.tools.stop)

    def write(self, path, text):
        """Create a synthetic input file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_templates(self):
        """Save the current synthetic compile/link templates."""
        self.write(self.source / build.TEMPLATES, json.dumps(self.templates))

    def plan(self):
        """Construct a plan using only synthetic local files."""
        return build.build_plan(self.source, self.execroot, self.output)

    def test_fresh_units_and_both_archives(self):
        """Compile three units and replace the old archive in both links."""
        plan = self.plan()
        self.assertEqual(
            list(plan["compile"]), ["classic", "dummy", "render_utils"]
        )
        self.assertFalse(self.output.exists())
        archive = str(
            self.output / "build/classic/libclassic_control_env.pic.a"
        )
        self.assertEqual(plan["archive"][2], archive)
        for name, command in plan["compile"].items():
            self.assertIn("-MD", command)
            self.assertIn(str(self.output / f"{name}.d"), command)
            self.assertIn(str(self.opencv), command)
            self.assertFalse(any("$" in token for token in command))
        for name, command in plan["link"].items():
            self.assertIn(archive, command)
            self.assertIn(f"-Wl,-Map,{self.output / (name + '.map')}", command)
            self.assertFalse(any("/bin/envpool/" in token for token in command))
        self.assertEqual(plan["external_libraries"], [str(self.external)])

    def test_opencv4_fallback_and_compiler_override(self):
        """Discover OpenCV 4 and use the explicitly selected compiler."""
        self.opencv.rename(self.opencv.with_name("opencv4"))
        plan = build.build_plan(
            self.source, self.execroot, self.output, "clang++"
        )
        self.assertEqual(plan["compiler"], "/tools/clang++")
        self.assertIn(
            str(self.opencv.with_name("opencv4")),
            plan["compile"]["render_utils"],
        )

    def test_existing_output_rejected(self):
        """Refuse even an empty existing output directory."""
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_output_inside_inputs_rejected(self):
        """Never create products inside read-only inputs."""
        for root in (self.source, self.execroot):
            with self.subTest(root=root), self.assertRaises(ValueError):
                build.build_plan(
                    self.source, self.execroot, root / "new-output"
                )

    def test_unresolved_placeholder_rejected(self):
        """Unknown template placeholders cannot silently survive remapping."""
        with self.assertRaisesRegex(ValueError, "unresolved placeholder"):
            build.remap(["$UNKNOWN_SOURCE/envpool/core/env.h"], {})

    def test_missing_library_rejected(self):
        """Fail before building if an external archive is absent."""
        self.external.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_missing_opencv_rejected(self):
        """Fail before building if neither supported include directory exists."""
        (self.opencv / "opencv2/opencv.hpp").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "OpenCV"):
            self.plan()

    def test_stale_envpool_archive_rejected(self):
        """Do not reuse additional project objects from the Bazel tree."""
        stale = "$BAZEL_EXECROOT/bazel-out/k8-opt/bin/envpool/dummy/libdummy.a"
        self.templates[build.CLIENTS["dummy"]][1].append(stale)
        self.write_templates()
        with self.assertRaisesRegex(ValueError, "stale EnvPool"):
            self.plan()

    def test_inherited_cpath_preserved(self):
        """Keep supplied dependency includes instead of clearing CPATH."""
        includes = self.root / "deps/include"
        includes.mkdir(parents=True)
        with patch.dict(os.environ, {"CPATH": str(includes)}):
            self.assertEqual(
                self.plan()["inherited_search_environment"]["CPATH"],
                str(includes),
            )

    def test_stale_cpath_rejected(self):
        """Reject an inherited include root for another EnvPool checkout."""
        stale = self.root / "candidate"
        self.write(stale / "envpool/core/env.h", "stale")
        with patch.dict(os.environ, {"CPATH": str(stale)}):
            with self.assertRaisesRegex(ValueError, "stale EnvPool"):
                self.plan()

    def test_dependency_hashes_and_escaped_spaces(self):
        """Read continued GCC dependencies, retaining exact project hashes."""
        unit = self.source / "envpool/classic_control/classic_control.cc"
        system_header = self.root / "system include/header.h"
        self.write(system_header, "system")
        depfile = self.root / "unit.d"
        escaped = str(system_header).replace(" ", "\\ ")
        depfile.write_text(f"unit: {unit} \\\n {self.header} {escaped}\n")
        records = build.dependencies(depfile, self.source, unit)
        core = next(
            record for record in records if record["path"] == str(self.header)
        )
        self.assertTrue(core["project_source"])
        self.assertEqual(
            core["sha256"], build.file_record(self.header)["sha256"]
        )
        self.assertEqual(len(records), 3)

    def test_dependency_symlink_to_stale_header_rejected(self):
        """A fresh-looking symlink must not conceal a candidate header."""
        stale = self.root / "candidate/envpool/core/async_envpool.h"
        self.write(stale, "stale")
        self.header.unlink()
        self.header.symlink_to(stale)
        depfile = self.root / "unit.d"
        depfile.write_text(f"unit: {self.header}\n")
        with self.assertRaisesRegex(ValueError, "stale EnvPool"):
            build.dependencies(depfile, self.source, self.header)

    def test_dependency_requires_source(self):
        """An incomplete dependency file cannot pass provenance validation."""
        depfile = self.root / "unit.d"
        depfile.write_text("unit: /missing/header.h\n")
        with self.assertRaisesRegex(ValueError, "source missing"):
            build.dependencies(depfile, self.source, self.header)

    def test_source_identity_requires_clean_pinned_tree(self):
        """Check commit, tree, and dirty state using mocked Git responses."""
        for values in (
            ("wrong", build.TREE, ""),
            (build.COMMIT, build.TREE, " M source.cc"),
        ):
            with self.subTest(values=values):
                with patch.object(
                    build.subprocess, "check_output", side_effect=values
                ):
                    with self.assertRaises(ValueError):
                        build.source_identity(self.source)
        with patch.object(
            build.subprocess,
            "check_output",
            side_effect=(build.COMMIT, build.TREE, ""),
        ):
            self.assertEqual(
                build.source_identity(self.source)["commit"], build.COMMIT
            )


if __name__ == "__main__":
    unittest.main()
