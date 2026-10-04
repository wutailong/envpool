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
"""Synthetic provenance guards and an exact training-body AST audit.

No EnvPool import, environment construction, training, or timing is performed.
"""

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HARNESS = Path(__file__).with_name("verify_ppo_parity.py")
spec = importlib.util.spec_from_file_location("verify_ppo_parity", HARNESS)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

# Captured from the original pre-adaptation harness (SHA-256 de3a0c8c...) and
# its unchanged default configuration. No historical checkout is needed.
ORIGINAL_TRAINING_AST_SHA256 = (
    "cac7b263f89ebb047c6eb7a48b8003d166d7fa7effa0f3e81cf640bfca724de2"
)
ORIGINAL_CONFIG_SHA256 = (
    "66d3aef1d59e7b99b390dc4f60f77c1743543c40c97e6322366cb4a71792373f"
)


def metadata(root, native="a", wrapper="b", protocol=True):
    """Construct a record without touching any installed runtime."""
    package = Path(root) / "envpool"
    names = harness.REQUIRED_WRAPPERS
    if protocol:
        names += harness.OPTIONAL_WRAPPERS
    return {
        "envpool_package": str(package / "__init__.py"),
        "native_binary": str(
            package / "classic_control/classic_control_envpool.so"
        ),
        "native_sha256": native * 64,
        "python_wrappers": {
            name: {
                "path": str(
                    package.joinpath(*name.split(".")[1:]).with_suffix(".py")
                ),
                "sha256": wrapper * 64,
            }
            for name in names
        },
    }


class RuntimeGuardTests(unittest.TestCase):
    """Validate native and Python-only identity guards with synthetic metadata."""

    def setUp(self):
        """Prepare independent synthetic fixtures for each test."""
        self.left = metadata("/synthetic/original")
        self.right = metadata("/synthetic/candidate", wrapper="c")

    def test_python_only_accepts_same_native_and_changed_wrapper_contents(self):
        """Accept distinct wrappers that share the identical native implementation."""
        harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_accepts_one_changed_required_wrapper(self):
        """Allow a change to only one required wrapper."""
        self.right = metadata("/synthetic/candidate")
        record = self.right["python_wrappers"][harness.REQUIRED_WRAPPERS[0]]
        record["sha256"] = "d" * 64
        harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_accepts_absent_optional_protocol_on_both_sides(self):
        """Allow the optional module to be absent from both manifests."""
        for record in (self.left, self.right):
            del record["python_wrappers"][harness.OPTIONAL_WRAPPERS[0]]
        harness.verify_runtime_distinction(self.left, self.right, True)

    def test_default_preserves_legacy_records_without_wrapper_metadata(self):
        """Accept old-schema records under the unchanged default native guard."""
        self.right["native_sha256"] = "d" * 64
        for record in (self.left, self.right):
            del record["python_wrappers"]
        harness.verify_runtime_distinction(self.left, self.right)

    def test_default_rejects_equal_native_despite_changed_wrappers(self):
        """Require explicit opt-in before comparing equal native contents."""
        with self.assertRaisesRegex(AssertionError, "same native binary"):
            harness.verify_runtime_distinction(self.left, self.right)

    def test_python_only_rejects_changed_native(self):
        """Reject native changes in Python-only comparisons."""
        self.right["native_sha256"] = "d" * 64
        with self.assertRaisesRegex(AssertionError, "identical native"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_both_modes_reject_same_native_path(self):
        """Preserve the distinct-native-path requirement in both modes."""
        self.right["native_binary"] = self.left["native_binary"]
        for python_only in (False, True):
            with (
                self.subTest(python_only=python_only),
                self.assertRaises(AssertionError),
            ):
                harness.verify_runtime_distinction(
                    self.left, self.right, python_only
                )

    def test_python_only_rejects_same_selected_package(self):
        """Reject repeated use of the same selected package."""
        self.right["envpool_package"] = self.left["envpool_package"]
        with self.assertRaisesRegex(AssertionError, "same EnvPool package"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_equal_wrappers_despite_paths_or_labels(self):
        """Reject path and label differences without changed wrapper contents."""
        self.right = metadata("/synthetic/candidate")
        self.left["label"], self.right["label"] = "original", "candidate"
        with self.assertRaisesRegex(
            AssertionError, "same Python wrapper contents"
        ):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_legacy_missing_wrapper_provenance(self):
        """Require fresh wrapper provenance for Python-only comparisons."""
        del self.right["python_wrappers"]
        with self.assertRaisesRegex(AssertionError, "Missing loaded"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_missing_required_wrapper(self):
        """Reject manifests that omit a required wrapper."""
        del self.right["python_wrappers"][harness.REQUIRED_WRAPPERS[0]]
        with self.assertRaisesRegex(AssertionError, "Missing required"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_different_loaded_module_sets(self):
        """Reject differing module sets instead of treating them as source changes."""
        del self.right["python_wrappers"][harness.OPTIONAL_WRAPPERS[0]]
        with self.assertRaisesRegex(
            AssertionError, "same loaded wrapper-module set"
        ):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_unknown_wrapper_name(self):
        """Reject unexpected module names in the focused wrapper manifest."""
        self.right["python_wrappers"]["unrelated"] = {}
        with self.assertRaisesRegex(AssertionError, "Unexpected loaded"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_wrapper_outside_selected_package(self):
        """Reject a recorded wrapper path outside its selected package."""
        record = self.right["python_wrappers"][harness.REQUIRED_WRAPPERS[0]]
        record["path"] = "/outside/envpool.py"
        with self.assertRaisesRegex(AssertionError, "outside selected package"):
            harness.verify_runtime_distinction(self.left, self.right, True)

    def test_python_only_rejects_malformed_wrapper_digest(self):
        """Reject caller labels and malformed strings used in place of hashes."""
        record = self.right["python_wrappers"][harness.REQUIRED_WRAPPERS[0]]
        record["sha256"] = "claimed-change"
        with self.assertRaisesRegex(AssertionError, "Invalid wrapper SHA-256"):
            harness.verify_runtime_distinction(self.left, self.right, True)


class LoadedManifestTests(unittest.TestCase):
    """Check already-loaded module provenance using temporary text fixtures."""

    def setUp(self):
        """Prepare independent synthetic fixtures for each test."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.package = Path(self.tmp.name) / "envpool"
        self.modules = {}
        for name in harness.REQUIRED_WRAPPERS + harness.OPTIONAL_WRAPPERS:
            path = self.package.joinpath(*name.split(".")[1:]).with_suffix(
                ".py"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic fixture for " + name + "\n")
            self.modules[name] = SimpleNamespace(
                __file__=str(path), __spec__=SimpleNamespace(origin=str(path))
            )

    def test_hashes_actual_loaded_files_including_already_loaded_protocol(self):
        """Hash the real fixture files identified by the loaded module objects."""
        with mock.patch.dict(sys.modules, self.modules):
            records = harness.loaded_wrapper_provenance(self.package)
        self.assertEqual(set(records), set(self.modules))
        for name, module in self.modules.items():
            self.assertEqual(records[name]["path"], module.__file__)
            self.assertEqual(
                records[name]["sha256"],
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            )

    def test_does_not_import_optional_protocol_when_absent(self):
        """Do not import another module solely for provenance collection."""
        self.modules[harness.OPTIONAL_WRAPPERS[0]] = None
        with mock.patch.dict(sys.modules, self.modules):
            records = harness.loaded_wrapper_provenance(self.package)
        self.assertEqual(set(records), set(harness.REQUIRED_WRAPPERS))

    def test_rejects_missing_required_loaded_module(self):
        """Reject a required wrapper that was never loaded."""
        self.modules[harness.REQUIRED_WRAPPERS[0]] = None
        with (
            mock.patch.dict(sys.modules, self.modules),
            self.assertRaisesRegex(
                AssertionError, "Required wrapper was not loaded"
            ),
        ):
            harness.loaded_wrapper_provenance(self.package)

    def test_rejects_loaded_module_from_another_package(self):
        """Reject a loaded module whose file belongs to another package."""
        module = self.modules[harness.REQUIRED_WRAPPERS[0]]
        module.__file__ = str(Path(self.tmp.name) / "other.py")
        with (
            mock.patch.dict(sys.modules, self.modules),
            self.assertRaisesRegex(AssertionError, "outside selected package"),
        ):
            harness.loaded_wrapper_provenance(self.package)

    def test_rejects_file_origin_disagreement(self):
        """Reject disagreement between a loaded module file and import origin."""
        module = self.modules[harness.REQUIRED_WRAPPERS[0]]
        module.__spec__.origin = str(Path(self.tmp.name) / "other.py")
        with (
            mock.patch.dict(sys.modules, self.modules),
            self.assertRaisesRegex(AssertionError, "import origin differs"),
        ):
            harness.loaded_wrapper_provenance(self.package)


def canonical_ast(node):
    """Serialize semantic AST fields independently of source formatting."""
    if isinstance(node, ast.AST):
        return {
            "node": type(node).__name__,
            "fields": {
                key: canonical_ast(value)
                for key, value in ast.iter_fields(node)
                # Empty type parameters were added to AST nodes in Python 3.12.
                if not (key == "type_params" and value == [])
            },
        }
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    return node


def training_ast_audit():
    """Remove exactly two provenance additions and check the original AST."""
    adapted = next(
        node
        for node in ast.parse(HARNESS.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    assignment = next(
        node
        for node in adapted.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "provenance"
            for target in node.targets
        )
    )
    manifest = assignment.value
    indices = [
        i
        for i, key in enumerate(manifest.keys)
        if isinstance(key, ast.Constant) and key.value == "python_wrappers"
    ]
    assert len(indices) == 1
    index = indices[0]
    expected_call = ast.parse(
        "loaded_wrapper_provenance(args.package)", mode="eval"
    ).body
    assert ast.dump(manifest.values[index]) == ast.dump(expected_call)
    del manifest.keys[index]
    del manifest.values[index]
    declared_assert = ast.parse(
        'assert loaded_wrapper_provenance(args.package) == provenance["python_wrappers"]'
    ).body[0]
    indices = [
        i
        for i, node in enumerate(adapted.body)
        if ast.dump(node) == ast.dump(declared_assert)
    ]
    assert len(indices) == 1
    del adapted.body[indices[0]]
    serialized = json.dumps(
        canonical_ast(adapted), sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(serialized).hexdigest() == ORIGINAL_TRAINING_AST_SHA256
    ), "Training AST changed beyond declared provenance insertions"
    assert (
        harness.sha256(HARNESS.with_name("config.json"))
        == ORIGINAL_CONFIG_SHA256
    ), "Default config changed"


class TrainingPreservationTests(unittest.TestCase):
    """Pin the unchanged training semantics and default configuration."""

    def test_training_ast_and_default_config(self):
        """Require the original training AST and byte-identical default config."""
        training_ast_audit()


if __name__ == "__main__":
    unittest.main()
