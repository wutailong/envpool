# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Actual-header input lifetime checks using only ordinary positive arrays.

Build numpy_input_owner_probe and put its directory on PYTHONPATH. These tests
do not import the EnvPool package or an environment. See README.md for running
the intentionally failing baseline check in a separate process.
"""

import gc
import sys
import threading
import unittest
import weakref

import numpy as np
import numpy_input_owner_probe as probe

SHAPES = ((6,), (2, 3), (2, 3, 4))
CONVERTERS = (
    (np.int32, probe.convert_int32),
    (np.float64, probe.convert_float64),
)


def owning_array(shape, dtype):
    """Make a small owning array, so its weakref tracks the payload owner."""
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape).copy()


class NumpyInputOwnerTest(unittest.TestCase):
    """Check rejection cleanup and unchanged accepted-input behavior."""

    def test_readonly_rejection_refcount_stays_stable(self):
        """Reject same-dtype contiguous read-only inputs without retaining them."""
        for dtype, convert in CONVERTERS:
            for shape in SHAPES:
                with self.subTest(dtype=dtype, shape=shape):
                    source = owning_array(shape, dtype)
                    source.flags.writeable = False
                    before = sys.getrefcount(source)
                    for _ in range(32):
                        with self.assertRaisesRegex(ValueError, "writeable"):
                            convert(source)
                    self.assertEqual(sys.getrefcount(source), before)

    def test_readonly_rejection_owner_is_reclaimed(self):
        """Let a rejected Python owner die immediately after repeated attempts."""
        source = owning_array((2, 3), np.int32)
        source.flags.writeable = False
        reclaimed = []
        observer = weakref.ref(source, lambda _: reclaimed.append(True))
        for _ in range(8):
            with self.assertRaisesRegex(ValueError, "writeable"):
                probe.convert_int32(source)
        del source
        gc.collect()
        self.assertIsNone(observer())
        self.assertEqual(reclaimed, [True])

    def test_writable_zero_copy_aliases_and_retains_owner(self):
        """Keep the same pointer and one Python owner through C++ aliases."""
        for dtype, convert in CONVERTERS:
            for shape in SHAPES:
                with self.subTest(dtype=dtype, shape=shape):
                    source = owning_array(shape, dtype)
                    observer = weakref.ref(source)
                    before = sys.getrefcount(source)
                    owner = convert(source)
                    alias = owner.clone()
                    view = alias.view()
                    self.assertEqual(sys.getrefcount(source), before + 1)
                    self.assertEqual(owner.address, source.ctypes.data)
                    self.assertEqual(alias.address, source.ctypes.data)
                    self.assertEqual(view.ctypes.data, source.ctypes.data)
                    self.assertEqual(view.shape, shape)
                    self.assertEqual(view.dtype, np.dtype(dtype))
                    self.assertTrue(view.flags.c_contiguous)
                    self.assertTrue(view.flags.writeable)
                    self.assertTrue(np.shares_memory(source, view))
                    self.assertEqual(owner.strong_count, 3)
                    source.flat[0] = 101
                    self.assertEqual(view.flat[0], 101)
                    view.flat[-1] = 202
                    self.assertEqual(source.flat[-1], 202)
                    del source, owner, alias
                    gc.collect()
                    self.assertIsNotNone(observer())
                    self.assertEqual(view.flat[0], 101)
                    self.assertEqual(view.flat[-1], 202)
                    del view
                    gc.collect()
                    self.assertIsNone(observer())

    def check_copy(self, source, convert, dtype):
        """Check copy payload, independence, and converted-owner lifetime."""
        expected = np.asarray(source, dtype=dtype).copy(order="C")
        before = sys.getrefcount(source)
        observer = weakref.ref(source)
        owner = convert(source)
        view = owner.view()
        self.assertEqual(sys.getrefcount(source), before)
        self.assertNotEqual(owner.address, source.ctypes.data)
        self.assertFalse(np.shares_memory(source, view))
        self.assertEqual(view.shape, source.shape)
        self.assertEqual(view.dtype, np.dtype(dtype))
        self.assertTrue(view.flags.c_contiguous)
        self.assertTrue(view.flags.writeable)
        np.testing.assert_array_equal(view, expected)
        view.flat[0] = 77
        self.assertEqual(source.flat[0], expected.flat[0])
        # The caller still holds source; return only the converted view and a
        # weak observer, with the C++ Owner already destroyed.
        del owner
        return view, observer

    def test_forced_dtype_preserves_copy_semantics(self):
        """Retain a writable converted owner for both writable/read-only input."""
        for readonly in (False, True):
            for dtype, convert in CONVERTERS:
                with self.subTest(readonly=readonly, dtype=dtype):
                    other_dtype = np.float64 if dtype is np.int32 else np.int32
                    source = owning_array((2, 3, 4), other_dtype)
                    source.flags.writeable = not readonly
                    view, observer = self.check_copy(source, convert, dtype)
                    del source
                    gc.collect()
                    self.assertIsNone(observer())
                    self.assertEqual(view.flat[0], 77)
                    self.assertEqual(view.flat[-1], 23)
                    del view

    def test_noncontiguous_preserves_copy_semantics(self):
        """Copy ordinary strided and reversed inputs to C-contiguous storage."""
        for readonly in (False, True):
            for reverse in (False, True):
                with self.subTest(readonly=readonly, reverse=reverse):
                    source = owning_array((4, 12), np.int32)[:, ::2]
                    if reverse:
                        source = source[::-1, ::-1]
                    self.assertFalse(source.flags.c_contiguous)
                    source.flags.writeable = not readonly
                    last = int(source.flat[-1])
                    view, observer = self.check_copy(
                        source, probe.convert_int32, np.int32
                    )
                    del source
                    gc.collect()
                    self.assertIsNone(observer())
                    self.assertEqual(view.flat[0], 77)
                    self.assertEqual(view.flat[-1], last)
                    del view

    def test_native_last_strong_release_precedes_weak_destruction(self):
        """Release Python ownership under the GIL at strong, not weak, death."""
        main_ident = threading.get_ident()
        for dtype, convert in CONVERTERS:
            with self.subTest(dtype=dtype):
                source = owning_array((2, 3), dtype)
                callbacks = []
                observer = weakref.ref(
                    source,
                    lambda _, counts=callbacks: counts.append(
                        threading.get_ident()
                    ),
                )
                owner = convert(source)
                del source
                self.assertIsNotNone(observer())
                self.assertEqual(owner.strong_count, 1)
                self.assertTrue(owner.has_weak_owner)
                self.assertFalse(owner.weak_expired)
                strong = owner.drop_strong_native()
                self.assertTrue(strong["started_without_gil"])
                self.assertTrue(strong["finished_without_gil"])
                self.assertNotEqual(strong["thread_ident"], main_ident)
                self.assertIsNone(observer())
                self.assertEqual(callbacks, [strong["thread_ident"]])
                self.assertTrue(owner.weak_expired)
                self.assertEqual(owner.strong_count, 0)
                self.assertTrue(owner.has_weak_owner)
                weak = owner.drop_weak_native()
                self.assertTrue(weak["started_without_gil"])
                self.assertTrue(weak["finished_without_gil"])
                self.assertNotEqual(weak["thread_ident"], main_ident)
                self.assertFalse(owner.has_weak_owner)
                self.assertEqual(callbacks, [strong["thread_ident"]])


class AllocationAuditTest(unittest.TestCase):
    """Exercise replacement-new auditing in the normal build only."""

    def test_operator_new_interception_is_scoped_and_active(self):
        """Count the one explicit allocation inside the audit scope only."""
        self.assertTrue(probe.audit_enabled)
        audit = probe.interception_self_test()
        self.assertEqual(audit["attempts"], 1)
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["new_calls"], 1)
        self.assertEqual(audit["new_array_calls"], 0)
        self.assertEqual(audit["attempted_sizes"], (37,))
        self.assertEqual(audit["requested_bytes"], 37)
        self.assertTrue(audit["clean"])

    def test_each_conversion_allocation_failure_reclaims_owner(self):
        """Reclaim the wrapper at wrapper, shape, and control-block failures."""
        self.assertTrue(probe.audit_enabled)
        source = owning_array((2, 3, 4), np.int32)
        observer = weakref.ref(source)
        before = sys.getrefcount(source)
        reference = probe.audit_conversion(source)
        self.assertTrue(reference["clean"])
        self.assertFalse(reference["bad_alloc"])
        self.assertFalse(reference["other_exception"])
        self.assertEqual(reference["source_refcount_delta"], 0)
        # ArrayT wrapper, ShapeSpec vector<int>, Array vector<size_t>, and
        # shared_ptr control block. No payload copy is needed for this input.
        self.assertEqual(reference["attempts"], 4)
        for ordinal in range(1, reference["attempts"] + 2):
            with self.subTest(fail_at=ordinal):
                trial = probe.audit_conversion(source, ordinal)
                failed = ordinal <= reference["attempts"]
                self.assertEqual(trial["bad_alloc"], failed)
                self.assertFalse(trial["other_exception"])
                self.assertEqual(trial["attempts"], min(ordinal, 4))
                self.assertEqual(trial["calls"], min(ordinal - 1, 4))
                self.assertEqual(
                    trial["attempted_sizes"],
                    reference["attempted_sizes"][: min(ordinal, 4)],
                )
                self.assertTrue(trial["clean"])
                self.assertEqual(trial["source_refcount_delta"], 0)
                self.assertEqual(sys.getrefcount(source), before)
        del source
        gc.collect()
        self.assertIsNone(observer())


if __name__ == "__main__":
    unittest.main()
