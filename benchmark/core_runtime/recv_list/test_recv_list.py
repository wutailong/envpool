# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Positive-shape contract and allocation tests for the direct-list candidate.

Build recv_list_probe first and put its directory on PYTHONPATH. This test does
not import EnvPool's package or any earlier conversion/empty-array test suite.
"""

import gc
import unittest
import weakref

import numpy as np
import recv_list_probe as probe


def receive(mode):
    """Return converted arrays after destroying their native pool."""
    counts = probe.Lifetime()
    pool = probe.MixedPool(counts)
    result = getattr(pool, f"recv_{mode}")()
    del pool
    return result, counts


def check_output(test, result, counts):
    """Check the mixed-field public contract and retained native owners."""
    test.assertIs(type(result), list)
    test.assertEqual(len(result), 5)
    dtypes = (np.int32, object, np.float32, np.uint8, np.float64)
    shapes = ((2, 3), (2,), (3,), (2, 2), (1, 2))
    for index, (array, dtype, shape) in enumerate(
        zip(result, dtypes, shapes, strict=True)
    ):
        test.assertIs(type(array), np.ndarray)
        test.assertEqual(array.dtype, np.dtype(dtype))
        test.assertEqual(array.shape, shape)
        test.assertTrue(array.flags.c_contiguous)
        test.assertTrue(array.flags.writeable)
        test.assertFalse(array.flags.owndata)
        test.assertIsNotNone(array.base)
        if index != 1:
            first = (index + 1) * 10
            expected = np.arange(first, first + array.size, dtype=dtype)
            np.testing.assert_array_equal(array, expected.reshape(shape))
            test.assertEqual(array.ctypes.data, counts.addresses[index])
    for index, inner in enumerate(result[1]):
        test.assertIs(type(inner), np.ndarray)
        test.assertEqual(inner.dtype, np.dtype(np.int64))
        test.assertEqual(inner.shape, (3,))
        test.assertFalse(inner.flags.owndata)
        test.assertIsNotNone(inner.base)
        test.assertEqual(inner.ctypes.data, counts.addresses[5 + index])
        np.testing.assert_array_equal(
            inner, np.arange(1000 + 100 * index, 1003 + 100 * index)
        )
    test.assertEqual(counts.created, (1,) * 7)
    # Native outer Container storage is already gone; inner owners and primitive
    # owners are kept alive solely by the returned NumPy arrays/capsules.
    test.assertEqual(counts.destroyed, (0, 1, 0, 0, 0, 0, 0))
    test.assertTrue(counts.source_expired[1])
    test.assertEqual(counts.pools_destroyed, 1)
    test.assertEqual(counts.recv_calls, 1)
    test.assertTrue(counts.recv_released_gil)


class RecvListTest(unittest.TestCase):
    """Check direct-list parity, retained storage, and allocation cleanup."""

    def test_actual_bound_recv_contract_and_order(self):
        """Keep list type, ordered fields, and array metadata unchanged."""
        for mode in ("legacy", "direct"):
            with self.subTest(mode=mode):
                result, counts = receive(mode)
                check_output(self, result, counts)
                del result
                gc.collect()
                self.assertEqual(counts.destroyed, (1,) * 7)
                self.assertEqual(counts.source_expired, (True,) * 7)

    def test_exact_legacy_direct_parity(self):
        """Compare all primitive and nested payload bytes exactly."""
        legacy, old_counts = receive("legacy")
        direct, new_counts = receive("direct")
        for index in (0, 2, 3, 4):
            self.assertEqual(legacy[index].dtype, direct[index].dtype)
            self.assertEqual(legacy[index].shape, direct[index].shape)
            self.assertEqual(legacy[index].tobytes(), direct[index].tobytes())
        for index in range(2):
            self.assertEqual(
                legacy[1][index].tobytes(), direct[1][index].tobytes()
            )
        del legacy, direct
        gc.collect()
        self.assertEqual(old_counts.destroyed, (1,) * 7)
        self.assertEqual(new_counts.destroyed, (1,) * 7)

    def test_retained_primitive_and_container_views(self):
        """Keep selected views alive after their other owners are released."""
        for mode in ("legacy", "direct"):
            with self.subTest(mode=mode):
                result, counts = receive(mode)
                primitive = result[0][:, 1:]
                inner = result[1][1][1:]
                outer_ref = weakref.ref(result[1])
                discarded_ref = weakref.ref(result[2])
                del result
                gc.collect()
                self.assertIsNone(outer_ref())
                self.assertIsNone(discarded_ref())
                self.assertEqual(counts.destroyed, (0, 1, 1, 1, 1, 1, 0))
                np.testing.assert_array_equal(primitive, [[11, 12], [14, 15]])
                np.testing.assert_array_equal(inner, [1101, 1102])
                primitive[0, 0] = 91
                inner[0] = 9001
                np.testing.assert_array_equal(primitive, [[91, 12], [14, 15]])
                np.testing.assert_array_equal(inner, [9001, 1102])
                del primitive
                gc.collect()
                self.assertEqual(counts.destroyed, (1, 1, 1, 1, 1, 1, 0))
                del inner
                gc.collect()
                self.assertEqual(counts.destroyed, (1,) * 7)

    def test_operator_new_interception_is_scoped_and_active(self):
        """Count only the explicit allocation inside the audit scope."""
        audit = probe.interception_self_test()
        self.assertEqual(audit["attempts"], 1)
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["new_calls"], 1)
        self.assertEqual(audit["new_array_calls"], 0)
        self.assertEqual(audit["requested_bytes"], 37)
        self.assertTrue(audit["clean"])

    def test_public_conversion_allocation_delta(self):
        """Verify that direct conversion removes exactly one vector allocation."""
        iterations = 8
        for fields in (1, 4, 8, 16, 32):
            with self.subTest(fields=fields):
                legacy = probe.measure("legacy", fields, iterations)
                direct = probe.measure("direct", fields, iterations)
                self.assertTrue(legacy["clean"])
                self.assertTrue(direct["clean"])
                self.assertEqual(legacy["checksum"], direct["checksum"])
                self.assertEqual(legacy["calls"] - direct["calls"], iterations)
                self.assertEqual(
                    legacy["requested_bytes"] - direct["requested_bytes"],
                    fields * direct["array_handle_bytes"] * iterations,
                )

    def test_each_intercepted_failure_reclaims_partial_conversion(self):
        """Reclaim every owner after each injected conversion allocation failure."""
        for mode in ("legacy", "direct"):
            with self.subTest(mode=mode):
                trials = probe.failure_sweep(mode)
                allocation_sites = trials[0]["reference_attempts"]
                self.assertEqual(len(trials), allocation_sites + 1)
                partial_cases = 0
                for trial in trials:
                    with self.subTest(mode=mode, fail_at=trial["fail_at"]):
                        failed = trial["fail_at"] <= allocation_sites
                        self.assertEqual(trial["bad_alloc"], failed)
                        self.assertFalse(trial["other_exception"])
                        self.assertEqual(
                            trial["attempts"],
                            trial["fail_at"] if failed else allocation_sites,
                        )
                        self.assertTrue(trial["clean"])
                        self.assertTrue(trial["reclaimed_exactly_once"])
                        self.assertEqual(trial["pools_destroyed"], 1)
                        partial_cases += (
                            trial["first_retained_at_failure"]
                            and trial["third_retained_at_failure"]
                            and trial["moved_containers_at_failure"] == 2
                        )
                # A trailing primitive has begun converting after both the
                # first primitive and complete Container field were converted.
                self.assertGreater(partial_cases, 0)


if __name__ == "__main__":
    unittest.main()
