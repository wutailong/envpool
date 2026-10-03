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
"""Candidate-only Container ownership checks against a separate native probe.

Build conversion_probe.cc as container_conversion_probe for this Python ABI;
its header documents the native include/link dependencies. Put the extension's
directory on PYTHONPATH and run this file with Python (NumPy is required).
Do not run the source-slot assertions against the old explicit-destructor
converter. Existing Dummy Python tests cover baseline output parity.
Only the negative-shape case requires one of the source-audited NumPy versions;
the normal ownership test runs on every version.
"""

import gc
import unittest

import container_conversion_probe as probe
import numpy as np


class ContainerConversionTest(unittest.TestCase):
    """Track real AsyncEnvPool payloads through NumPy ownership transfer."""

    def assert_sources_released(self, counts, partial=False):
        """All native source owners have gone, with live/null slots checked."""
        self.assertEqual(counts.created, (1, 1, 1))
        self.assertEqual(counts.checked_null_slots, 2 if partial else 3)
        self.assertEqual(counts.checked_untouched_slots, 1 if partial else 0)
        self.assertEqual(counts.environments_destroyed, 1)
        self.assertTrue(counts.source_released)

    def test_last_inner_view_owns_payload(self):
        """Inner arrays outlive the pool, output buffer, and outer array."""
        counts = probe.Counts()
        outer = probe.convert(counts)
        self.assert_sources_released(counts)
        self.assertEqual(outer.shape, (1, 3))
        self.assertEqual(outer.dtype, np.dtype(object))
        self.assertEqual(counts.destroyed, (0, 0, 0))

        first = outer[0, 0]
        second = outer[0, 1]
        third = outer[0, 2]
        self.assertEqual(first.dtype, np.dtype(np.intc))
        self.assertEqual(second.dtype, np.dtype(np.intc))
        self.assertEqual(third.dtype, np.dtype(np.intc))
        np.testing.assert_array_equal(first, [10, 11, 12])
        np.testing.assert_array_equal(second, [20, 21, 22])
        np.testing.assert_array_equal(third, [30, 31, 32])
        retained_view = first[1:]
        third_view = third[::2]
        retained_view[0] = 111
        self.assertEqual(first[1], 111)

        del first, third, outer
        gc.collect()
        self.assertEqual(counts.destroyed, (0, 0, 0))
        np.testing.assert_array_equal(retained_view, [111, 12])
        np.testing.assert_array_equal(second, [20, 21, 22])
        np.testing.assert_array_equal(third_view, [30, 32])

        del second
        gc.collect()
        self.assertEqual(counts.destroyed, (0, 1, 0))
        retained_view[1] = 112
        np.testing.assert_array_equal(retained_view, [111, 112])

        del retained_view
        gc.collect()
        self.assertEqual(counts.destroyed, (1, 1, 0))
        third_view[1] = 132
        np.testing.assert_array_equal(third_view, [30, 132])

        del third_view
        gc.collect()
        self.assertEqual(counts.destroyed, (1, 1, 1))
        gc.collect()
        self.assertEqual(counts.destroyed, (1, 1, 1))

    @unittest.skipUnless(
        np.__version__ in {"1.26.4", "2.1.0"},
        "Negative-shape rejection before data access is source-audited only "
        f"for NumPy 1.26.4 and 2.1.0; found {np.__version__}",
    )
    def test_second_element_failure_reclaims_all_payloads(self):
        """Failure frees converted, rejected, and untouched source payloads."""
        counts = probe.Counts()
        with self.assertRaisesRegex(ValueError, "negative dimensions"):
            probe.convert(counts, invalid_second=True)
        gc.collect()
        self.assert_sources_released(counts, partial=True)
        self.assertEqual(counts.destroyed, (1, 1, 1))
        gc.collect()
        self.assertEqual(counts.destroyed, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
