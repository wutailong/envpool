/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Measurement-only actual-header native conversion probe.
// Every converted Array's last native strong owner is released within its loop
// iteration. Input creation, Python argument conversion/validation, warmup and
// result dictionary construction are outside the measured interval.
#include <time.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "conversion_timing_consumer.h"
#include "envpool/core/py_envpool.h"

static_assert(sizeof(int) == sizeof(std::int32_t),
              "int32 native target required");
namespace {
std::int64_t ClockNs(clockid_t id) {
  timespec stamp{};
  if (clock_gettime(id, &stamp) != 0)
    throw std::runtime_error("clock_gettime failed");
  return static_cast<std::int64_t>(stamp.tv_sec) * 1000000000LL + stamp.tv_nsec;
}
std::int64_t ClockResolutionNs(clockid_t id) {
  timespec stamp{};
  if (clock_getres(id, &stamp) != 0)
    throw std::runtime_error("clock_getres failed");
  return static_cast<std::int64_t>(stamp.tv_sec) * 1000000000LL + stamp.tv_nsec;
}
std::uint64_t ValidateAndExpected(const py::array& input) {
  const auto exact_type = py::module_::import("numpy").attr("ndarray");
  if (reinterpret_cast<PyObject*>(Py_TYPE(input.ptr())) != exact_type.ptr())
    throw py::value_error("input must be an ordinary exact numpy.ndarray");
  if (!input.dtype().is(py::dtype::of<int>()) || !input.writeable() ||
      (input.flags() & py::array::c_style) == 0)
    throw py::value_error(
        "input must be native int32, writable and C-contiguous");
  if (input.ndim() <= 0 || input.size() <= 0)
    throw py::value_error("input must have at least one positive dimension");
  for (py::ssize_t i = 0; i < input.ndim(); ++i)
    if (input.shape(i) <= 0 || input.shape(i) > INT_MAX)
      throw py::value_error("each dimension must be in [1, INT_MAX]");
  if (reinterpret_cast<std::uintptr_t>(input.data()) % alignof(int) != 0)
    throw py::value_error("input data must be aligned for native int32 reads");
  const auto* data = static_cast<const int*>(input.data());
  return static_cast<std::uint64_t>(input.size()) * 1000003ULL +
         static_cast<std::uint64_t>(input.ndim()) * 10007ULL +
         static_cast<std::uint32_t>(data[0]) * 101ULL +
         static_cast<std::uint32_t>(data[input.size() - 1]) * 1009ULL;
}

template <bool ConvertEachIteration>
py::dict Measure(const py::array& input, std::uint64_t iterations,
                 std::uint64_t warmup) {
  if (iterations == 0) throw py::value_error("iterations must be positive");
  if (!PyGILState_Check())
    throw std::runtime_error("calling-thread GIL required");
  const std::uint64_t per_iteration = ValidateAndExpected(input);
  const auto refs_before = Py_REFCNT(input.ptr());
  std::optional<Array> prebuilt;
  if constexpr (!ConvertEachIteration)
    prebuilt.emplace(NumpyToArrayIncRef<int>(input));
  // Separate compile-time instantiations: no conversion/calibration mode branch
  // inside either hot loop. The consumer definition is in a different TU.
  auto loop = [&](std::uint64_t count) {
    std::uint64_t checksum = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
      if constexpr (ConvertEachIteration) {
        Array converted = NumpyToArrayIncRef<int>(input);
        checksum += ConversionTimingConsume(converted);
      } else {
        checksum += ConversionTimingConsume(*prebuilt);
      }
    }
    return checksum;
  };
  const auto warmup_checksum = loop(warmup);
  if (warmup_checksum != per_iteration * warmup)
    throw std::runtime_error("warmup checksum mismatch");
  // Nested clock order is fixed. Wall excludes the other four clock reads;
  // thread and process intervals include the nested clock-call overhead.
  const auto process_start = ClockNs(CLOCK_PROCESS_CPUTIME_ID);
  const auto thread_start = ClockNs(CLOCK_THREAD_CPUTIME_ID);
  const auto wall_start = ClockNs(CLOCK_MONOTONIC);
  const auto checksum = loop(iterations);
  const auto wall_end = ClockNs(CLOCK_MONOTONIC);
  const auto thread_end = ClockNs(CLOCK_THREAD_CPUTIME_ID);
  const auto process_end = ClockNs(CLOCK_PROCESS_CPUTIME_ID);
  prebuilt
      .reset();  // Calibration-only ownership release is outside all clocks.
  if (checksum != per_iteration * iterations)
    throw std::runtime_error("measured checksum mismatch");
  if (Py_REFCNT(input.ptr()) != refs_before)
    throw std::runtime_error("input reference count changed across the loops");
  if (!PyGILState_Check())
    throw std::runtime_error("calling-thread GIL was lost");
  py::dict result;
  result["mode"] = ConvertEachIteration ? "convert" : "consumer_calibration";
  result["variant"] = CONVERSION_TIMING_VARIANT;
  result["iterations"] = iterations;
  result["warmup"] = warmup;
  result["wall_ns"] = wall_end - wall_start;
  result["thread_cpu_ns"] = thread_end - thread_start;
  result["process_cpu_ns"] = process_end - process_start;
  result["checksum"] = checksum;
  result["expected_checksum"] = per_iteration * iterations;
  result["warmup_checksum"] = warmup_checksum;
  result["expected_warmup_checksum"] = per_iteration * warmup;
  result["checksum_per_iteration"] = per_iteration;
  result["checksum_arithmetic"] = "uint64 modulo 2^64";
  result["input_reference_count_balanced"] = true;
  result["calling_thread_gil_held"] = true;
  result["array_size"] = input.size();
  result["array_ndim"] = input.ndim();
  return result;
}
py::dict Info() {
  py::dict result;
  result["variant"] = CONVERSION_TIMING_VARIANT;
  result["compiler"] = __VERSION__;
  result["pybind_version"] = std::to_string(PYBIND11_VERSION_MAJOR) + "." +
                             std::to_string(PYBIND11_VERSION_MINOR) + "." +
                             std::to_string(PYBIND11_VERSION_MICRO);
  result["sizeof_array"] = sizeof(Array);
  result["sizeof_int"] = sizeof(int);
  result["clock_monotonic_resolution_ns"] = ClockResolutionNs(CLOCK_MONOTONIC);
  result["clock_thread_resolution_ns"] =
      ClockResolutionNs(CLOCK_THREAD_CPUTIME_ID);
  result["clock_process_resolution_ns"] =
      ClockResolutionNs(CLOCK_PROCESS_CPUTIME_ID);
  result["calibration_note"] =
      "Consumer-loop context only; do not blindly subtract";
  result["scope"] =
      "Calling-thread GIL held; no worker or GIL-contention claim";
  return result;
}
}  // namespace
PYBIND11_MODULE(envpool_conversion_timing, module) {
  module.def("measure", &Measure<true>, py::arg("input").noconvert(),
             py::arg("iterations"), py::arg("warmup") = 0);
  module.def("calibrate_consumer", &Measure<false>,
             py::arg("input").noconvert(), py::arg("iterations"),
             py::arg("warmup") = 0);
  module.def("info", &Info);
}
