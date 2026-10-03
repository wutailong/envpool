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

// Candidate-only probe: the old converter explicitly destroyed source slots,
// so inspecting those slots afterward would be undefined behavior on baseline.
// Build separately as container_conversion_probe with C++17, -fPIC, -shared,
// Python/pybind11 headers, the repository root, Abseil, concurrentqueue,
// ThreadPool, and XLA FFI include paths. Link the same Abseil logging libraries
// and pthread dependencies as the existing Dummy extension. No registration,
// generated environment binding, or production test hook is needed.

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/py_envpool.h"

namespace {

struct Counts {
  std::array<std::atomic<int>, 3> created{{0, 0, 0}};
  std::array<std::atomic<int>, 3> destroyed{{0, 0, 0}};
  std::atomic<int> environments_destroyed{0};
  int checked_null_slots{0};
  int checked_untouched_slots{0};
  std::weak_ptr<char> source_owner;
};

Container<int> MakePayload(const std::shared_ptr<Counts>& counts, int index,
                           bool invalid) {
  // The invalid array owns only one int. No Array allocating constructor, Fill,
  // indexing, or copy is ever applied to its deliberately malformed metadata.
  const ::Spec<int> shape({invalid ? -1 : 3});
  const int length = invalid ? 1 : 3;
  auto allocation = std::make_unique<int[]>(length);
  for (int i = 0; i < length; ++i) {
    allocation[i] = 10 * (index + 1) + i;
  }
  ++counts->created[index];
  std::shared_ptr<int> storage(allocation.release(), [counts, index](int* p) {
    delete[] p;
    ++counts->destroyed[index];
  });
  // The Array's custom deleter keeps the counted allocation alive. Capturing
  // shared ownership also makes the probe's own construction exception-safe.
  Array external(shape, reinterpret_cast<char*>(storage.get()),
                 [storage](char* /*unused*/) {});
  return std::make_unique<TArray<int>>(std::move(external));
}

class ProbeEnvFns {
 public:
  static auto DefaultConfig() {
    return MakeDict("counts"_.Bind(std::shared_ptr<Counts>()),
                    "invalid_second"_.Bind(false));
  }

  template <typename Config>
  static auto StateSpec(const Config& /*conf*/) {
    return MakeDict(
        "obs:dyn"_.Bind(Spec<Container<int>>({3}, Spec<int>({-1}))));
  }

  template <typename Config>
  static auto ActionSpec(const Config& /*conf*/) {
    return MakeDict();
  }
};

using ProbeEnvSpec = EnvSpec<ProbeEnvFns>;

class ProbeEnv : public Env<ProbeEnvSpec> {
 public:
  ProbeEnv(const Spec& spec, int env_id) : Env<ProbeEnvSpec>(spec, env_id) {}

  ~ProbeEnv() override { ++spec_.config["counts"_]->environments_destroyed; }

  bool IsDone() override { return false; }

  void Reset() override {
    auto state = Allocate();
    state["reward"_] = 0.0F;
    for (int i = 0; i < 3; ++i) {
      Container<int>& slot = state["obs:dyn"_][i];
      slot = MakePayload(spec_.config["counts"_], i,
                         i == 1 && spec_.config["invalid_second"_]);
    }
  }
};

py::array Convert(const std::shared_ptr<Counts>& counts, bool invalid_second) {
  // ShapeSpec converts -1 to size_t; pybind11 converts it back to ssize_t.
  // Require that mapping explicitly rather than assuming it on every target.
  static_assert(static_cast<py::ssize_t>(static_cast<std::size_t>(-1)) == -1);
  // For one dimension, pybind11's default stride calculation cannot overflow.
  // NumPy rejects the negative dimension before assigning/accessing data in
  // PyArray_NewFromDescr_int. See the dimension loop and later fa->data = data:
  // https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/ctors.c
  // https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/src/multiarray/ctors.c
  // This exercises partial array-construction failure, not capsule allocation
  // failure, and never depends on exhausting memory.
  auto config = ProbeEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["counts"_] = counts;
  config["invalid_second"_] = invalid_second;
  ProbeEnvSpec spec(config);
  AsyncEnvPool<ProbeEnv> pool(spec);
  TArray<int> ids(::Spec<int>({1}));
  ids[0] = 0;
  pool.Reset(ids);
  auto output = pool.Recv();
  const Array& source = output.back();
  if (source.Shape() != std::vector<std::size_t>({1, 3})) {
    throw std::logic_error("unexpected Container output shape");
  }
  Array alias = source;
  Array truncated = source.Truncate(1);
  counts->source_owner = source.SharedPtr();
  const auto* third_payload =
      static_cast<const Container<int>*>(source.Data())[2].get();
  const auto check_sources = [&](bool partial) {
    const int null_slots = partial ? 2 : 3;
    for (const Array* owner :
         std::array<const Array*, 3>{&source, &alias, &truncated}) {
      const auto* slots = static_cast<const Container<int>*>(owner->Data());
      for (int i = 0; i < null_slots; ++i) {
        if (slots[i] != nullptr) {
          throw std::logic_error("converted source slot was not null");
        }
      }
      if (partial) {
        // The third slot was never visited by conversion. Its original valid
        // payload must survive until the last typed backing owner is released.
        if (slots[2] == nullptr || slots[2].get() != third_payload ||
            slots[2]->Shape() != std::vector<std::size_t>({3}) ||
            counts->destroyed[2].load() != 0) {
          throw std::logic_error("unconverted source payload was not intact");
        }
        const auto* data = static_cast<const int*>(slots[2]->Data());
        for (int i = 0; i < 3; ++i) {
          if (data[i] != 30 + i) {
            throw std::logic_error("unconverted source payload was modified");
          }
        }
      }
    }
    counts->checked_null_slots = null_slots;
    counts->checked_untouched_slots = partial ? 1 : 0;
  };
  try {
    auto result = ArrayToNumpyHelper<Container<int>>::Convert(source);
    check_sources(false);
    return result;
  } catch (const py::error_already_set&) {
    check_sources(true);
    throw;
  }
  // Every source alias, the Recv vector, and the pool die before the caller
  // receives either the returned NumPy array or the conversion exception.
}

}  // namespace

PYBIND11_MODULE(container_conversion_probe, m) {
  py::class_<Counts, std::shared_ptr<Counts>>(m, "Counts")
      .def(py::init<>())
      .def_property_readonly("created",
                             [](const Counts& c) {
                               return py::make_tuple(c.created[0].load(),
                                                     c.created[1].load(),
                                                     c.created[2].load());
                             })
      .def_property_readonly("destroyed",
                             [](const Counts& c) {
                               return py::make_tuple(c.destroyed[0].load(),
                                                     c.destroyed[1].load(),
                                                     c.destroyed[2].load());
                             })
      .def_property_readonly(
          "environments_destroyed",
          [](const Counts& c) { return c.environments_destroyed.load(); })
      .def_readonly("checked_null_slots", &Counts::checked_null_slots)
      .def_readonly("checked_untouched_slots", &Counts::checked_untouched_slots)
      .def_property_readonly("source_released", [](const Counts& c) {
        return c.source_owner.expired();
      });
  m.def("convert", &Convert, py::arg("counts"),
        py::arg("invalid_second") = false);
}
