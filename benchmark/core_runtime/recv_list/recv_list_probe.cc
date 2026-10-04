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

// Candidate-header-only probe. See README.md for the required local-symbol
// linker flag: these operator-new replacements must not interpose on Python or
// another extension. No allocation or source-array construction is measured
// until AuditScope is entered. No empty or malformed arrays are constructed.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "envpool/core/py_envpool.h"

namespace {

constexpr std::size_t kMaxOwners = 64;
constexpr std::size_t kTrackedPointers = 4096;

struct Lifetime {
  std::array<int, kMaxOwners> created{};
  std::array<int, kMaxOwners> destroyed{};
  std::array<std::uintptr_t, kMaxOwners> addresses{};
  std::array<std::weak_ptr<char>, kMaxOwners> sources;
  Container<std::int64_t>* container_slots = nullptr;
  std::size_t owners = 0;
  int pools_destroyed = 0;
  int recv_calls = 0;
  bool recv_released_gil = false;

  bool Reclaimed() const {
    for (std::size_t i = 0; i < owners; ++i) {
      if (created[i] != 1 || destroyed[i] != 1) return false;
    }
    return true;
  }
};

struct Audit {
  std::size_t attempts = 0;
  std::size_t calls = 0;
  std::size_t array_calls = 0;
  std::size_t bytes = 0;
  std::size_t live = 0;
  std::size_t peak_live = 0;
  std::size_t fail_at = 0;
  bool overflow = false;
  bool first_retained_at_failure = false;
  bool third_retained_at_failure = false;
  std::size_t moved_containers_at_failure = 0;
  const Lifetime* lifetime = nullptr;
  std::array<void*, kTrackedPointers> pointers{};
};

thread_local Audit* active_audit = nullptr;

class AuditScope {
 public:
  explicit AuditScope(Audit* audit) {
    if (active_audit != nullptr) std::abort();
    active_audit = audit;
  }
  ~AuditScope() { active_audit = nullptr; }
  AuditScope(const AuditScope&) = delete;
  AuditScope& operator=(const AuditScope&) = delete;
};

void NoteFailure(Audit* audit) noexcept {
  const Lifetime* c = audit->lifetime;
  if (c == nullptr) return;
  audit->first_retained_at_failure = c->sources[0].use_count() > 1;
  audit->third_retained_at_failure = c->sources[2].use_count() > 1;
  // The weak owner check prevents reading slots after their typed owner dies.
  if (c->container_slots != nullptr && !c->sources[1].expired()) {
    for (std::size_t i = 0; i < 2; ++i) {
      audit->moved_containers_at_failure += c->container_slots[i] == nullptr;
    }
  }
}

void* Allocate(std::size_t bytes, bool array) {
  Audit* audit = active_audit;
  if (audit != nullptr && ++audit->attempts == audit->fail_at) {
    NoteFailure(audit);
    throw std::bad_alloc();
  }
  void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
  if (pointer == nullptr) throw std::bad_alloc();
  if (audit != nullptr) {
    ++audit->calls;
    audit->array_calls += array;
    audit->bytes += bytes;
    ++audit->live;
    if (audit->live > audit->peak_live) audit->peak_live = audit->live;
    for (void*& slot : audit->pointers) {
      if (slot == nullptr) {
        slot = pointer;
        return pointer;
      }
    }
    audit->overflow = true;
  }
  return pointer;
}

void Release(void* pointer) noexcept {
  if (pointer != nullptr && active_audit != nullptr) {
    for (void*& slot : active_audit->pointers) {
      if (slot == pointer) {
        slot = nullptr;
        --active_audit->live;
        break;
      }
    }
  }
  std::free(pointer);
}

}  // namespace

void* operator new(std::size_t bytes) { return Allocate(bytes, false); }
void* operator new[](std::size_t bytes) { return Allocate(bytes, true); }
void operator delete(void* pointer) noexcept { Release(pointer); }
void operator delete[](void* pointer) noexcept { Release(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { Release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept {
  Release(pointer);
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return Allocate(bytes, false);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return Allocate(bytes, true);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  Release(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  Release(pointer);
}

namespace {

// Alias the actual typed backing owner. A separate Array custom deleter that
// captures that owner would retain it inside its control block while the weak
// source observers survive, even after Array's last strong owner disappears.
struct OwnedArray final : Array {
  OwnedArray(std::shared_ptr<char> owner, const ShapeSpec& spec)
      : Array(std::move(owner), spec.Shape(), spec.element_size) {}
};

template <typename Dtype>
Array MakePrimitive(const std::shared_ptr<Lifetime>& counts, std::size_t owner,
                    const std::vector<int>& shape, int first) {
  const ::Spec<Dtype> spec(shape);
  const auto dimensions = spec.Shape();
  const std::size_t size = Prod(dimensions.data(), dimensions.size());
  auto allocation = std::make_unique<Dtype[]>(size);
  for (std::size_t i = 0; i < size; ++i) {
    allocation[i] = static_cast<Dtype>(first + i);
  }
  ++counts->created[owner];
  std::weak_ptr<Lifetime> observer = counts;
  std::shared_ptr<Dtype> storage(allocation.release(),
                                 [observer, owner](Dtype* p) {
                                   delete[] p;
                                   if (auto counts = observer.lock())
                                     ++counts->destroyed[owner];
                                 });
  counts->addresses[owner] = reinterpret_cast<std::uintptr_t>(storage.get());
  Array array = OwnedArray(
      std::shared_ptr<char>(storage, reinterpret_cast<char*>(storage.get())),
      spec);
  counts->sources[owner] = array.SharedPtr();
  return array;
}

struct MixedFields {
  static auto Specs() {
    return MakeDict("integer"_.Bind(::Spec<std::int32_t>({2, 3})),
                    "container"_.Bind(::Spec<Container<std::int64_t>>(
                        {2}, ::Spec<std::int64_t>({3}))),
                    "float"_.Bind(::Spec<float>({3})),
                    "byte"_.Bind(::Spec<std::uint8_t>({2, 2})),
                    "double"_.Bind(::Spec<double>({1, 2})));
  }

  static std::vector<Array> Arrays(const std::shared_ptr<Lifetime>& counts) {
    counts->owners = 7;
    std::vector<Array> arrays;
    arrays.reserve(5);
    arrays.push_back(MakePrimitive<std::int32_t>(counts, 0, {2, 3}, 10));

    auto slots = std::make_unique<Container<std::int64_t>[]>(2);
    for (std::size_t i = 0; i < 2; ++i) {
      slots[i] = std::make_unique<TArray<std::int64_t>>(
          MakePrimitive<std::int64_t>(counts, 5 + i, {3}, 1000 + 100 * i));
    }
    counts->container_slots = slots.get();
    ++counts->created[1];
    std::weak_ptr<Lifetime> observer = counts;
    std::shared_ptr<Container<std::int64_t>> storage(
        slots.release(), [observer](Container<std::int64_t>* p) {
          delete[] p;
          if (auto counts = observer.lock()) ++counts->destroyed[1];
        });
    const ::Spec<Container<std::int64_t>> outer({2}, ::Spec<std::int64_t>({3}));
    arrays.push_back(OwnedArray(
        std::shared_ptr<char>(storage, reinterpret_cast<char*>(storage.get())),
        outer));
    counts->sources[1] = arrays.back().SharedPtr();

    arrays.push_back(MakePrimitive<float>(counts, 2, {3}, 30));
    arrays.push_back(MakePrimitive<std::uint8_t>(counts, 3, {2, 2}, 40));
    arrays.push_back(MakePrimitive<double>(counts, 4, {1, 2}, 50));
    return arrays;
  }
};

template <std::size_t Fields>
struct PrimitiveFields {
  template <std::size_t... I>
  static auto SpecsImpl(std::index_sequence<I...> /*unused*/) {
    return MakeDict(
        Key<'f', static_cast<char>('A' + I)>{}.Bind(::Spec<int>({2, 3}))...);
  }
  static auto Specs() { return SpecsImpl(std::make_index_sequence<Fields>{}); }

  static std::vector<Array> Arrays(const std::shared_ptr<Lifetime>& counts) {
    counts->owners = Fields;
    std::vector<Array> arrays;
    arrays.reserve(Fields);
    for (std::size_t i = 0; i < Fields; ++i) {
      arrays.push_back(MakePrimitive<int>(counts, i, {2, 3}, 10 * (i + 1)));
    }
    return arrays;
  }
};

// This small, ordinary Dict-backed spec avoids constructing real environments.
// It satisfies PyEnvSpec's existing contract without changing production code.
template <typename Fields>
struct ProbeSpec {
  using Config = decltype(MakeDict("probe"_.Bind(true)));
  using ConfigValues = typename Config::Values;
  using StateSpec = decltype(Fields::Specs());
  using ActionSpec = decltype(MakeDict("action"_.Bind(::Spec<int>({1}))));
  static inline const Config kDefaultConfig = MakeDict("probe"_.Bind(true));
  Config config;
  StateSpec state_spec;
  ActionSpec action_spec;

  explicit ProbeSpec(const ConfigValues& values)
      : config(values),
        state_spec(Fields::Specs()),
        action_spec(MakeDict("action"_.Bind(::Spec<int>({1})))) {}
};

template <typename Fields>
class FakePool {
 public:
  using Spec = ProbeSpec<Fields>;
  using State = NamedVector<typename Spec::StateSpec::Keys, std::vector<Array>>;
  Spec spec;
  std::shared_ptr<Lifetime> counts;
  std::vector<Array> staged;

  explicit FakePool(const Spec& spec) : spec(spec) {}
  ~FakePool() {
    if (counts) ++counts->pools_destroyed;
  }

  void Stage(const std::shared_ptr<Lifetime>& lifetime) {
    counts = lifetime;
    staged = Fields::Arrays(counts);
  }

  std::vector<Array> Recv() {
    ++counts->recv_calls;
    counts->recv_released_gil = PyGILState_Check() == 0;
    // A vector move, with all source allocation outside the audited interval.
    return std::move(staged);
  }
};

template <typename Fields>
class ProbePool : public PyEnvPool<FakePool<Fields>> {
 public:
  using Native = FakePool<Fields>;
  using Base = PyEnvPool<Native>;

  explicit ProbePool(const std::shared_ptr<Lifetime>& counts)
      : Base(typename Base::PySpec(Native::Spec::kDefaultConfig.AllValues())) {
    this->Stage(counts);
  }

  // Preserve the pre-change PyRecv body, including its exact reserve size.
  std::vector<py::array> LegacyPyRecv() {
    std::vector<Array> arr;
    {
      py::gil_scoped_release release;
      arr = Native::Recv();
      DCHECK_EQ(arr.size(), std::tuple_size_v<typename Native::State::Keys>);
    }
    std::vector<py::array> ret;
    ret.reserve(Native::State::kSize);
    ToNumpy(arr, this->py_spec.state_spec, &ret);
    return ret;
  }
};

using MixedPool = ProbePool<MixedFields>;
static_assert(
    std::is_same_v<decltype(std::declval<MixedPool&>().PyRecv()), py::list>,
    "build the probe against the direct-list candidate headers");

enum class Mode { kLegacy, kDirect };

Mode ParseMode(const std::string& name) {
  if (name == "legacy") return Mode::kLegacy;
  if (name == "direct") return Mode::kDirect;
  throw std::invalid_argument("mode must be legacy or direct");
}

template <typename Fields>
py::object PublicConversion(ProbePool<Fields>* pool, Mode mode) {
  if (mode == Mode::kLegacy) {
    // This invokes the same STL vector return caster as the actual Python
    // binding. The final Python list is included, not just native conversion.
    return py::cast(pool->LegacyPyRecv(), py::return_value_policy::move);
  }
  return pool->PyRecv();
}

bool Clean(const Audit& audit) { return !audit.overflow && audit.live == 0; }

py::dict AuditResult(const Audit& audit) {
  py::dict result;
  result["attempts"] = audit.attempts;
  result["calls"] = audit.calls;
  result["new_calls"] = audit.calls - audit.array_calls;
  result["new_array_calls"] = audit.array_calls;
  result["requested_bytes"] = audit.bytes;
  result["peak_live"] = audit.peak_live;
  result["outstanding"] = audit.live;
  result["tracking_overflow"] = audit.overflow;
  result["clean"] = Clean(audit);
  return result;
}

template <typename Fields>
void Warm(Mode mode) {
  for (int i = 0; i < 8; ++i) {
    auto counts = std::make_shared<Lifetime>();
    ProbePool<Fields> pool(counts);
    auto output = PublicConversion(&pool, mode);
  }
}

template <std::size_t Fields>
py::dict MeasurePrimitive(Mode mode, std::size_t iterations) {
  Warm<PrimitiveFields<Fields>>(mode);
  Audit audit;
  std::uint64_t checksum = 0;
  for (std::size_t i = 0; i < iterations; ++i) {
    auto counts = std::make_shared<Lifetime>();
    ProbePool<PrimitiveFields<Fields>> pool(counts);
    {
      AuditScope scope(&audit);
      auto output = PublicConversion(&pool, mode);
      if (!PyList_CheckExact(output.ptr()) ||
          PyList_Size(output.ptr()) != static_cast<py::ssize_t>(Fields)) {
        throw std::logic_error(
            "public conversion did not return the right list");
      }
      for (std::size_t field = 0; field < Fields; ++field) {
        auto array = py::reinterpret_borrow<py::array>(
            PyList_GET_ITEM(output.ptr(), field));
        const auto* data = static_cast<const int*>(array.data());
        checksum += data[0] + data[5];
      }
      // output and its arrays die while this same audit remains active.
    }
    if (!counts->Reclaimed()) {
      throw std::logic_error("primitive source was not reclaimed exactly once");
    }
  }
  auto result = AuditResult(audit);
  result["fields"] = Fields;
  result["iterations"] = iterations;
  result["checksum"] = checksum;
  result["array_handle_bytes"] = sizeof(py::array);
  return result;
}

py::dict Measure(const std::string& name, std::size_t fields,
                 std::size_t iterations) {
  const Mode mode = ParseMode(name);
  if (iterations < 1)
    throw std::invalid_argument("iterations must be positive");
  switch (fields) {
    case 1:
      return MeasurePrimitive<1>(mode, iterations);
    case 4:
      return MeasurePrimitive<4>(mode, iterations);
    case 8:
      return MeasurePrimitive<8>(mode, iterations);
    case 16:
      return MeasurePrimitive<16>(mode, iterations);
    case 32:
      return MeasurePrimitive<32>(mode, iterations);
    default:
      throw std::invalid_argument("fields must be one of 1, 4, 8, 16, 32");
  }
}

py::list FailureSweep(const std::string& name) {
  const Mode mode = ParseMode(name);
  Warm<MixedFields>(mode);
  Audit reference;
  {
    auto counts = std::make_shared<Lifetime>();
    MixedPool pool(counts);
    AuditScope scope(&reference);
    auto output = PublicConversion(&pool, mode);
  }
  if (!Clean(reference) || reference.attempts < 1 || reference.attempts > 512) {
    throw std::logic_error(
        "invalid reference audit or allocation interception");
  }
  py::list trials;
  for (std::size_t ordinal = 1; ordinal <= reference.attempts + 1; ++ordinal) {
    Audit audit;
    audit.fail_at = ordinal;
    auto counts = std::make_shared<Lifetime>();
    audit.lifetime = counts.get();
    bool bad_alloc = false;
    bool other_exception = false;
    {
      MixedPool pool(counts);
      AuditScope scope(&audit);
      try {
        auto output = PublicConversion(&pool, mode);
      } catch (const std::bad_alloc&) {
        bad_alloc = true;
      } catch (...) {
        other_exception = true;
      }
    }
    auto result = AuditResult(audit);
    result["fail_at"] = ordinal;
    result["reference_attempts"] = reference.attempts;
    result["bad_alloc"] = bad_alloc;
    result["other_exception"] = other_exception;
    result["reclaimed_exactly_once"] = counts->Reclaimed();
    result["pools_destroyed"] = counts->pools_destroyed;
    result["first_retained_at_failure"] = audit.first_retained_at_failure;
    result["third_retained_at_failure"] = audit.third_retained_at_failure;
    result["moved_containers_at_failure"] = audit.moved_containers_at_failure;
    trials.append(result);
  }
  return trials;
}

py::dict InterceptionSelfTest() {
  Audit audit;
  // Volatile dispatch prevents replacement-new allocation elision under O3.
  void* (*volatile allocate)(std::size_t) = &::operator new;
  void (*volatile release)(void*) = &::operator delete;
  void* before = allocate(41);
  release(before);
  {
    AuditScope scope(&audit);
    void* pointer = allocate(37);
    release(pointer);
  }
  void* after = allocate(43);
  release(after);
  return AuditResult(audit);
}

template <typename T>
py::tuple OwnerTuple(const std::array<T, kMaxOwners>& values, std::size_t n) {
  py::tuple result(n);
  for (std::size_t i = 0; i < n; ++i) result[i] = values[i];
  return result;
}

}  // namespace

PYBIND11_MODULE(recv_list_probe, m) {
  py::class_<Lifetime, std::shared_ptr<Lifetime>>(m, "Lifetime")
      .def(py::init<>())
      .def_property_readonly(
          "created",
          [](const Lifetime& c) { return OwnerTuple(c.created, c.owners); })
      .def_property_readonly(
          "destroyed",
          [](const Lifetime& c) { return OwnerTuple(c.destroyed, c.owners); })
      .def_property_readonly(
          "addresses",
          [](const Lifetime& c) { return OwnerTuple(c.addresses, c.owners); })
      .def_property_readonly("source_expired",
                             [](const Lifetime& c) {
                               py::tuple result(c.owners);
                               for (std::size_t i = 0; i < c.owners; ++i) {
                                 result[i] = c.sources[i].expired();
                               }
                               return result;
                             })
      .def_readonly("pools_destroyed", &Lifetime::pools_destroyed)
      .def_readonly("recv_calls", &Lifetime::recv_calls)
      .def_readonly("recv_released_gil", &Lifetime::recv_released_gil);
  py::class_<MixedPool>(m, "MixedPool")
      .def(py::init<const std::shared_ptr<Lifetime>&>())
      .def("recv_direct", &MixedPool::PyRecv)
      .def("recv_legacy", &MixedPool::LegacyPyRecv);
  m.def("interception_self_test", &InterceptionSelfTest);
  m.def("measure", &Measure, py::arg("mode"), py::arg("fields"),
        py::arg("iterations") = 1000);
  m.def("failure_sweep", &FailureSweep, py::arg("mode"));
}
