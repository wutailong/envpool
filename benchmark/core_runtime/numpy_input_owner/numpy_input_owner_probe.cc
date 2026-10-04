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

// Include the actual conversion implementation, never a copied substitute.
// Build separate modules against the baseline and fixed headers. The scoped
// operator-new audit requires local ELF symbol binding. Disable that audit
// for sanitizer builds; see README.md for both modes.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "envpool/core/py_envpool.h"

#ifndef NUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT

namespace {

constexpr std::size_t kTrackedPointers = 64;

struct Audit {
  std::size_t attempts = 0;
  std::size_t calls = 0;
  std::size_t array_calls = 0;
  std::size_t bytes = 0;
  std::size_t live = 0;
  std::size_t fail_at = 0;
  bool overflow = false;
  std::array<void*, kTrackedPointers> pointers{};
  std::array<std::size_t, kTrackedPointers> sizes{};
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

void* Allocate(std::size_t bytes, bool array) {
  Audit* audit = active_audit;
  if (audit != nullptr) {
    if (audit->attempts < kTrackedPointers) {
      audit->sizes[audit->attempts] = bytes;
    } else {
      audit->overflow = true;
    }
    if (++audit->attempts == audit->fail_at) throw std::bad_alloc();
  }
  void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
  if (pointer == nullptr) throw std::bad_alloc();
  if (audit != nullptr) {
    ++audit->calls;
    audit->array_calls += array;
    audit->bytes += bytes;
    ++audit->live;
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

#endif  // NUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT

namespace {

void ValidateInput(const py::array& array) {
  if (array.ndim() < 1 || array.ndim() > 3 || array.size() > 512) {
    throw std::invalid_argument(
        "probe requires rank 1-3 and at most 512 items");
  }
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    if (array.shape(axis) < 1) {
      throw std::invalid_argument("probe requires ordinary positive shapes");
    }
  }
}

struct ThreadResult {
  unsigned long ident = 0;  // NOLINT(runtime/int)
  bool started_without_gil = false;
  bool finished_without_gil = false;

  py::dict ToPython() const {
    py::dict result;
    result["thread_ident"] = ident;
    result["started_without_gil"] = started_without_gil;
    result["finished_without_gil"] = finished_without_gil;
    return result;
  }
};

class Owner {
 private:
  std::optional<Array> array_;
  std::weak_ptr<char> observer_;
  bool floating_;

  const Array& Get() const {
    if (!array_) throw std::logic_error("native strong owner was released");
    return *array_;
  }

 public:
  Owner(Array&& array, bool floating)
      : array_(std::move(array)),
        observer_(array_->SharedPtr()),
        floating_(floating) {}

  std::unique_ptr<Owner> Clone() const {
    return std::make_unique<Owner>(Array(Get()), floating_);
  }

  py::array View() const {
    if (floating_) return ArrayToNumpyHelper<double>::Convert(Get());
    return ArrayToNumpyHelper<std::int32_t>::Convert(Get());
  }

  std::uintptr_t Address() const {
    return reinterpret_cast<std::uintptr_t>(Get().Data());
  }

  bool WeakExpired() const { return observer_.expired(); }
  long StrongCount() const { return observer_.use_count(); }  // NOLINT

  bool HasWeakOwner() const {
    const std::weak_ptr<char> empty;
    return observer_.owner_before(empty) || empty.owner_before(observer_);
  }

  py::dict DropStrongNative() {
    Get();
    auto held = std::move(array_);
    array_.reset();
    ThreadResult result;
    {
      // join must not hold the main-thread GIL: the actual custom deleter
      // needs to acquire it to destroy its Python owner on the worker.
      py::gil_scoped_release release;
      std::thread worker([held = std::move(held), &result]() mutable {
        result.ident = PyThread_get_thread_ident();
        result.started_without_gil = !PyGILState_Check();
        held.reset();
        result.finished_without_gil = !PyGILState_Check();
      });
      worker.join();
    }
    return result.ToPython();
  }

  py::dict DropWeakNative() {
    if (array_ || !observer_.expired() || !HasWeakOwner()) {
      throw std::logic_error("release all strong owners before the weak owner");
    }
    auto held = std::move(observer_);
    observer_.reset();
    ThreadResult result;
    {
      py::gil_scoped_release release;
      std::thread worker([held = std::move(held), &result]() mutable {
        result.ident = PyThread_get_thread_ident();
        result.started_without_gil = !PyGILState_Check();
        held.reset();
        result.finished_without_gil = !PyGILState_Check();
      });
      worker.join();
    }
    return result.ToPython();
  }
};

template <typename Dtype>
std::unique_ptr<Owner> Convert(const py::array& array, bool floating) {
  ValidateInput(array);
  return std::make_unique<Owner>(NumpyToArrayIncRef<Dtype>(array), floating);
}

#ifndef NUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT

py::dict AuditResult(const Audit& audit) {
  py::dict result;
  result["attempts"] = audit.attempts;
  result["calls"] = audit.calls;
  result["new_calls"] = audit.calls - audit.array_calls;
  result["new_array_calls"] = audit.array_calls;
  result["requested_bytes"] = audit.bytes;
  result["live"] = audit.live;
  result["overflow"] = audit.overflow;
  result["clean"] = audit.live == 0 && !audit.overflow;
  py::tuple sizes(std::min(audit.attempts, kTrackedPointers));
  for (py::ssize_t i = 0; i < sizes.size(); ++i) sizes[i] = audit.sizes[i];
  result["attempted_sizes"] = sizes;
  return result;
}

py::dict AuditConversion(const py::array& array, std::size_t fail_at) {
  ValidateInput(array);
  using ArrayT = py::array_t<std::int32_t, py::array::c_style>;
  if (!py::isinstance<ArrayT>(array) || !array.writeable()) {
    throw std::invalid_argument("audit requires writable C-contiguous int32");
  }
  if (fail_at > kTrackedPointers) {
    throw std::invalid_argument("failure ordinal exceeds bounded audit");
  }
  {
    // Initialize pybind11/NumPy caches before intercepting native allocations.
    auto warmup = NumpyToArrayIncRef<std::int32_t>(array);
  }
  const auto refcount_before = Py_REFCNT(array.ptr());
  Audit audit;
  audit.fail_at = fail_at;
  bool bad_alloc = false;
  bool other_exception = false;
  {
    AuditScope scope(&audit);
    try {
      auto converted = NumpyToArrayIncRef<std::int32_t>(array);
    } catch (const std::bad_alloc&) {
      bad_alloc = true;
    } catch (...) {
      other_exception = true;
    }
  }
  const auto refcount_after = Py_REFCNT(array.ptr());
  auto result = AuditResult(audit);
  result["fail_at"] = fail_at;
  result["bad_alloc"] = bad_alloc;
  result["other_exception"] = other_exception;
  result["source_refcount_delta"] = refcount_after - refcount_before;
  return result;
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

#else

py::dict AuditConversion(const py::array& /*array*/, std::size_t /*fail_at*/) {
  throw std::runtime_error("allocation audit is disabled in this build");
}

py::dict InterceptionSelfTest() {
  throw std::runtime_error("allocation audit is disabled in this build");
}

#endif  // NUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT

}  // namespace

PYBIND11_MODULE(numpy_input_owner_probe, m) {
#ifdef NUMPY_INPUT_OWNER_DISABLE_ALLOCATION_AUDIT
  m.attr("audit_enabled") = false;
#else
  m.attr("audit_enabled") = true;
#endif
  py::class_<Owner>(m, "Owner")
      .def("clone", &Owner::Clone)
      .def("view", &Owner::View)
      .def_property_readonly("address", &Owner::Address)
      .def_property_readonly("weak_expired", &Owner::WeakExpired)
      .def_property_readonly("has_weak_owner", &Owner::HasWeakOwner)
      .def_property_readonly("strong_count", &Owner::StrongCount)
      .def("drop_strong_native", &Owner::DropStrongNative)
      .def("drop_weak_native", &Owner::DropWeakNative);
  m.def("convert_int32", [](const py::array& array) {
    return Convert<std::int32_t>(array, false);
  });
  m.def("convert_float64",
        [](const py::array& array) { return Convert<double>(array, true); });
  m.def("audit_conversion", &AuditConversion, py::arg("array"),
        py::arg("fail_at") = 0);
  m.def("interception_self_test", &InterceptionSelfTest);
}
