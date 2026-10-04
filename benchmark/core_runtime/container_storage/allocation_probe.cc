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

// Standalone C++17 allocation audit, not a throughput benchmark. Build against
// the repository headers, concurrentqueue, and Abseil logging dependencies.
// Use the same compiler/options for both revisions, without LTO. Volatile
// factory dispatch and a consumed checksum keep the constructed results live.
// Counters cover ordinary global new/new[] only (including nothrow forms), not
// malloc, aligned allocation, allocator overhead, or allocations on other
// threads. These cases have no over-aligned elements or background threads.
// The failure sweep proves cleanup only for the intercepted allocation sites.

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "envpool/core/state_buffer.h"

namespace {

constexpr std::size_t kIterations = 10000;
constexpr std::size_t kSweepLimit = 64;
constexpr std::size_t kTrackedPointers = 64;

struct Audit {
  std::size_t attempts = 0;
  std::size_t calls = 0;
  std::size_t array_calls = 0;
  std::size_t bytes = 0;
  std::size_t live = 0;
  std::size_t peak_live = 0;
  std::size_t fail_at = 0;
  bool overflow = false;
  std::array<void*, kTrackedPointers> pointers{};
};

thread_local Audit* active_audit = nullptr;
volatile std::uint64_t checksum_sink = 0;

class Scope {
 public:
  explicit Scope(Audit& audit) {
    if (active_audit != nullptr) std::abort();
    active_audit = &audit;
  }
  ~Scope() { active_audit = nullptr; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

void* Allocate(std::size_t bytes, bool array) {
  Audit* audit = active_audit;
  if (audit != nullptr && ++audit->attempts == audit->fail_at) {
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
    // Invalidate the audit rather than silently claiming complete tracking.
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

template <typename Dtype>
std::uint64_t MakeAndObserve(StateArrayFactory factory, const ShapeSpec& spec) {
  Array array = factory(spec);
  if (array.ndim != spec.shape.size() || array.element_size != sizeof(Dtype)) {
    std::abort();
  }
  std::size_t expected_size = 1;
  std::uint64_t checksum = 1 + array.ndim + array.element_size;
  for (std::size_t i = 0; i < array.ndim; ++i) {
    const auto dimension = static_cast<std::size_t>(spec.shape[i]);
    if (array.Shape(i) != dimension) std::abort();
    expected_size *= dimension;
    checksum += (i + 1) * array.Shape(i);
  }
  if (array.size != expected_size) std::abort();
  checksum += array.size;
  if constexpr (StateArrayFactoryHelper<Dtype>::kIsContainer) {
    const auto* slots = static_cast<const Dtype*>(array.Data());
    for (std::size_t i = 0; i < array.size; ++i) {
      if (slots[i] != nullptr) std::abort();
      checksum += 1;
    }
  } else {
    // Inspect the zero-initialized byte storage without assuming float object
    // lifetimes in the primitive Array's vector<char> backing allocation.
    const auto* bytes = static_cast<const unsigned char*>(array.Data());
    for (std::size_t i = 0; i < array.size * array.element_size; ++i) {
      if (bytes[i] != 0) std::abort();
      checksum += bytes[i] + 1;
    }
  }
  checksum_sink = checksum;
  return checksum;  // Array destruction occurs while the caller's Scope lives.
}

bool Clean(const Audit& audit) { return !audit.overflow && audit.live == 0; }

template <typename Dtype>
bool RunCase(const char* dtype, const char* name, const ShapeSpec& spec) {
  StateArrayFactory volatile factory = &StateArrayFactoryHelper<Dtype>::Make;
  for (int i = 0; i < 8; ++i) MakeAndObserve<Dtype>(factory, spec);

  Audit measured;
  std::uint64_t checksum = 0;
  {
    Scope scope(measured);
    for (std::size_t i = 0; i < kIterations; ++i) {
      checksum += MakeAndObserve<Dtype>(factory, spec);
    }
  }
  std::printf(
      "{\"kind\":\"allocations\",\"dtype\":\"%s\",\"case\":\"%s\","
      "\"iterations\":%zu,\"alloc_attempts\":%zu,\"alloc_calls\":%zu,"
      "\"new_calls\":%zu,\"new_array_calls\":%zu,\"alloc_bytes\":%zu,"
      "\"peak_live\":%zu,\"outstanding\":%zu,\"tracking_overflow\":%s,"
      "\"checksum\":%" PRIu64 "}\n",
      dtype, name, kIterations, measured.attempts, measured.calls,
      measured.calls - measured.array_calls, measured.array_calls,
      measured.bytes, measured.peak_live, measured.live,
      measured.overflow ? "true" : "false", checksum);
  if (!Clean(measured)) return false;

  // Discover this revision's allocation count; never hard-code the baseline's
  // layout. Every failure ordinal plus one non-failing sentinel is exercised.
  Audit reference;
  {
    Scope scope(reference);
    MakeAndObserve<Dtype>(factory, spec);
  }
  const std::size_t sites = reference.attempts;
  bool passed = Clean(reference) && sites <= kSweepLimit;
  std::size_t failures = 0;
  if (passed) {
    for (std::size_t ordinal = 1; ordinal <= sites + 1; ++ordinal) {
      Audit trial;
      trial.fail_at = ordinal;
      bool bad_alloc = false;
      bool other_exception = false;
      {
        Scope scope(trial);
        try {
          MakeAndObserve<Dtype>(factory, spec);
        } catch (const std::bad_alloc&) {
          bad_alloc = true;
        } catch (...) {
          other_exception = true;
        }
      }
      const bool should_fail = ordinal <= sites;
      const bool clean = Clean(trial);
      const bool expected = !other_exception && bad_alloc == should_fail &&
                            trial.attempts == (should_fail ? ordinal : sites);
      passed = passed && clean && expected;
      failures += bad_alloc;
      std::printf(
          "{\"kind\":\"failure_trial\",\"dtype\":\"%s\",\"case\":\"%s\","
          "\"fail_at\":%zu,\"alloc_attempts\":%zu,\"alloc_calls\":%zu,"
          "\"bad_alloc\":%s,\"other_exception\":%s,\"outstanding\":%zu,"
          "\"tracking_overflow\":%s,\"passed\":%s}\n",
          dtype, name, ordinal, trial.attempts, trial.calls,
          bad_alloc ? "true" : "false", other_exception ? "true" : "false",
          trial.live, trial.overflow ? "true" : "false",
          clean && expected ? "true" : "false");
      if (!passed) break;
    }
  }
  std::printf(
      "{\"kind\":\"failure_sweep\",\"dtype\":\"%s\",\"case\":\"%s\","
      "\"allocation_sites\":%zu,\"injected_failures\":%zu,\"limit\":%zu,"
      "\"passed\":%s}\n",
      dtype, name, sites, failures, kSweepLimit, passed ? "true" : "false");
  return passed;
}

template <typename Dtype>
bool RunType(const char* dtype) {
  static_assert(alignof(Dtype) <= alignof(std::max_align_t));
  // Construct every spec before warmup, counting, or fault injection.
  const std::array<ShapeSpec, 4> specs{{
      {sizeof(Dtype), {}},
      {sizeof(Dtype), {64}},
      {sizeof(Dtype), {64, 3}},
      {sizeof(Dtype), {0, 2}},
  }};
  const std::array<const char*, 4> names{
      {"scalar", "vector", "multidim", "zero"}};
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (!RunCase<Dtype>(dtype, names[i], specs[i])) return false;
  }
  return true;
}

}  // namespace

int main() {
  return RunType<Container<int>>("container_int") &&
                 RunType<Container<float>>("container_float") &&
                 RunType<float>("float")
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
