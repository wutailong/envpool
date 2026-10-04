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

// Standalone C++17 allocation audit of the existing public AsyncEnvPool APIs.
// Build this same source against each revision, with the same compiler/options
// and without LTO. Dependencies: core headers, ThreadPool.h, concurrentqueue,
// Abseil logging/check libraries and pthreads. No environment assets or Python.
//
// Only the caller's selected Send/Reset invocation is counted. Input owning
// buffers, the const Action, and every moved vector are staged before its
// thread-local scope. Construction, initial resets, warmup, Recv, checks,
// printing, destruction, worker threads and state-buffer background threads are
// excluded. Ordinary global new/new[] replacements pass through to malloc;
// they only record when this thread has an active scope. Nothrow forms are
// included. Direct malloc, aligned allocation, frees, live bytes, allocator
// overhead, RSS and timing are not measured. All actual array dimensions are
// positive; this is not an empty-input, failure-injection or lifetime audit.
//
// These are complete public-call totals, not attribution by allocation site.
// Compare matching records across revisions to establish the intended removal
// of one ActionSlice-vector allocation (submitted_envs * sizeof(ActionSlice)
// requested bytes). Other public-call allocations, including the typed Send's
// AllValues conversion and Reset's TArray copy, intentionally remain counted.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

#include "envpool/core/async_envpool.h"

namespace {

struct Sample {
  std::size_t calls = 0;
  std::size_t array_calls = 0;
  std::size_t bytes = 0;
};

thread_local Sample* active_sample = nullptr;

class Scope {
 public:
  explicit Scope(Sample* sample) {
    if (active_sample != nullptr) std::abort();
    active_sample = sample;
  }
  ~Scope() { active_sample = nullptr; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

void* Allocate(std::size_t bytes, bool array) {
  void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
  if (pointer == nullptr) throw std::bad_alloc();
  if (active_sample != nullptr) {
    ++active_sample->calls;
    active_sample->array_calls += array;
    active_sample->bytes += bytes;
  }
  return pointer;
}

}  // namespace

void* operator new(std::size_t bytes) { return Allocate(bytes, false); }
void* operator new[](std::size_t bytes) { return Allocate(bytes, true); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
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
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

namespace {

constexpr int kWarmups = 8;
constexpr int kIterations = 128;
constexpr int kRepeats = 3;
constexpr int kResetValue = -17;
constexpr std::uint64_t kHashSeed = UINT64_C(14695981039346656037);
volatile std::uint64_t checksum_sink = 0;

void Require(bool condition, const char* detail) {
  if (!condition) {
    active_sample = nullptr;
    std::fprintf(stderr, "generated_enqueue allocation_probe: %s\n", detail);
    std::abort();
  }
}

template <typename T>
T* Data(const Array& array) {
  return static_cast<T*>(array.Data());
}

std::uint64_t Hash(std::uint64_t hash, std::uint64_t value) {
  return (hash ^ value) * UINT64_C(1099511628211);
}

int ActionValue(int sequence, int env_id) {
  return 1000 * (sequence + 1) + env_id + 1;
}

class ProbeFns {
 public:
  static auto DefaultConfig() { return MakeDict(); }
  template <typename Config>
  static auto StateSpec(const Config&) {
    return MakeDict("obs"_.Bind(Spec<int>({3})));
  }
  template <typename Config>
  static auto ActionSpec(const Config&) {
    return MakeDict("action"_.Bind(Spec<int>({-1})));
  }
};

using ProbeSpec = EnvSpec<ProbeFns>;

// A deterministic, nonterminal single-player Env, using the real EnvStep,
// ParseAction, Allocate, PostProcess and AsyncEnvPool worker/queue protocols.
class ProbeEnv : public Env<ProbeSpec> {
  int step_ = 0;

  void Publish(int value) {
    auto state = Allocate();
    state["reward"_] = 0.0F;
    auto* obs = Data<int>(state["obs"_]);
    obs[0] = env_id_;
    obs[1] = value;
    obs[2] = step_;
  }

 public:
  ProbeEnv(const ProbeSpec& spec, int env_id) : Env<ProbeSpec>(spec, env_id) {}
  bool IsDone() override { return false; }
  void Reset() override {
    step_ = 0;
    Publish(kResetValue);
  }
  void Step(const Action& action) override {
    Require(active_sample == nullptr, "worker entered caller counting scope");
    Require(action["env_id"_].size == 1 &&
                Data<int>(action["env_id"_])[0] == env_id_,
            "worker received wrong env ID");
    Require(action["players.env_id"_].size == 1 &&
                Data<int>(action["players.env_id"_])[0] == env_id_ &&
                action["action"_].size == 1,
            "worker received wrong player ID or action size");
    ++step_;
    Publish(Data<int>(action["action"_])[0]);
  }
};

using Pool = AsyncEnvPool<ProbeEnv>;

struct Config {
  int num_envs;
  int batch_size;
  int num_threads;
};

enum class Operation { kConstVector, kMovedVector, kConstAction, kReset };

const char* Name(Operation operation) {
  switch (operation) {
    case Operation::kConstVector:
      return "Send(vector_const_ref)";
    case Operation::kMovedVector:
      return "Send(vector_rvalue)";
    case Operation::kConstAction:
      return "Send(Action_const_ref)";
    case Operation::kReset:
      return "Reset(Array_const_ref)";
  }
  std::abort();
}

std::vector<Array> MakeInput(int num_envs) {
  std::vector<Array> arrays;
  arrays.reserve(3);
  for (int field = 0; field < 3; ++field) {
    arrays.emplace_back(Spec<int>({num_envs}));
  }
  // A nonidentity ordering tests the synchronous queue's order payload too.
  for (int row = 0; row < num_envs; ++row) {
    const int id = num_envs - row - 1;
    Data<int>(arrays[0])[row] = id;
    Data<int>(arrays[1])[row] = id;
    Data<int>(arrays[2])[row] = ActionValue(0, id);
  }
  return arrays;
}

Sample Submit(Pool& pool, Operation operation, const std::vector<Array>& source,
              std::vector<Array>& moved, const Pool::Action& action,
              bool measured) {
  // Dispatch choice and all argument construction precede each active scope.
  // Volatile member pointers keep the actual public overload as the call site.
  Sample sample;
  switch (operation) {
    case Operation::kConstVector: {
      using Method = void (Pool::*)(const std::vector<Array>&);
      Method volatile method = static_cast<Method>(&Pool::Send);
      Scope scope(measured ? &sample : nullptr);
      (pool.*method)(source);
      break;
    }
    case Operation::kMovedVector: {
      using Method = void (Pool::*)(std::vector<Array>&&);
      Method volatile method = static_cast<Method>(&Pool::Send);
      Scope scope(measured ? &sample : nullptr);
      (pool.*method)(std::move(moved));
      break;
    }
    case Operation::kConstAction: {
      using Method = void (Pool::*)(const Pool::Action&);
      Method volatile method = static_cast<Method>(&Pool::Send);
      Scope scope(measured ? &sample : nullptr);
      (pool.*method)(action);
      break;
    }
    case Operation::kReset: {
      using Method = void (Pool::*)(const Array&);
      Method volatile method = &Pool::Reset;
      const Array& env_ids = source[0];
      Scope scope(measured ? &sample : nullptr);
      (pool.*method)(env_ids);
      break;
    }
  }
  return sample;
}

std::uint64_t ReceiveAndCheck(Pool& pool, const Config& config,
                              std::vector<unsigned char>& seen, int sequence,
                              bool reset) {
  Require(active_sample == nullptr, "Recv is inside the counting scope");
  std::fill(seen.begin(), seen.end(), 0);
  std::uint64_t checksum = 0;
  int received = 0;
  while (received < config.num_envs) {
    const Pool::State state(pool.Recv());
    const auto& ids = state["info:env_id"_];
    const auto& obs = state["obs"_];
    Require(ids.ndim == 1 && ids.size > 0 &&
                ids.size <= static_cast<std::size_t>(config.batch_size) &&
                obs.ndim == 2 && obs.Shape(0) == ids.size && obs.Shape(1) == 3,
            "received invalid state shape");
    Require(state["info:players.env_id"_].size == ids.size &&
                state["elapsed_step"_].size == ids.size &&
                state["done"_].size == ids.size &&
                state["reward"_].size == ids.size &&
                state["discount"_].size == ids.size &&
                state["step_type"_].size == ids.size &&
                state["trunc"_].size == ids.size,
            "received invalid common state size");
    const int expected_step = reset ? 0 : sequence + 1;
    for (std::size_t row = 0; row < ids.size; ++row) {
      const int id = Data<int>(ids)[row];
      Require(id >= 0 && id < config.num_envs && seen[id] == 0,
              "received duplicate or out-of-range env ID");
      seen[id] = 1;
      if (config.batch_size == config.num_envs) {
        Require(id == config.num_envs - received - 1,
                "synchronous state order differs from submitted order");
      }
      const int expected_value =
          reset ? kResetValue : ActionValue(sequence, id);
      Require(Data<int>(obs)[3 * row] == id &&
                  Data<int>(obs)[3 * row + 1] == expected_value &&
                  Data<int>(obs)[3 * row + 2] == expected_step &&
                  Data<int>(state["info:players.env_id"_])[row] == id &&
                  Data<int>(state["elapsed_step"_])[row] == expected_step &&
                  !Data<bool>(state["done"_])[row] &&
                  Data<float>(state["reward"_])[row] == 0.0F &&
                  Data<float>(state["discount"_])[row] == 1.0F &&
                  Data<int>(state["step_type"_])[row] == (reset ? 0 : 1) &&
                  !Data<bool>(state["trunc"_])[row],
              "received state differs from submitted action or reset");
      // Commutative across rows: asynchronous completion order may differ.
      checksum +=
          Hash(Hash(Hash(kHashSeed, id), expected_value), expected_step);
      ++received;
    }
  }
  Require(received == config.num_envs &&
              std::all_of(seen.begin(), seen.end(),
                          [](unsigned char value) { return value == 1; }),
          "did not consume exactly one state per submitted environment");
  checksum_sink = checksum;
  return checksum;
}

void RunCase(const Config& config, Operation operation) {
  auto values = ProbeSpec::kDefaultConfig;
  values["num_envs"_] = config.num_envs;
  values["batch_size"_] = config.batch_size;
  values["num_threads"_] = config.num_threads;
  values["max_num_players"_] = 1;
  const ProbeSpec spec(values);
  Pool pool(spec);
  auto source = MakeInput(config.num_envs);
  const Pool::Action action(source);
  std::vector<unsigned char> seen(config.num_envs);

  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    pool.Reset(source[0]);
    ReceiveAndCheck(pool, config, seen, 0, true);
    Sample total, minimum, maximum;
    std::uint64_t checksum = 0;
    for (int sequence = 0; sequence < kWarmups + kIterations; ++sequence) {
      for (int row = 0; row < config.num_envs; ++row) {
        const int id = Data<int>(source[0])[row];
        Data<int>(source[2])[row] = ActionValue(sequence, id);
      }
      // The rvalue overload consumes this vector. Rebuild it before counting;
      // copying source Array metadata and owning storage is not Send's cost.
      std::vector<Array> moved;
      if (operation == Operation::kMovedVector) moved = source;
      const bool measured = sequence >= kWarmups;
      const Sample sample =
          Submit(pool, operation, source, moved, action, measured);
      const auto observed = ReceiveAndCheck(pool, config, seen, sequence,
                                            operation == Operation::kReset);
      if (!measured) continue;
      checksum += observed;
      total.calls += sample.calls;
      total.array_calls += sample.array_calls;
      total.bytes += sample.bytes;
      if (sequence == kWarmups) minimum = maximum = sample;
      minimum.calls = std::min(minimum.calls, sample.calls);
      maximum.calls = std::max(maximum.calls, sample.calls);
      minimum.bytes = std::min(minimum.bytes, sample.bytes);
      maximum.bytes = std::max(maximum.bytes, sample.bytes);
    }
    std::printf(
        "{\"operation\":\"%s\",\"num_envs\":%d,\"batch_size\":%d,"
        "\"num_threads\":%d,\"submitted_envs\":%d,\"repeat\":%d,"
        "\"initial_resets_excluded\":1,\"warmup_calls_excluded\":%d,"
        "\"measured_calls\":%d,\"allocation_calls\":%zu,\"new_calls\":%zu,"
        "\"new_array_calls\":%zu,\"requested_bytes\":%zu,"
        "\"min_calls_per_operation\":%zu,\"max_calls_per_operation\":%zu,"
        "\"min_bytes_per_operation\":%zu,\"max_bytes_per_operation\":%zu,"
        "\"validated_states\":%d,\"checksum\":\"%016" PRIx64 "\"}\n",
        Name(operation), config.num_envs, config.batch_size, config.num_threads,
        config.num_envs, repeat, kWarmups, kIterations, total.calls,
        total.calls - total.array_calls, total.array_calls, total.bytes,
        minimum.calls, maximum.calls, minimum.bytes, maximum.bytes,
        kIterations * config.num_envs, checksum);
  }
}

}  // namespace

int main() {
  std::printf(
      "{\"audit\":\"generated_enqueue\",\"measurement\":\"public_Send_or_"
      "Reset\","
      "\"counter\":\"thread_local_ordinary_new_and_new_array\","
      "\"metric\":\"requested_allocation_traffic\",\"timing\":false,"
      "\"action_slice_bytes\":%zu,\"repeats\":%d,"
      "\"fresh_pool_per_operation_config\":true,"
      "\"excluded\":[\"setup\",\"input_staging\",\"initial_reset\",\"warmup\","
      "\"Recv\",\"validation\",\"printing\",\"destruction\",\"worker_threads\","
      "\"state_buffer_background_threads\",\"malloc\",\"aligned_allocation\"]}"
      "\n",
      sizeof(Pool::ActionSlice), kRepeats);
  const std::array<Config, 4> configs{
      {{1, 1, 1}, {20, 20, 1}, {256, 256, 4}, {256, 64, 4}}};
  const std::array<Operation, 4> operations{
      {Operation::kConstVector, Operation::kMovedVector,
       Operation::kConstAction, Operation::kReset}};
  for (const Config& config : configs) {
    for (Operation operation : operations) RunCase(config, operation);
  }
  return EXIT_SUCCESS;
}
