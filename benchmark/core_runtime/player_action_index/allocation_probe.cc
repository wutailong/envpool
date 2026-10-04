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

// Standalone C++17 allocation audit; build with the same compiler/options and
// without LTO against each revision's core/concurrentqueue/Abseil headers.
// Counts ordinary new/new[] requested bytes, including nothrow forms. This is
// allocation traffic, not RSS, live bytes, allocator overhead, or timing.
// Direct malloc and aligned allocation are excluded; these primitive fields
// need no over-alignment. Thread-local counters exclude queue background
// workers. The measured interval is EnvStep entry through Action construction,
// ending at the first statement in Step. Validation, Allocate, PostProcess,
// Wait, setup, the initial reset, ten warmups, and destruction are excluded.
// Each case has fresh Env/queue/input storage and 10,000 measured non-reset
// calls. Zero-player owning storage is deliberately exercised, without special
// cases.

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <vector>

#include "envpool/core/env.h"

namespace {

struct Sample {
  std::size_t calls = 0;
  std::size_t bytes = 0;
};

thread_local bool counting = false;
thread_local Sample current;

void* CountedAllocate(std::size_t bytes) {
  if (counting) {
    ++current.calls;
    current.bytes += bytes;
  }
  if (void* pointer = std::malloc(bytes == 0 ? 1 : bytes)) return pointer;
  throw std::bad_alloc();
}

Sample StopCounting() noexcept {
  const Sample result = current;
  counting = false;
  return result;
}

}  // namespace

void* operator new(std::size_t bytes) { return CountedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return CountedAllocate(bytes); }
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
    return CountedAllocate(bytes);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return CountedAllocate(bytes);
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

constexpr int kEnvId = 3;
constexpr int kEnvIndex = 1;  // Deliberately different from the env ID.
constexpr int kBatchEnvs = 4;
constexpr int kEnvIds[kBatchEnvs] = {4, kEnvId, 0, 2};
constexpr int kWarmups = 10;
constexpr int kIterations = 10000;
constexpr std::uint64_t kHashSeed = UINT64_C(14695981039346656037);

std::uint64_t Hash(std::uint64_t hash, std::uint64_t value) {
  return (hash ^ value) * UINT64_C(1099511628211);
}

struct Case {
  const char* name;
  int max_players;
  int total_rows;
  std::vector<int> expected_rows;  // Explicit source order, allocated up front.
};

void Require(bool condition, const Case& test, const char* detail) {
  if (!condition) {
    counting = false;
    std::fprintf(stderr, "allocation_probe: case=%s: %s\n", test.name, detail);
    std::abort();
  }
}

class ProbeFns {
 public:
  static auto DefaultConfig() { return MakeDict(); }
  template <typename Config>
  static auto StateSpec(const Config&) {
    return MakeDict("obs"_.Bind(Spec<int>({})));
  }
  template <typename Config>
  static auto ActionSpec(const Config&) {
    return MakeDict("shared"_.Bind(Spec<double>({})),
                    "shared_vector"_.Bind(Spec<int>({3})),
                    "player_scalar"_.Bind(Spec<int>({-1})),
                    "player_matrix"_.Bind(Spec<double>({-1, 2, 3})));
  }
};

using ProbeSpec = EnvSpec<ProbeFns>;

struct Input {
  std::shared_ptr<std::vector<Array>> arrays;
  std::uint64_t checksum = kHashSeed;
};

template <typename T>
T* Data(const Array& array) {
  return static_cast<T*>(array.Data());
}

Input MakeInput(const ProbeSpec& spec, const Case& test) {
  Input input{std::make_shared<std::vector<Array>>(), kHashSeed};
  for (auto shape : spec.action_spec.AllValues<ShapeSpec>()) {
    if (!shape.shape.empty() && shape.shape[0] == -1) {
      shape.shape[0] = test.total_rows;
    } else {
      shape = shape.Batch(kBatchEnvs);
    }
    input.arrays->emplace_back(shape);
  }
  auto& arrays = *input.arrays;
  for (int i = 0; i < kBatchEnvs; ++i) {
    Data<int>(arrays[0])[i] = kEnvIds[i];
    Data<double>(arrays[2])[i] = 300 + 7 * i;
    for (int j = 0; j < 3; ++j) {
      Data<int>(arrays[3])[3 * i + j] = 2000 + 100 * i + j;
    }
  }
  for (int i = 0; i < test.total_rows; ++i) {
    Data<int>(arrays[1])[i] = test.max_players == 1 ? kEnvIds[i] : i % 3;
    Data<int>(arrays[4])[i] = 100 + 13 * i;
    for (int j = 0; j < 6; ++j) {
      Data<double>(arrays[5])[6 * i + j] = 1000 + 100 * i + 3 * j;
    }
  }
  Require(
      test.expected_rows.size() <= static_cast<std::size_t>(test.max_players),
      test, "too many expected players");
  int previous = -1;
  for (int row : test.expected_rows) {
    Require(row > previous && row < test.total_rows, test,
            "expected rows must be increasing and in bounds");
    previous = row;
    Data<int>(arrays[1])[row] = kEnvId;
  }
  // Sanity-check the explicit expected indices before any measured work.
  std::size_t found = 0;
  for (int i = 0; i < test.total_rows; ++i) {
    if (Data<int>(arrays[1])[i] == kEnvId) {
      Require(
          found < test.expected_rows.size() && test.expected_rows[found] == i,
          test, "source player list differs from expected rows");
      ++found;
    }
  }
  Require(found == test.expected_rows.size(), test, "missing expected row");
  input.checksum = Hash(input.checksum, Data<int>(arrays[0])[kEnvIndex]);
  input.checksum = Hash(input.checksum, Data<double>(arrays[2])[kEnvIndex]);
  for (int j = 0; j < 3; ++j) {
    input.checksum =
        Hash(input.checksum, Data<int>(arrays[3])[3 * kEnvIndex + j]);
  }
  for (int row : test.expected_rows) {
    input.checksum = Hash(input.checksum, Data<int>(arrays[1])[row]);
    input.checksum = Hash(input.checksum, Data<int>(arrays[4])[row]);
    for (int j = 0; j < 6; ++j) {
      input.checksum =
          Hash(input.checksum, Data<double>(arrays[5])[6 * row + j]);
    }
  }
  return input;
}

class ProbeEnv : public Env<ProbeSpec> {
  const Case& test_;
  const Input& input_;

  template <typename T>
  void CheckShape(const TArray<T>& array,
                  std::initializer_list<std::size_t> shape) const {
    Require(array.ndim == shape.size() &&
                array.Shape().size() == shape.size() &&
                array.element_size == sizeof(T),
            test_, "action rank or element size mismatch");
    std::size_t size = 1;
    std::size_t axis = 0;
    for (std::size_t dimension : shape) {
      Require(array.Shape(axis++) == dimension, test_, "action shape mismatch");
      size *= dimension;
    }
    Require(array.size == size, test_, "action size mismatch");
  }

  void Publish() {
    // One output player, including for empty actions, keeps the protocol
    // simple; all output work is outside the action-allocation measurement.
    auto state = Allocate(1);
    state["reward"_] = 0.0F;
    state["obs"_] = kEnvId;
  }

 public:
  Sample sample;
  int validated_steps = 0;
  std::uint64_t last_checksum = 0;

  ProbeEnv(const ProbeSpec& spec, const Case& test, const Input& input)
      : Env<ProbeSpec>(spec, kEnvId), test_(test), input_(input) {}

  bool IsDone() override { return false; }
  void Reset() override {
    Require(!counting, test_, "reset entered the measured interval");
    Publish();
  }

  void Step(const Action& action) override {
    sample = StopCounting();  // First statement: validation/setup excluded.
    const auto players = test_.expected_rows.size();
    CheckShape(action["env_id"_], {});
    CheckShape(action["players.env_id"_], {players});
    CheckShape(action["shared"_], {});
    CheckShape(action["shared_vector"_], {3});
    CheckShape(action["player_scalar"_], {players});
    CheckShape(action["player_matrix"_], {players, 2, 3});
    const auto& source = *input_.arrays;
    std::uint64_t checksum = kHashSeed;
    const auto check = [&](auto actual, auto expected) {
      Require(actual == expected, test_,
              "action value or source-row order mismatch");
      checksum = Hash(checksum, static_cast<std::uint64_t>(actual));
    };
    check(Data<int>(action["env_id"_])[0], Data<int>(source[0])[kEnvIndex]);
    check(Data<double>(action["shared"_])[0],
          Data<double>(source[2])[kEnvIndex]);
    for (int j = 0; j < 3; ++j) {
      check(Data<int>(action["shared_vector"_])[j],
            Data<int>(source[3])[3 * kEnvIndex + j]);
    }
    for (std::size_t i = 0; i < players; ++i) {
      const int row = test_.expected_rows[i];
      check(Data<int>(action["players.env_id"_])[i], Data<int>(source[1])[row]);
      check(Data<int>(action["player_scalar"_])[i], Data<int>(source[4])[row]);
      for (int j = 0; j < 6; ++j) {
        check(Data<double>(action["player_matrix"_])[6 * i + j],
              Data<double>(source[5])[6 * row + j]);
      }
    }
    Require(checksum == input_.checksum, test_, "action checksum mismatch");
    last_checksum = checksum;
    ++validated_steps;
    Publish();
  }
};

void RunCase(const Case& test) {
  auto config = ProbeSpec::kDefaultConfig;
  config["num_envs"_] = 5;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["max_num_players"_] = test.max_players;
  const ProbeSpec spec(config);
  const Input input = MakeInput(spec, test);
  StateBufferQueue queue(1, 1, test.max_players,
                         spec.state_spec.AllValues<ShapeSpec>());
  ProbeEnv env(spec, test, input);
  env.SetAction(input.arrays, kEnvIndex);
  const auto wait = [&]() {
    Require(!counting, test, "counting leaked past Step entry");
    const auto result = queue.Wait();
    Require(result.size() == 9 && result[0].size == 1 &&
                Data<int>(result[0])[0] == kEnvId &&
                Data<int>(result[8])[0] == kEnvId,
            test, "one-shot state result mismatch");
  };
  env.EnvStep(&queue, 0, true, false);
  wait();
  Sample total, minimum, maximum;
  for (int i = -kWarmups; i < kIterations; ++i) {
    current = {};
    counting = i >= 0;  // Immediately before the actual EnvStep call.
    env.EnvStep(&queue, 0, false, false);
    wait();
    if (i >= 0) {
      total.calls += env.sample.calls;
      total.bytes += env.sample.bytes;
      if (i == 0) minimum = maximum = env.sample;
      minimum.calls = std::min(minimum.calls, env.sample.calls);
      minimum.bytes = std::min(minimum.bytes, env.sample.bytes);
      maximum.calls = std::max(maximum.calls, env.sample.calls);
      maximum.bytes = std::max(maximum.bytes, env.sample.bytes);
    }
  }
  Require(env.validated_steps == kWarmups + kIterations, test,
          "wrong number of validated non-reset steps");
  std::printf(
      "{\"case\":\"%s\",\"max_players\":%d,\"input_player_rows\":%d,"
      "\"matching_players\":%zu,\"reset_calls_excluded\":1,"
      "\"warmup_calls_excluded\":%d,"
      "\"measured_calls\":%d,\"allocation_calls\":%zu,\"requested_bytes\":%zu,"
      "\"min_calls_per_step\":%zu,\"max_calls_per_step\":%zu,"
      "\"min_bytes_per_step\":%zu,\"max_bytes_per_step\":%zu,"
      "\"validated_steps\":%d,\"action_checksum\":\"%016" PRIx64 "\"}\n",
      test.name, test.max_players, test.total_rows, test.expected_rows.size(),
      kWarmups, kIterations, total.calls, total.bytes, minimum.calls,
      maximum.calls, minimum.bytes, maximum.bytes, env.validated_steps,
      env.last_checksum);
}

}  // namespace

int main() {
  std::puts(
      "{\"audit\":\"player_action_index\","
      "\"measurement\":\"EnvStep_to_Step_entry\","
      "\"counter\":\"thread_local_ordinary_new_and_new_array\","
      "\"metric\":\"requested_allocation_traffic\",\"timing\":false,"
      "\"fresh_env_per_case\":true,\"output_players_per_step\":1,"
      "\"excluded\":[\"setup\",\"reset\",\"warmup\",\"validation\","
      "\"Allocate\","
      "\"PostProcess\",\"Wait\",\"destruction\",\"background_threads\"]}");
  const Case cases[] = {
      {"single_player_control", 1, 4, {1}},
      {"contiguous_begin_1", 16, 9, {0}},
      {"contiguous_middle_4", 16, 12, {4, 5, 6, 7}},
      {"contiguous_end_1", 16, 9, {8}},
      {"contiguous_begin_16",
       16,
       24,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
      {"contiguous_end_16",
       16,
       24,
       {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
      {"scattered_4", 16, 12, {1, 4, 7, 10}},
      {"interleaved_16",
       16,
       32,
       {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
      {"zero_matching_players", 16, 9, {}},
      {"empty_player_list", 16, 0, {}},
  };
  for (const Case& test : cases) RunCase(test);
}
