// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/env.h"
#include "envpool/core/env_spec.h"
#include "envpool/core/spec.h"
#include "envpool/core/state_buffer.h"
#include "envpool/core/state_buffer_queue.h"

using PrimitiveValues =
    std::tuple<TArray<int>, TArray<double>, TArray<float>, TArray<int64_t>>;

namespace {

std::vector<ShapeSpec> PrimitiveSpecs(int batch = 0, int max_players = 3) {
  if (batch == 0) {
    return {Spec<int>({}), Spec<double>({2, 3}), Spec<float>({-1}),
            Spec<int64_t>({-1, 2, 1, 2})};
  }
  return {Spec<int>({batch}), Spec<double>({batch, 2, 3}),
          Spec<float>({batch * max_players}),
          Spec<int64_t>({batch * max_players, 2, 1, 2})};
}

PrimitiveValues FromVector(const std::vector<Array>& arrays) {
  return PrimitiveValues{TArray<int>(arrays[0]), TArray<double>(arrays[1]),
                         TArray<float>(arrays[2]), TArray<int64_t>(arrays[3])};
}

void WritePrimitive(PrimitiveValues* values, int env, int player_offset) {
  std::get<0>(*values) = env + 10;
  auto& matrix = std::get<1>(*values);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrix(i, j) = env * 10.0 + i * 3 + j + 0.25;
    }
  }
  auto& players = std::get<2>(*values);
  auto& player_matrix = std::get<3>(*values);
  for (std::size_t player = 0; player < players.size; ++player) {
    const int index = player_offset + static_cast<int>(player);
    players[player] = index + 0.5F;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        player_matrix(player, i, 0, j) = int64_t{index} * 100 + i * 2 + j;
      }
    }
  }
}

template <typename Dtype>
void ExpectSameArray(const TArray<Dtype>& actual, const Array& expected) {
  ASSERT_EQ(actual.Shape(), expected.Shape());
  ASSERT_EQ(actual.size, expected.size);
  EXPECT_EQ(actual.ndim, expected.ndim);
  EXPECT_EQ(actual.element_size, expected.element_size);
  const auto* lhs = static_cast<const Dtype*>(actual.Data());
  const auto* rhs = static_cast<const Dtype*>(expected.Data());
  for (std::size_t i = 0; i < actual.size; ++i) {
    EXPECT_EQ(lhs[i], rhs[i]);
  }
}

template <typename Dtype>
void ExpectSameNonOwningArray(const TArray<Dtype>& actual,
                              const Array& expected) {
  ExpectSameArray(actual, expected);
  EXPECT_EQ(actual.SharedPtr().use_count(), 0);
}

void ExpectPrimitive(const std::vector<Array>& output, int first_env, int envs,
                     int players) {
  ASSERT_EQ(output.size(), 4U);
  ASSERT_EQ(output[0].Shape(),
            std::vector<std::size_t>({static_cast<std::size_t>(envs)}));
  ASSERT_EQ(output[1].Shape(),
            std::vector<std::size_t>({static_cast<std::size_t>(envs), 2, 3}));
  ASSERT_EQ(output[2].Shape(),
            std::vector<std::size_t>({static_cast<std::size_t>(players)}));
  ASSERT_EQ(
      output[3].Shape(),
      std::vector<std::size_t>({static_cast<std::size_t>(players), 2, 1, 2}));
  ASSERT_EQ(output[0].size, static_cast<std::size_t>(envs));
  ASSERT_EQ(output[1].size, static_cast<std::size_t>(envs * 6));
  ASSERT_EQ(output[2].size, static_cast<std::size_t>(players));
  ASSERT_EQ(output[3].size, static_cast<std::size_t>(players * 4));
  for (int env = 0; env < envs; ++env) {
    EXPECT_EQ(static_cast<const int*>(output[0].Data())[env],
              first_env + env + 10);
    for (int element = 0; element < 6; ++element) {
      EXPECT_EQ(static_cast<const double*>(output[1].Data())[env * 6 + element],
                (first_env + env) * 10.0 + element + 0.25);
    }
  }
  for (int player = 0; player < players; ++player) {
    EXPECT_EQ(static_cast<const float*>(output[2].Data())[player],
              player + 0.5F);
    for (int element = 0; element < 4; ++element) {
      EXPECT_EQ(
          static_cast<const int64_t*>(output[3].Data()) [player * 4 + element],
          int64_t { player } * 100 + element);
    }
  }
}

void CheckPrimitiveEquivalence(int batch, const std::vector<int>& players) {
  StateBuffer typed(batch, 3, PrimitiveSpecs(batch),
                    {false, false, true, true});
  StateBuffer legacy(batch, 3, PrimitiveSpecs(batch),
                     {false, false, true, true});
  int player_offset = 0;
  for (std::size_t env = 0; env < players.size(); ++env) {
    auto direct = typed.AllocateTuple<PrimitiveValues>(players[env]);
    static_assert(std::is_aggregate_v<decltype(direct)>);
    static_assert(std::is_same_v<decltype(direct.arr), PrimitiveValues>);
    auto vector = legacy.Allocate(players[env]);
    auto converted = FromVector(vector.arr);
    WritePrimitive(&direct.arr, static_cast<int>(env), player_offset);
    WritePrimitive(&converted, static_cast<int>(env), player_offset);
    std::size_t field = 0;
    std::apply(
        [&](const auto&... array) {
          (ExpectSameNonOwningArray(array, vector.arr[field++]), ...);
        },
        direct.arr);
    player_offset += players[env];
    EXPECT_EQ(typed.Offsets(), legacy.Offsets());
    EXPECT_EQ(typed.Offsets().first, player_offset);
    EXPECT_EQ(typed.Offsets().second, env + 1);
    direct.done_write();
    vector.done_write();
  }
  const std::size_t unused = batch - players.size();
  auto direct_output = typed.Wait(unused);
  auto vector_output = legacy.Wait(unused);
  ExpectPrimitive(direct_output, 0, static_cast<int>(players.size()),
                  player_offset);
  ExpectPrimitive(vector_output, 0, static_cast<int>(players.size()),
                  player_offset);
}

class ObservedBuffer : public StateBuffer {
 public:
  using StateBuffer::StateBuffer;

  std::size_t Allocations() const { return alloc_count_.load(); }
  std::size_t Completions() const { return done_count_.load(); }
};

}  // namespace

TEST(StateTupleTest, MatchesVectorScalarMultidimensionalAndPartialPlayerViews) {
  CheckPrimitiveEquivalence(6, {2, 0, 3, 0});
}

TEST(StateTupleTest, MatchesVectorWhenEveryEnvironmentHasZeroPlayers) {
  CheckPrimitiveEquivalence(3, {0, 0, 0});
}

TEST(StateTupleTest, SinglePlayerReversedOrderMatchesBothFieldLayouts) {
  using Values = std::tuple<TArray<int>, TArray<int>>;
  StateBuffer buffer(4, 1, {Spec<int>({4}), Spec<int>({4, 2})}, {false, true});
  for (int visit = 0; visit < 4; ++visit) {
    auto slice = buffer.AllocateTuple<Values>(1, 3 - visit);
    EXPECT_EQ(std::get<0>(slice.arr).Shape(), std::vector<std::size_t>());
    EXPECT_EQ(std::get<1>(slice.arr).Shape(), std::vector<std::size_t>({1, 2}));
    std::get<0>(slice.arr) = visit;
    std::get<1>(slice.arr).Fill(visit + 10);
    slice.done_write();
  }
  auto output = buffer.Wait();
  for (int row = 0; row < 4; ++row) {
    EXPECT_EQ(static_cast<const int*>(output[0].Data())[row], 3 - row);
    EXPECT_EQ(static_cast<const int*>(output[1].Data())[row * 2], 13 - row);
    EXPECT_EQ(static_cast<const int*>(output[1].Data())[row * 2 + 1], 13 - row);
  }
}

TEST(StateTupleTest, MultiPlayerIgnoresOrderIncludingEmptySlices) {
  using Values = std::tuple<TArray<int>, TArray<int>>;
  StateBuffer buffer(3, 3, {Spec<int>({3}), Spec<int>({9, 2})}, {false, true});
  const std::array<int, 3> players{2, 0, 1};
  for (int visit = 0; visit < 3; ++visit) {
    auto slice = buffer.AllocateTuple<Values>(players[visit], 2 - visit);
    std::get<0>(slice.arr) = visit;
    std::get<1>(slice.arr).Fill(visit + 10);
    slice.done_write();
  }
  auto output = buffer.Wait();
  EXPECT_EQ(output[1].Shape(), std::vector<std::size_t>({3, 2}));
  for (int row = 0; row < 3; ++row) {
    EXPECT_EQ(static_cast<const int*>(output[0].Data())[row], row);
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(static_cast<const int*>(output[1].Data())[row * 2 + i],
                row < 2 ? 10 : 12);
    }
  }
}

TEST(StateTupleTest, MixedAllocationsShareQuotaAndPreserveFailureOffsets) {
  using Values = std::tuple<TArray<int>>;
  ObservedBuffer buffer(2, 2, {Spec<int>({2})}, {false});
  auto typed = buffer.AllocateTuple<Values>(1);
  auto vector = buffer.Allocate(2);
  std::get<0>(typed.arr) = 11;
  vector.arr[0] = 22;
  const auto offsets = buffer.Offsets();
  EXPECT_EQ(offsets, (std::pair<uint32_t, uint32_t>{3, 2}));
  EXPECT_THROW(buffer.AllocateTuple<Values>(0), std::out_of_range);
  EXPECT_THROW(buffer.Allocate(0), std::out_of_range);
  EXPECT_EQ(buffer.Allocations(), 4U);
  EXPECT_EQ(buffer.Offsets(), offsets);
  typed.done_write();
  vector.done_write();
  auto output = buffer.Wait();
  EXPECT_EQ(static_cast<const int*>(output[0].Data())[0], 11);
  EXPECT_EQ(static_cast<const int*>(output[0].Data())[1], 22);
}

TEST(StateTupleDeathTest, RejectsBothShortAndLongTuples) {
  using TooFew = std::tuple<>;
  using TooMany = std::tuple<TArray<int>, TArray<int>>;
  StateBuffer buffer(1, 1, {Spec<int>({1})}, {false});
  EXPECT_DEATH(buffer.AllocateTuple<TooFew>(1), "Check failed");
  EXPECT_DEATH(buffer.AllocateTuple<TooMany>(1), "Check failed");
}

TEST(StateTupleTest, DestroyingTypedSliceDoesNotCompleteItsReservation) {
  using Values = std::tuple<TArray<int>>;
  ObservedBuffer buffer(1, 1, {Spec<int>({1})}, {false});
  std::function<void()> done;
  {
    auto slice = buffer.AllocateTuple<Values>(1);
    std::get<0>(slice.arr) = 42;
    // Keep the original callback populated when the aggregate is destroyed.
    done = slice.done_write;
    EXPECT_EQ(buffer.Completions(), 0U);
  }
  EXPECT_EQ(buffer.Allocations(), 1U);
  EXPECT_EQ(buffer.Completions(), 0U);
  done();
  EXPECT_EQ(buffer.Completions(), 1U);
  auto output = buffer.Wait();
  EXPECT_EQ(static_cast<const int*>(output[0].Data())[0], 42);
}

TEST(StateTupleTest, MixedQueueAllocationsSurvivePartialAndReplacementCycles) {
  constexpr int kBatch = 3;
  // This queue has six slots. Retained outputs cross multiple complete wraps.
  constexpr int kRounds = 20;
  StateBufferQueue queue(kBatch, kBatch, 3, PrimitiveSpecs());
  std::vector<std::vector<Array>> outputs;
  std::vector<int> player_totals;
  for (int round = 0; round < kRounds; ++round) {
    const int written = kBatch - round % kBatch;
    int player_offset = 0;
    for (int env = 0; env < written; ++env) {
      const int players = (round + env) % 4;
      if ((round + env) % 2 == 0) {
        auto slice = queue.AllocateTuple<PrimitiveValues>(players);
        WritePrimitive(&slice.arr, round * 10 + env, player_offset);
        slice.done_write();
      } else {
        auto slice = queue.Allocate(players);
        auto values = FromVector(slice.arr);
        WritePrimitive(&values, round * 10 + env, player_offset);
        slice.done_write();
      }
      player_offset += players;
    }
    outputs.emplace_back(queue.Wait(kBatch - written));
    player_totals.push_back(player_offset);
  }
  for (int round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(round);
    ExpectPrimitive(outputs[round], round * 10, kBatch - round % kBatch,
                    player_totals[round]);
  }
}

TEST(StateTupleTest, QueueForwardsSyncOrderAcrossMixedReplacementCycles) {
  using Values = std::tuple<TArray<int>, TArray<int>>;
  StateBufferQueue queue(3, 3, 1, {Spec<int>({}), Spec<int>({-1, 2})});
  for (int round = 0; round < 14; ++round) {
    for (int visit = 0; visit < 3; ++visit) {
      const int value = round * 10 + visit;
      if ((round + visit) % 2 == 0) {
        auto slice = queue.AllocateTuple<Values>(1, 2 - visit);
        std::get<0>(slice.arr) = value;
        std::get<1>(slice.arr).Fill(value + 100);
        slice.done_write();
      } else {
        auto slice = queue.Allocate(1, 2 - visit);
        slice.arr[0] = value;
        slice.arr[1].Fill(value + 100);
        slice.done_write();
      }
    }
    const auto output = queue.Wait();
    for (int row = 0; row < 3; ++row) {
      const int value = round * 10 + 2 - row;
      EXPECT_EQ(static_cast<const int*>(output[0].Data())[row], value);
      EXPECT_EQ(static_cast<const int*>(output[1].Data())[row * 2],
                value + 100);
      EXPECT_EQ(static_cast<const int*>(output[1].Data())[row * 2 + 1],
                value + 100);
    }
  }
}

TEST(StateTupleTest, PartialWaitStillRequiresDelayedTypedLastWriter) {
  using Values = std::tuple<TArray<int>, TArray<int>>;
  constexpr int kBatch = 4;
  ObservedBuffer buffer(kBatch, 2,
                        {Spec<int>({kBatch * 2, 2}), Spec<int>({kBatch})},
                        {true, false});
  auto first = buffer.Allocate(1);
  auto last = buffer.AllocateTuple<Values>(2);
  first.arr[0].Fill(11);
  first.arr[1] = 12;
  first.done_write();
  auto result = std::async(std::launch::async, [&] { return buffer.Wait(2); });
  // Observe Wait's unused-quota publication, rather than sleeping and assuming
  // the consumer has reached its blocking point.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (buffer.Completions() < kBatch - 1 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool unused_accounted_for = buffer.Completions() == kBatch - 1;
  const auto pending = result.wait_for(std::chrono::seconds(0));
  std::get<0>(last.arr).Fill(21);
  std::get<1>(last.arr) = 22;
  last.done_write();
  auto output = result.get();
  // Release the outstanding writer and join the consumer before assertions.
  ASSERT_TRUE(unused_accounted_for);
  EXPECT_EQ(pending, std::future_status::timeout);
  ASSERT_EQ(output.size(), 2U);
  EXPECT_EQ(output[0].Shape(), std::vector<std::size_t>({3, 2}));
  EXPECT_EQ(output[1].Shape(), std::vector<std::size_t>({2}));
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(static_cast<const int*>(output[0].Data())[i], i < 2 ? 11 : 21);
  }
  EXPECT_EQ(static_cast<const int*>(output[1].Data())[0], 12);
  EXPECT_EQ(static_cast<const int*>(output[1].Data())[1], 22);
}

TEST(StateTupleTest, TypedCompletionAllowsImmediateBufferRetirement) {
  using Values = std::tuple<TArray<int>, TArray<int>>;
  constexpr int kBatch = 4;
  constexpr int kRounds = 64;
  std::vector<std::vector<Array>> outputs;
  for (int round = 0; round < kRounds; ++round) {
    auto buffer = std::make_unique<StateBuffer>(
        kBatch, 1,
        std::vector<ShapeSpec>{Spec<int>({kBatch, 2}), Spec<int>({kBatch})},
        std::vector<bool>{true, false});
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::array<std::thread, kBatch> writers;
    for (int writer = 0; writer < kBatch; ++writer) {
      auto slice = buffer->AllocateTuple<Values>(1);
      writers[writer] = std::thread(
          [ready, round, writer, slice = std::move(slice)]() mutable {
            ready.wait();
            std::get<0>(slice.arr).Fill(round * 100 + writer);
            std::get<1>(slice.arr) = round * 100 + writer + 50;
            slice.done_write();
          });
    }
    start.set_value();
    outputs.emplace_back(buffer->Wait());
    // Mirrors the existing completion regression: callbacks may still be
    // returning when the consumed buffer is retired.
    buffer.reset();
    for (auto& writer : writers) {
      writer.join();
    }
  }
  for (int round = 0; round < kRounds; ++round) {
    for (int writer = 0; writer < kBatch; ++writer) {
      const auto* players = static_cast<const int*>(outputs[round][0].Data());
      const auto* shared = static_cast<const int*>(outputs[round][1].Data());
      EXPECT_EQ(players[writer * 2], round * 100 + writer);
      EXPECT_EQ(players[writer * 2 + 1], round * 100 + writer);
      EXPECT_EQ(shared[writer], round * 100 + writer + 50);
    }
  }
}

namespace {

class TupleContainerFns {
 public:
  static decltype(auto) DefaultConfig() { return MakeDict(); }

  template <typename Config>
  static decltype(auto) StateSpec(const Config&) {
    return MakeDict(
        "scalar"_.Bind(Spec<Container<int>>({}, Spec<int>({1}))),
        "matrix"_.Bind(Spec<Container<int>>({2, 3}, Spec<int>({1}))),
        "players"_.Bind(Spec<Container<int>>({-1, 2}, Spec<int>({1}))),
        "empty"_.Bind(Spec<Container<int>>({0, 2}, Spec<int>({1}))),
        "empty_players"_.Bind(
            Spec<Container<int>>({-1, 0, 3}, Spec<int>({1}))));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config&) {
    return MakeDict();
  }
};

using TupleContainerSpec = EnvSpec<TupleContainerFns>;

class TupleContainerEnv : public Env<TupleContainerSpec> {
 public:
  TupleContainerEnv(const Spec& spec, int env_id, int players)
      : Env<TupleContainerSpec>(spec, env_id), players_(players) {}

  void Reset() override {
    auto state = Allocate(players_);
    state["reward"_].Fill(0.0F);
    Fill(state["scalar"_], env_id_ * 100);
    Fill(state["matrix"_], env_id_ * 100 + 10);
    Fill(state["players"_], env_id_ * 100 + 20);
    Fill(state["empty"_], 0);
    Fill(state["empty_players"_], 0);
  }

  bool IsDone() override { return false; }

 private:
  static void Fill(TArray<Container<int>>& array, int base) {
    auto* slots = static_cast<Container<int>*>(array.Data());
    for (std::size_t i = 0; i < array.size; ++i) {
      EXPECT_EQ(slots[i], nullptr);
      slots[i] = std::make_unique<TArray<int>>(::Spec<int>({1}));
      (*slots[i])[0] = base + static_cast<int>(i);
    }
  }

  int players_;
};

void ExpectPayloads(const Array& array, int base) {
  const auto* slots = static_cast<const Container<int>*>(array.Data());
  for (std::size_t i = 0; i < array.size; ++i) {
    ASSERT_NE(slots[i], nullptr);
    EXPECT_EQ(static_cast<int>((*slots[i])[0]), base + static_cast<int>(i));
  }
}

}  // namespace

TEST(StateTupleTest, EnvInitializesScalarMultidimensionalAndEmptyContainers) {
  auto config = TupleContainerSpec::kDefaultConfig;
  config["num_envs"_] = 3;
  config["max_num_players"_] = 2;
  TupleContainerSpec spec(config);
  // All outer Container slots have real typed lifetimes, including unused
  // capacity. Reclamation, weak owners and fallback recipes have their own
  // dedicated container_output_test.cc coverage.
  StateBufferQueue queue(3, 3, 2, spec.state_spec.AllValues<ShapeSpec>(),
                         MakeStateArrayFactories(spec.state_spec.AllValues()));
  TupleContainerEnv empty(spec, 0, 0);
  TupleContainerEnv populated(spec, 1, 2);
  empty.EnvStep(&queue, -1, true, false);
  populated.EnvStep(&queue, -1, true, false);
  TupleContainerEnv::State state(queue.Wait(1));
  ASSERT_EQ(state["scalar"_].Shape(), std::vector<std::size_t>({2}));
  ASSERT_EQ(state["matrix"_].Shape(), std::vector<std::size_t>({2, 2, 3}));
  ASSERT_EQ(state["players"_].Shape(), std::vector<std::size_t>({2, 2}));
  EXPECT_EQ(state["empty"_].Shape(), std::vector<std::size_t>({2, 0, 2}));
  EXPECT_EQ(state["empty"_].size, 0U);
  EXPECT_EQ(state["empty_players"_].Shape(),
            std::vector<std::size_t>({2, 0, 3}));
  EXPECT_EQ(state["empty_players"_].size, 0U);
  ExpectPayloads(state["scalar"_][0], 0);
  ExpectPayloads(state["scalar"_][1], 100);
  ExpectPayloads(state["matrix"_][0], 10);
  ExpectPayloads(state["matrix"_][1], 110);
  ExpectPayloads(state["players"_], 120);
  EXPECT_EQ(static_cast<int>(state["info:env_id"_][0]), 0);
  EXPECT_EQ(static_cast<int>(state["info:env_id"_][1]), 1);
  for (int player = 0; player < 2; ++player) {
    EXPECT_EQ(static_cast<int>(state["info:players.env_id"_][player]), 1);
    EXPECT_FLOAT_EQ(static_cast<float>(state["discount"_][player]), 1.0F);
  }
}
