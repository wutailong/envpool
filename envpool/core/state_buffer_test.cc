// Copyright 2021 Garena Online Private Limited
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

#include "envpool/core/state_buffer.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "envpool/core/spec.h"

TEST(StateBufferTest, Basic) {
  int batch = 32;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  int max_num_players = 10;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, false}));
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    std::size_t num = 1;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(std::get<0>(offset), std::get<1>(offset));
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
}

TEST(StateBufferTest, SinglePlayerSync) {
  int batch = 32;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  int max_num_players = 1;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, false}));
  auto offset = buffer.Offsets();
  std::size_t total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    std::size_t num = 1;
    total += num;
    // use reversed order to write data
    auto r = buffer.Allocate(num, batch - 1 - i);
    offset = buffer.Offsets();
    EXPECT_EQ(std::get<0>(offset), std::get<1>(offset));
    EXPECT_EQ(r.arr[0].Shape(), std::vector<std::size_t>({10, 2, 2}));
    EXPECT_EQ(r.arr[1].Shape(), std::vector<std::size_t>({1, 2, 2}));
    r.arr[1](0, 0, 0) = i;  // only the first element is modified
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
  for (int i = 0; i < batch; ++i) {
    auto* ptr = reinterpret_cast<int*>(bs[1][i].Data());
    EXPECT_EQ(ptr[0], batch - 1 - i);
  }
}

TEST(StateBufferTest, Truncate) {
  int batch = 32;
  int max_num_players = 10;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch, 10, 2, 2}),
                               ShapeSpec(4, {batch * max_num_players, 2, 2})};
  std::size_t player_num = 3;
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({false, true}));
  auto r = buffer.Allocate(player_num);
  r.done_write();
  auto bs = buffer.Wait(batch - 1);
  EXPECT_EQ(bs[0].Shape(), std::vector<std::size_t>({1, 10, 2, 2}));
  EXPECT_EQ(bs[1].Shape(), std::vector<std::size_t>(
                               {static_cast<std::size_t>(player_num), 2, 2}));
}

TEST(StateBufferTest, MultiPlayers) {
  int batch = 32;
  int max_num_players = 10;
  std::vector<ShapeSpec> specs{ShapeSpec(1, {batch * max_num_players, 2, 2}),
                               ShapeSpec(4, {batch, 1, 2, 2})};
  StateBuffer buffer(batch, max_num_players, specs,
                     std::vector<bool>({true, false}));
  auto offset = buffer.Offsets();
  int total = 0;
  std::srand(std::time(nullptr));
  for (int i = 0; i < batch; ++i) {
    int num = 1 + std::rand() % max_num_players;
    total += num;
    auto r = buffer.Allocate(num);
    offset = buffer.Offsets();
    EXPECT_EQ(num, r.arr[0].Shape()[0]);
    EXPECT_EQ(std::get<0>(offset), total);
    EXPECT_EQ(std::get<1>(offset), i + 1);
    r.done_write();
  }
  auto bs = buffer.Wait();
  EXPECT_EQ(bs[0].Shape(0), total);
  EXPECT_EQ(bs[1].Shape(0), batch);
}

TEST(StateBufferTest, PartialSyncShuffledOutputsSurviveBufferDestruction) {
  constexpr int kBatch = 7;
  constexpr int kWritten = 4;
  const std::array<std::size_t, 3> row_sizes{8, 6, 1};
  std::vector<Array> result;
  {
    std::vector<ShapeSpec> specs{ShapeSpec(sizeof(int), {kBatch, 2, 1, 2, 2}),
                                 ShapeSpec(sizeof(int), {kBatch, 2, 3}),
                                 ShapeSpec(sizeof(int), {kBatch})};
    StateBuffer buffer(kBatch, 1, specs, {false, true, false});
    for (int order : {2, 0, 3, 1}) {
      auto slice = buffer.Allocate(1, order);
      ASSERT_EQ(slice.arr.size(), row_sizes.size());
      for (std::size_t field = 0; field < slice.arr.size(); ++field) {
        ASSERT_EQ(slice.arr[field].size, row_sizes[field]);
        auto* data = reinterpret_cast<int*>(slice.arr[field].Data());
        for (std::size_t i = 0; i < row_sizes[field]; ++i) {
          data[i] = 1000 * static_cast<int>(field) + 100 * order +
                    static_cast<int>(i);
        }
      }
      slice.done_write();
    }
    result = buffer.Wait(kBatch - kWritten);
  }

  ASSERT_EQ(result.size(), 3U);
  EXPECT_EQ(result[0].Shape(),
            std::vector<std::size_t>({kWritten, 2, 1, 2, 2}));
  EXPECT_EQ(result[1].Shape(), std::vector<std::size_t>({kWritten, 2, 3}));
  EXPECT_EQ(result[2].Shape(), std::vector<std::size_t>({kWritten}));
  for (std::size_t field = 0; field < result.size(); ++field) {
    EXPECT_EQ(result[field].size, kWritten * row_sizes[field]);
    const auto* data = reinterpret_cast<const int*>(result[field].Data());
    for (int row = 0; row < kWritten; ++row) {
      for (std::size_t i = 0; i < row_sizes[field]; ++i) {
        EXPECT_EQ(
            data[row * row_sizes[field] + i],
            1000 * static_cast<int>(field) + 100 * row + static_cast<int>(i));
      }
    }
  }
}

TEST(StateBufferTest, PartialMultiPlayerOutputsIncludeZeroPlayerEnvironments) {
  constexpr int kBatch = 6;
  constexpr int kMaxPlayers = 4;
  const std::array<int, 4> player_counts{2, 0, 3, 0};
  std::vector<Array> result;
  {
    std::vector<ShapeSpec> specs{
        ShapeSpec(sizeof(int), {kBatch * kMaxPlayers, 2, 3}),
        ShapeSpec(sizeof(int), {kBatch, 2}),
        ShapeSpec(sizeof(int), {kBatch * kMaxPlayers, 1, 2, 2})};
    StateBuffer buffer(kBatch, kMaxPlayers, specs, {true, false, true});
    std::size_t player_offset = 0;
    for (std::size_t env = 0; env < player_counts.size(); ++env) {
      const auto players = static_cast<std::size_t>(player_counts[env]);
      auto slice = buffer.Allocate(players);
      EXPECT_EQ(slice.arr[0].Shape(),
                std::vector<std::size_t>({players, 2, 3}));
      EXPECT_EQ(slice.arr[1].Shape(), std::vector<std::size_t>({2}));
      EXPECT_EQ(slice.arr[2].Shape(),
                std::vector<std::size_t>({players, 1, 2, 2}));
      for (std::size_t field : {0U, 2U}) {
        const std::size_t row_size = field == 0 ? 6 : 4;
        auto* data = reinterpret_cast<int*>(slice.arr[field].Data());
        for (std::size_t player = 0; player < players; ++player) {
          for (std::size_t i = 0; i < row_size; ++i) {
            data[player * row_size + i] =
                1000 * static_cast<int>(field) +
                100 * static_cast<int>(player_offset + player) +
                static_cast<int>(i);
          }
        }
      }
      auto* shared = reinterpret_cast<int*>(slice.arr[1].Data());
      for (int i = 0; i < 2; ++i) {
        shared[i] = 500 + 10 * static_cast<int>(env) + i;
      }
      player_offset += players;
      EXPECT_EQ(buffer.Offsets().first, player_offset);
      EXPECT_EQ(buffer.Offsets().second, env + 1);
      slice.done_write();
    }
    result = buffer.Wait(kBatch - player_counts.size());
  }

  ASSERT_EQ(result.size(), 3U);
  EXPECT_EQ(result[0].Shape(), std::vector<std::size_t>({5, 2, 3}));
  EXPECT_EQ(result[1].Shape(), std::vector<std::size_t>({4, 2}));
  EXPECT_EQ(result[2].Shape(), std::vector<std::size_t>({5, 1, 2, 2}));
  EXPECT_EQ(result[0].size, 30U);
  EXPECT_EQ(result[1].size, 8U);
  EXPECT_EQ(result[2].size, 20U);
  for (std::size_t field : {0U, 2U}) {
    const std::size_t row_size = field == 0 ? 6 : 4;
    const auto* data = reinterpret_cast<const int*>(result[field].Data());
    for (int player = 0; player < 5; ++player) {
      for (std::size_t i = 0; i < row_size; ++i) {
        EXPECT_EQ(data[player * row_size + i], 1000 * static_cast<int>(field) +
                                                   100 * player +
                                                   static_cast<int>(i));
      }
    }
  }
  const auto* shared = reinterpret_cast<const int*>(result[1].Data());
  for (std::size_t env = 0; env < player_counts.size(); ++env) {
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(shared[env * 2 + i], 500 + 10 * static_cast<int>(env) + i);
    }
  }
}

TEST(StateBufferTest, NoPlayersReturnsEmptyPlayerShapesAndSharedData) {
  constexpr int kBatch = 3;
  std::vector<Array> result;
  {
    std::vector<ShapeSpec> specs{ShapeSpec(sizeof(int), {kBatch * 2, 2, 3}),
                                 ShapeSpec(sizeof(int), {kBatch, 2})};
    StateBuffer buffer(kBatch, 2, specs, {true, false});
    for (int env = 0; env < kBatch; ++env) {
      auto slice = buffer.Allocate(0);
      EXPECT_EQ(slice.arr[0].Shape(), std::vector<std::size_t>({0, 2, 3}));
      EXPECT_EQ(slice.arr[0].size, 0U);
      slice.arr[1](0) = env;
      slice.arr[1](1) = env + 10;
      slice.done_write();
    }
    result = buffer.Wait();
  }

  ASSERT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0].Shape(), std::vector<std::size_t>({0, 2, 3}));
  EXPECT_EQ(result[0].size, 0U);
  EXPECT_EQ(result[1].Shape(), std::vector<std::size_t>({kBatch, 2}));
  EXPECT_EQ(result[1].size, 6U);
  const auto* shared = reinterpret_cast<const int*>(result[1].Data());
  for (int env = 0; env < kBatch; ++env) {
    EXPECT_EQ(shared[env * 2], env);
    EXPECT_EQ(shared[env * 2 + 1], env + 10);
  }
}

TEST(StateBufferTest, PartialWaitRequiresEveryAllocatedWriterToComplete) {
  constexpr int kBatch = 4;
  std::vector<ShapeSpec> specs{ShapeSpec(sizeof(int), {kBatch * 2, 2}),
                               ShapeSpec(sizeof(int), {kBatch, 2})};
  StateBuffer buffer(kBatch, 2, specs, {true, false});
  auto first = buffer.Allocate(1);
  auto last = buffer.Allocate(2);
  first.arr[0].Fill(11);
  first.arr[1].Fill(12);
  first.done_write();

  std::promise<void> wait_started;
  auto started = wait_started.get_future();
  auto ready = std::async(std::launch::async, [&buffer, &wait_started]() {
    wait_started.set_value();
    return buffer.Wait(kBatch - 2);
  });
  started.wait();
  // Unused quota is accounted for, but the final writer is still outstanding.
  EXPECT_EQ(ready.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  last.arr[0].Fill(21);
  last.arr[1].Fill(22);
  last.done_write();

  auto result = ready.get();
  ASSERT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0].Shape(), std::vector<std::size_t>({3, 2}));
  EXPECT_EQ(result[1].Shape(), std::vector<std::size_t>({2, 2}));
  EXPECT_EQ(result[0].size, 6U);
  EXPECT_EQ(result[1].size, 4U);
  const auto* players = reinterpret_cast<const int*>(result[0].Data());
  const auto* shared = reinterpret_cast<const int*>(result[1].Data());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(players[i], i < 2 ? 11 : 21);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(shared[i], i < 2 ? 12 : 22);
  }
}

TEST(StateBufferTest, CompletionAllowsImmediateDestruction) {
  constexpr int kRounds = 256;
  constexpr int kBatch = 4;
  const std::vector<ShapeSpec> specs{ShapeSpec(sizeof(int), {kBatch, 2}),
                                     ShapeSpec(sizeof(int), {kBatch})};
  std::vector<std::vector<Array>> outputs;
  outputs.reserve(kRounds);
  for (int round = 0; round < kRounds; ++round) {
    auto buffer = std::make_unique<StateBuffer>(kBatch, 1, specs,
                                                std::vector<bool>{true, false});
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::array<std::thread, kBatch> writers;
    for (int writer = 0; writer < kBatch; ++writer) {
      auto slice = buffer->Allocate(1);
      writers[writer] = std::thread(
          [ready, round, writer, slice = std::move(slice)]() mutable {
            ready.wait();
            slice.arr[0].Fill(round * 100 + writer);
            slice.arr[1] = round * 100 + writer + 50;
            slice.done_write();
          });
    }
    start.set_value();
    outputs.emplace_back(buffer->Wait());
    // Completion permits destroying the buffer immediately. A non-final
    // writer must not read buffer members after publishing its done count.
    buffer.reset();
    for (auto& writer : writers) {
      writer.join();
    }
  }

  // Keep all owning outputs across later allocations and buffer destruction.
  for (int round = 0; round < kRounds; ++round) {
    const auto& output = outputs[round];
    ASSERT_EQ(output.size(), 2U);
    EXPECT_EQ(output[0].Shape(), std::vector<std::size_t>({kBatch, 2}));
    EXPECT_EQ(output[1].Shape(), std::vector<std::size_t>({kBatch}));
    const auto* players = reinterpret_cast<const int*>(output[0].Data());
    const auto* shared = reinterpret_cast<const int*>(output[1].Data());
    for (int writer = 0; writer < kBatch; ++writer) {
      EXPECT_EQ(players[writer * 2], round * 100 + writer);
      EXPECT_EQ(players[writer * 2 + 1], round * 100 + writer);
      EXPECT_EQ(shared[writer], round * 100 + writer + 50);
    }
  }
}
