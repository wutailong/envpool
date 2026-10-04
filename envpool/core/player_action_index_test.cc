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

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/env.h"
#include "envpool/core/env_spec.h"
#include "envpool/core/spec.h"
#include "envpool/core/state_buffer_queue.h"

namespace {

constexpr int kEnvId = 2;
constexpr int kBatchSize = 3;

class PlayerActionEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("matrix_columns"_.Bind(3));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config&) {
    return MakeDict();
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config& config) {
    return MakeDict(
        "player_scalar"_.Bind(Spec<int>({-1})),
        "player_matrix"_.Bind(Spec<double>({-1, 2, config["matrix_columns"_]})),
        "shared_scalar"_.Bind(Spec<int>({})),
        "shared_matrix"_.Bind(Spec<double>({2, 2})));
  }
};

using PlayerActionEnvSpec = EnvSpec<PlayerActionEnvFns>;

class PlayerActionEnv : public Env<PlayerActionEnvSpec> {
 public:
  using Env<PlayerActionEnvSpec>::Env;

  void Expect(int players, std::function<void(const Action&)> check) {
    players_ = players;
    check_ = std::move(check);
  }

  void Step(const Action& action) override {
    check_(action);
    // Complete the real state reservation even if an assertion in check fails.
    Allocate(players_)["reward"_].Fill(0.0F);
  }

  bool IsDone() override { return false; }

 private:
  int players_{0};
  std::function<void(const Action&)> check_;
};

PlayerActionEnvSpec MakeSpec(int max_players, int matrix_columns) {
  auto config = PlayerActionEnvSpec::kDefaultConfig;
  config["num_envs"_] = 8;
  config["max_num_players"_] = max_players;
  config["matrix_columns"_] = matrix_columns;
  return PlayerActionEnvSpec(config);
}

std::shared_ptr<std::vector<Array>> MakeBatch(
    const PlayerActionEnvSpec& spec, const std::vector<int>& player_ids,
    int batch_tag, int env_index) {
  auto batch = std::make_shared<std::vector<Array>>();
  for (auto shape : spec.action_spec.AllValues<ShapeSpec>()) {
    if (!shape.shape.empty() && shape.shape[0] == -1) {
      shape.shape[0] = static_cast<int>(player_ids.size());
    } else {
      shape = shape.Batch(kBatchSize);
    }
    if (shape.shape.size() == 3 && shape.shape[2] == 0) {
      // Keep the empty matrix's data pointer valid for zero-offset Slice.
      // The empty-inner-dimension case never takes the memcpy gather path.
      batch->emplace_back(shape, new char[1],
                          [](char* data) { delete[] data; });
    } else {
      batch->emplace_back(shape);
    }
  }
  PlayerActionEnv::Action action(*batch);
  for (int env = 0; env < kBatchSize; ++env) {
    action["env_id"_][env] = env == env_index ? kEnvId : 5 + env;
    action["shared_scalar"_][env] = batch_tag * 1000 + env * 100 + 7;
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 2; ++col) {
        action["shared_matrix"_](env, row, col) =
            batch_tag * 1000 + env * 100 + row * 10 + col + 0.25;
      }
    }
  }
  for (std::size_t player = 0; player < player_ids.size(); ++player) {
    const int index = static_cast<int>(player);
    action["players.env_id"_][index] = player_ids[player];
    action["player_scalar"_][index] =
        batch_tag * 1000 + (index % 2 == 0 ? index : -index) * 10 + 3;
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < spec.config["matrix_columns"_]; ++col) {
        action["player_matrix"_](index, row, col) =
            batch_tag * 1000 + index * 100 + row * 10 + col + 0.5;
      }
    }
  }
  return batch;
}

template <typename Dtype>
void ExpectSharedField(const TArray<Dtype>& actual, const Array& source,
                       int env_index) {
  const Array expected = source[env_index];
  ASSERT_EQ(actual.Shape(), expected.Shape());
  ASSERT_EQ(actual.size, expected.size);
  EXPECT_EQ(actual.ndim, expected.ndim);
  ASSERT_EQ(actual.element_size, sizeof(Dtype));
  EXPECT_EQ(actual.Data(), expected.Data());
  EXPECT_EQ(actual.SharedPtr().use_count(), 0);
  const auto* values = static_cast<const Dtype*>(actual.Data());
  const auto* input = static_cast<const Dtype*>(expected.Data());
  for (std::size_t i = 0; i < actual.size; ++i) {
    EXPECT_EQ(values[i], input[i]);
  }
}

template <typename Dtype>
void ExpectPlayerField(const TArray<Dtype>& actual, const Array& source,
                       const std::vector<int>& rows, bool is_slice) {
  auto shape = source.Shape();
  shape[0] = rows.size();
  std::size_t row_size = 1;
  for (std::size_t i = 1; i < shape.size(); ++i) {
    row_size *= shape[i];
  }
  ASSERT_EQ(actual.Shape(), shape);
  ASSERT_EQ(actual.size, rows.size() * row_size);
  EXPECT_EQ(actual.ndim, shape.size());
  ASSERT_EQ(actual.element_size, sizeof(Dtype));
  if (is_slice) {
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(actual.Data(), static_cast<const char*>(source.Data()) +
                                 rows.front() * row_size * sizeof(Dtype));
    EXPECT_EQ(actual.SharedPtr().use_count(), 0);
  } else {
    // Empty selections own their zero-sized arrays, just like gathered rows.
    EXPECT_GT(actual.SharedPtr().use_count(), 0);
    if (actual.size > 0) {
      EXPECT_NE(actual.Data(), source.Data());
    }
  }
  const auto* values = static_cast<const Dtype*>(actual.Data());
  const auto* input = static_cast<const Dtype*>(source.Data());
  for (std::size_t player = 0; player < rows.size(); ++player) {
    for (std::size_t i = 0; i < row_size; ++i) {
      EXPECT_EQ(values[player * row_size + i],
                input[rows[player] * row_size + i]);
    }
  }
}

class PlayerActionHarness {
 public:
  explicit PlayerActionHarness(int max_players = 8, int matrix_columns = 3)
      : spec_(MakeSpec(max_players, matrix_columns)),
        queue_(1, 1, max_players, spec_.state_spec.AllValues<ShapeSpec>()),
        env_(spec_, kEnvId) {}

  std::weak_ptr<std::vector<Array>> Run(const std::vector<int>& player_ids,
                                        const std::vector<int>& rows,
                                        bool is_slice, int batch_tag = 0,
                                        int env_index = 1) {
    SCOPED_TRACE(testing::Message()
                 << "batch_tag=" << batch_tag << ", env_index=" << env_index);
    auto batch = MakeBatch(spec_, player_ids, batch_tag, env_index);
    std::weak_ptr<std::vector<Array>> weak_batch = batch;
    env_.SetAction(batch, env_index);
    EXPECT_TRUE(previous_batch_.expired());
    previous_batch_ = weak_batch;
    env_.Expect(static_cast<int>(rows.size()), [weak_batch, rows, is_slice,
                                                env_index](const auto& action) {
      auto source = weak_batch.lock();
      ASSERT_NE(source, nullptr);
      // Views borrow input storage; Env retains the batch.
      EXPECT_EQ(source.use_count(), 2);
      ExpectSharedField(action["env_id"_], (*source)[0], env_index);
      ExpectPlayerField(action["players.env_id"_], (*source)[1], rows,
                        is_slice);
      ExpectPlayerField(action["player_scalar"_], (*source)[2], rows, is_slice);
      ExpectPlayerField(action["player_matrix"_], (*source)[3], rows, is_slice);
      ExpectSharedField(action["shared_scalar"_], (*source)[4], env_index);
      ExpectSharedField(action["shared_matrix"_], (*source)[5], env_index);
    });
    batch.reset();
    EXPECT_EQ(weak_batch.use_count(), 1);
    env_.EnvStep(&queue_, -1, false, false);
    PlayerActionEnv::State state(queue_.Wait());
    EXPECT_EQ(state["info:players.env_id"_].size, rows.size());
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][0]), kEnvId);
    EXPECT_EQ(weak_batch.use_count(), 1);
    return weak_batch;
  }

 private:
  PlayerActionEnvSpec spec_;
  StateBufferQueue queue_;
  PlayerActionEnv env_;
  std::weak_ptr<std::vector<Array>> previous_batch_;
};

}  // namespace

TEST(PlayerActionIndexTest, EmptyInputKeepsOwningZeroPlayerArrays) {
  PlayerActionHarness harness;
  harness.Run({}, {}, false);
}

TEST(PlayerActionIndexTest, NoMatchingIdsKeepsOwningZeroPlayerArrays) {
  PlayerActionHarness harness;
  harness.Run({5, 7, 5}, {}, false);
}

TEST(PlayerActionIndexTest, OneMatchingPlayerUsesSingletonSlice) {
  PlayerActionHarness harness;
  harness.Run({5, 2, 7}, {1}, true);
}

TEST(PlayerActionIndexTest, ContiguousPrefixIgnoresForeignSuffix) {
  PlayerActionHarness harness;
  harness.Run({2, 2, 2, 5, 7}, {0, 1, 2}, true);
}

TEST(PlayerActionIndexTest, ContiguousMiddleRetainsParentBatch) {
  std::weak_ptr<std::vector<Array>> batch;
  {
    PlayerActionHarness harness;
    batch = harness.Run({5, 2, 2, 2, 7, 5}, {1, 2, 3}, true);
    EXPECT_FALSE(batch.expired());
  }
  EXPECT_TRUE(batch.expired());
}

TEST(PlayerActionIndexTest, ContiguousSuffixUsesOffsetSlice) {
  PlayerActionHarness harness;
  harness.Run({5, 7, 2, 2, 2}, {2, 3, 4}, true);
}

TEST(PlayerActionIndexTest, AllIdsMatchUsesWholeBatchSlice) {
  PlayerActionHarness harness;
  harness.Run({2, 2, 2, 2}, {0, 1, 2, 3}, true);
}

TEST(PlayerActionIndexTest, EarlyGapGathersEveryLaterMatchInInputOrder) {
  PlayerActionHarness harness;
  harness.Run({2, 5, 2, 2, 7, 2}, {0, 2, 3, 5}, false);
}

TEST(PlayerActionIndexTest, LateGapIncludesEntireContiguousPrefix) {
  PlayerActionHarness harness;
  harness.Run({5, 2, 2, 2, 2, 7, 2, 5}, {1, 2, 3, 4, 6}, false);
}

TEST(PlayerActionIndexTest, ContiguousSlicePreservesEmptyInnerDimension) {
  PlayerActionHarness harness(8, 0);
  harness.Run({5, 2, 2, 7}, {1, 2}, true);
}

TEST(PlayerActionIndexTest, RepeatedStepsReplaceIndicesAndInputBatch) {
  PlayerActionHarness harness;
  harness.Run({2, 5, 2, 7, 2}, {0, 2, 4}, false, 1, 0);
  harness.Run({5, 2, 2, 7}, {1, 2}, true, 2, 1);
  harness.Run({7, 5}, {}, false, 3, 0);
  harness.Run({5, 2, 7, 2, 2, 5, 2}, {1, 3, 4, 6}, false, 4, 1);
}

TEST(PlayerActionIndexTest, SinglePlayerUsesBatchIndexWithoutScanningIds) {
  PlayerActionHarness harness(1);
  // Payload IDs deliberately differ from the selected environment: the
  // single-player path selects row env_index even when that ID is foreign.
  harness.Run({2, 7, 2}, {1}, true);
}
