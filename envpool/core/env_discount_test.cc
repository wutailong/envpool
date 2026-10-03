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
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "envpool/core/env.h"
#include "envpool/core/env_spec.h"
#include "envpool/core/state_buffer_queue.h"

namespace {

class DiscountEnvFns {
 public:
  static decltype(auto) DefaultConfig() { return MakeDict(); }

  template <typename Config>
  static decltype(auto) StateSpec(const Config&) {
    return MakeDict();
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config&) {
    return MakeDict();
  }
};

using DiscountEnvSpec = EnvSpec<DiscountEnvFns>;

class DiscountEnv : public Env<DiscountEnvSpec> {
 public:
  DiscountEnv(const Spec& spec, int env_id, int num_players, bool done,
              std::function<void()> before_done = {})
      : Env<DiscountEnvSpec>(spec, env_id),
        num_players_(num_players),
        done_(done),
        before_done_(std::move(before_done)) {}

  void Reset() override { Allocate(num_players_)["reward"_].Fill(0.0F); }

  bool IsDone() override {
    if (before_done_) {
      before_done_();
    }
    return done_;
  }

 private:
  int num_players_;
  bool done_;
  std::function<void()> before_done_;
};

void CheckDiscounts(int num_players, bool done) {
  SCOPED_TRACE(testing::Message()
               << "num_players=" << num_players << ", done=" << done);
  auto config = DiscountEnvSpec::kDefaultConfig;
  config["max_num_players"_] = num_players;
  DiscountEnvSpec spec(config);
  StateBufferQueue queue(1, 1, num_players,
                         spec.state_spec.AllValues<ShapeSpec>());
  DiscountEnv env(spec, 0, num_players, done);
  env.EnvStep(&queue, -1, true, false);
  DiscountEnv::State state(queue.Wait());
  ASSERT_EQ(state["discount"_].Shape(),
            std::vector<std::size_t>({static_cast<std::size_t>(num_players)}));
  ASSERT_EQ(state["info:players.env_id"_].size, state["discount"_].size);
  EXPECT_EQ(static_cast<bool>(state["done"_][0]), done);
  for (int player = 0; player < num_players; ++player) {
    EXPECT_FLOAT_EQ(static_cast<float>(state["discount"_][player]),
                    done ? 0.0F : 1.0F);
    EXPECT_EQ(static_cast<int>(state["info:players.env_id"_][player]), 0);
  }
}

}  // namespace

TEST(EnvDiscountTest, SinglePlayerDiscountMatchesDone) {
  CheckDiscounts(1, false);
  CheckDiscounts(1, true);
}

TEST(EnvDiscountTest, NonterminalDiscountIncludesEveryPlayer) {
  CheckDiscounts(4, false);
}

TEST(EnvDiscountTest, TerminalDiscountIncludesEveryPlayer) {
  CheckDiscounts(4, true);
}

TEST(EnvDiscountTest, ZeroPlayersDoNotOverwriteNextEnvironment) {
  auto config = DiscountEnvSpec::kDefaultConfig;
  config["num_envs"_] = 2;
  config["max_num_players"_] = 2;
  DiscountEnvSpec spec(config);
  StateBufferQueue queue(2, 2, 2, spec.state_spec.AllValues<ShapeSpec>());
  std::promise<void> reserved;
  std::promise<void> release;
  auto reserved_future = reserved.get_future();
  auto release_future = release.get_future();
  DiscountEnv empty_env(spec, 0, 0, true, [&] {
    // IsDone is called after reserving the empty slice, before initializing
    // discount. The next environment will reserve the same player offset.
    reserved.set_value();
    release_future.wait();
  });
  DiscountEnv player_env(spec, 1, 1, false);
  std::thread empty_writer([&] { empty_env.EnvStep(&queue, -1, true, false); });
  reserved_future.wait();
  player_env.EnvStep(&queue, -1, true, false);
  // Finish the positive-sized write first, then let the empty slice initialize.
  // This ordering has no data race and does not depend on scheduling or sleeps.
  release.set_value();
  empty_writer.join();
  // All explicit threads are joined before any assertion can return early.
  DiscountEnv::State state(queue.Wait());
  ASSERT_EQ(state["discount"_].Shape(), std::vector<std::size_t>({1}));
  ASSERT_EQ(state["info:players.env_id"_].Shape(),
            std::vector<std::size_t>({1}));
  ASSERT_EQ(state["info:env_id"_].Shape(), std::vector<std::size_t>({2}));
  ASSERT_EQ(state["done"_].Shape(), std::vector<std::size_t>({2}));
  EXPECT_EQ(static_cast<int>(state["info:env_id"_][0]), 0);
  EXPECT_EQ(static_cast<int>(state["info:env_id"_][1]), 1);
  EXPECT_TRUE(static_cast<bool>(state["done"_][0]));
  EXPECT_FALSE(static_cast<bool>(state["done"_][1]));
  EXPECT_EQ(static_cast<int>(state["info:players.env_id"_][0]), 1);
  EXPECT_FLOAT_EQ(static_cast<float>(state["discount"_][0]), 1.0F);
}
