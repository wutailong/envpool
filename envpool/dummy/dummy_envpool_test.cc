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

#include "envpool/dummy/dummy_envpool.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"

using DummyAction = typename dummy::DummyEnv::Action;
using DummyState = typename dummy::DummyEnv::State;

namespace {

class TimingDummyEnvPool : public dummy::DummyEnvPool {
 public:
  explicit TimingDummyEnvPool(const dummy::DummyEnvSpec& spec)
      : dummy::DummyEnvPool(spec) {}

  double SendDurationCount() const { return dur_send_.count(); }
  double RecvDurationCount() const { return dur_recv_.count(); }
  double SendAllDurationCount() const { return dur_send_all_.count(); }
};

struct ResetGate {
  std::mutex mutex;
  std::condition_variable condition;
  bool reset_entered{false};
  bool release_reset{false};
  bool reset_finished{false};
  bool environment_destroyed{false};
  bool reset_finished_before_environment_destroyed{false};
  bool destruction_started{false};
  bool destruction_finished{false};
};

struct GatedDummyEnvSpec : public dummy::DummyEnvSpec {
  std::shared_ptr<ResetGate> gate;

  GatedDummyEnvSpec(const dummy::DummyEnvSpec& spec,
                    std::shared_ptr<ResetGate> reset_gate)
      : dummy::DummyEnvSpec(spec), gate(std::move(reset_gate)) {}
};

class GatedDummyEnv : public dummy::DummyEnv {
 public:
  using Spec = GatedDummyEnvSpec;

  GatedDummyEnv(const Spec& spec, int env_id)
      : dummy::DummyEnv(spec, env_id), gate_(spec.gate) {}

  ~GatedDummyEnv() override {
    std::lock_guard<std::mutex> lock(gate_->mutex);
    gate_->environment_destroyed = true;
    gate_->reset_finished_before_environment_destroyed = gate_->reset_finished;
  }

  void Reset() override {
    const auto gate = gate_;
    {
      std::unique_lock<std::mutex> lock(gate->mutex);
      gate->reset_entered = true;
      gate->condition.notify_all();
      gate->condition.wait(lock, [&] { return gate->release_reset; });
    }
    dummy::DummyEnv::Reset();
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      gate->reset_finished = true;
    }
    gate->condition.notify_all();
  }

 private:
  std::shared_ptr<ResetGate> gate_;
};

void CheckPoolDestruction(bool pending_reset) {
  constexpr int kIterations = 32;
  const std::vector<std::pair<int, int>> shapes{{1, 2}, {1, 8}, {2, 3}};
  for (const auto& [num_envs, num_threads] : shapes) {
    SCOPED_TRACE(testing::Message()
                 << "num_envs=" << num_envs << ", num_threads=" << num_threads);
    auto config = dummy::DummyEnvSpec::kDefaultConfig;
    config["num_envs"_] = num_envs;
    config["batch_size"_] = num_envs;
    config["num_threads"_] = num_threads;
    config["max_num_players"_] = 1;
    config["seed"_] = 42;
    dummy::DummyEnvSpec spec(config);
    Array all_env_ids(Spec<int>({num_envs}));
    for (int i = 0; i < num_envs; ++i) {
      all_env_ids[i] = i;
    }
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      SCOPED_TRACE(iteration);
      dummy::DummyEnvPool envpool(spec);
      if (pending_reset) {
        envpool.Reset(all_env_ids);
      }
      // Destroy immediately, without a Recv or a concurrent public operation.
      // The destructor must wake idle workers and join any pending reset
      // workers before releasing their queues or environments.
    }
  }
}

}  // namespace

TEST(DummyEnvPoolTest, TimingAccumulatorsStartAtZero) {
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["max_num_players"_] = 1;
  dummy::DummyEnvSpec spec(config);
  TimingDummyEnvPool envpool(spec);
  // No public Reset, Send, or Recv has run yet.
  EXPECT_EQ(envpool.SendDurationCount(), 0.0);
  EXPECT_EQ(envpool.RecvDurationCount(), 0.0);
  EXPECT_EQ(envpool.SendAllDurationCount(), 0.0);
}

TEST(DummyEnvPoolTest, DiscountMatchesDoneForEveryPlayer) {
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 1;
  config["max_num_players"_] = 4;
  config["seed"_] = 10;
  dummy::DummyEnvSpec spec(config);
  dummy::DummyEnvPool envpool(spec);
  TArray env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  TArray list_action(Spec<double>({1, 6}));
  list_action.Fill(0.0);
  envpool.Reset(env_ids);
  for (int step = 0; step <= 10; ++step) {
    SCOPED_TRACE(step);
    DummyState state(envpool.Recv());
    const int num_players = step % 3 + 1;
    const bool done = step == 10;
    ASSERT_EQ(state["discount"_].Shape(0), num_players);
    ASSERT_EQ(state["info:players.env_id"_].Shape(0), num_players);
    EXPECT_EQ(static_cast<int>(state["elapsed_step"_][0]), step);
    EXPECT_EQ(static_cast<bool>(state["done"_][0]), done);
    // Step 1 has two nonterminal players; step 10 has two terminal players.
    for (int player = 0; player < num_players; ++player) {
      EXPECT_FLOAT_EQ(static_cast<float>(state["discount"_][player]),
                      done ? 0.0F : 1.0F);
      EXPECT_EQ(static_cast<int>(state["info:players.env_id"_][player]), 0);
    }
    if (!done) {
      DummyAction action;
      action["env_id"_] = state["info:env_id"_];
      action["players.env_id"_] = state["info:players.env_id"_];
      action["list_action"_] = list_action;
      action["players.action"_] = state["info:players.id"_];
      action["players.id"_] = state["info:players.id"_];
      envpool.Send(action);
    }
  }
}

TEST(DummyEnvPoolTest, ShutdownIdleWorkers) { CheckPoolDestruction(false); }

TEST(DummyEnvPoolTest, ShutdownWithPendingReset) { CheckPoolDestruction(true); }

TEST(DummyEnvPoolTest, ShutdownJoinsInFlightResetBeforeDestroyingEnvironment) {
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 2;
  config["max_num_players"_] = 1;
  const auto gate = std::make_shared<ResetGate>();
  GatedDummyEnvSpec spec(dummy::DummyEnvSpec(config), gate);
  auto envpool = std::make_unique<AsyncEnvPool<GatedDummyEnv>>(spec);
  Array env_ids(Spec<int>({1}));
  env_ids[0] = 0;
  envpool->Reset(env_ids);
  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    EXPECT_TRUE(gate->condition.wait_for(lock, std::chrono::seconds(5),
                                         [&] { return gate->reset_entered; }));
  }
  // Reset() has returned to the caller, but its worker is held inside the
  // environment. Transfer sole ownership before starting destruction.
  std::thread destroyer([pool = std::move(envpool), gate]() mutable {
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      gate->destruction_started = true;
    }
    gate->condition.notify_all();
    pool.reset();
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      gate->destruction_finished = true;
    }
    gate->condition.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    EXPECT_TRUE(gate->condition.wait_for(lock, std::chrono::seconds(5), [&] {
      return gate->destruction_started;
    }));
    EXPECT_FALSE(
        gate->condition.wait_for(lock, std::chrono::milliseconds(50),
                                 [&] { return gate->destruction_finished; }));
    EXPECT_FALSE(gate->reset_finished);
    EXPECT_FALSE(gate->environment_destroyed);
    // Release even if an expectation failed: no worker may outlive its gate.
    gate->release_reset = true;
  }
  gate->condition.notify_all();
  destroyer.join();
  std::lock_guard<std::mutex> lock(gate->mutex);
  EXPECT_TRUE(gate->reset_finished);
  EXPECT_TRUE(gate->environment_destroyed);
  EXPECT_TRUE(gate->reset_finished_before_environment_destroyed);
  EXPECT_TRUE(gate->destruction_finished);
}

TEST(DummyEnvPoolTest, SplitZeroAction) {
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  int num_envs = 4;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = 4;
  config["num_threads"_] = 1;
  config["seed"_] = 42;
  config["max_num_players"_] = 4;
  dummy::DummyEnvSpec spec(config);
  dummy::DummyEnvPool envpool(spec);
  Array all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  auto state_vec = envpool.Recv();
  // construct action
  std::vector<Array> raw_action({Array(Spec<int>({4})), Array(Spec<int>({8})),
                                 Array(Spec<double>({4, 6})),
                                 Array(Spec<int>({8})), Array(Spec<int>({8}))});
  DummyAction action(raw_action);
  for (int i = 0; i < 4; ++i) {
    action["env_id"_][i] = i;
    for (int j = 0; j < 6; ++j) {
      action["list_action"_][i][j] = 3.0 + i;
    }
  }
  std::vector<int> player_env_id({1, 2, 0, 2, 0, 1, 1, 2});
  for (int i = 0; i < 8; ++i) {
    action["players.env_id"_][i] = player_env_id[i];
  }
  // send
  envpool.Send(action);
  DummyState state(envpool.Recv());
  EXPECT_EQ(action["env_id"_].Shape(0), state["info:env_id"_].Shape(0));
  EXPECT_EQ(state["info:players.env_id"_].Shape(0), 8);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][i]), i);
  }
  auto obs = state["obs:raw"_];
  auto dyn = state["obs:dyn"_];
  auto peid = state["info:players.env_id"_];
  std::vector<int> counter({2, 3, 3, 0});
  for (int i = 0; i < 8; ++i) {
    int p = peid[i];
    EXPECT_EQ(static_cast<int>(obs(i, 1)), counter[p]);
    // check dyn
    const Container<int>& c = dyn[i];
    EXPECT_EQ(c->Shape(0), p + 1);
    auto* data = reinterpret_cast<int*>(c->Data());
    for (std::size_t j = 0; j < c->size; ++j) {
      EXPECT_EQ(data[j], p);
    }
  }
  // construct continuous action
  envpool.Reset(action["env_id"_]);
  envpool.Recv();
  player_env_id = std::vector<int>({0, 0, 0, 2, 2, 3, 3, 3});
  for (int i = 0; i < 8; ++i) {
    action["players.env_id"_][i] = player_env_id[i];
  }
  // send
  envpool.Send(action);
  state = DummyState(envpool.Recv());
  EXPECT_EQ(action["env_id"_].Shape(0), state["info:env_id"_].Shape(0));
  EXPECT_EQ(state["info:players.env_id"_].Shape(0), 8);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<int>(state["info:env_id"_][i]), i);
  }
  obs = state["obs:raw"_];
  dyn = state["obs:dyn"_];
  peid = state["info:players.env_id"_];
  counter = std::vector<int>({3, 0, 2, 3});
  for (int i = 0; i < 8; ++i) {
    int p = peid[i];
    EXPECT_EQ(static_cast<int>(obs(i, 1)), counter[p]);
    // check dyn
    const Container<int>& c = dyn[i];
    EXPECT_EQ(c->Shape(0), p + 1);
    auto* data = reinterpret_cast<int*>(c->Data());
    for (std::size_t j = 0; j < c->size; ++j) {
      EXPECT_EQ(data[j], p);
    }
  }
}

void Runner(int num_envs, int batch, int seed, int total_iter, int num_threads,
            int max_num_players) {
  LOG(INFO) << num_envs << " " << batch << " " << seed << " " << total_iter
            << " " << num_threads << " " << max_num_players;
  bool is_sync = num_envs == batch && max_num_players == 1;
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = batch;
  config["num_threads"_] = num_threads;
  config["seed"_] = seed;
  config["max_num_players"_] = max_num_players;
  std::vector<int> length;
  std::vector<int> counter;
  for (int i = 0; i < num_envs; ++i) {
    length.push_back(seed + i);
    counter.push_back(-1);
  }
  dummy::DummyEnvSpec spec(config);
  dummy::DummyEnvPool envpool(spec);
  TArray all_env_ids(Spec<int>({num_envs}));
  for (int i = 0; i < num_envs; ++i) {
    all_env_ids[i] = i;
  }
  envpool.Reset(all_env_ids);
  auto list_action = TArray(Spec<double>({num_envs, 6}));
  for (int i = 0; i < num_envs; ++i) {
    for (int j = 0; j < 6; ++j) {
      list_action[i][j] = 5.0 + i;
    }
  }
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < total_iter; ++i) {
    // recv
    DummyState state(envpool.Recv());
    // check state
    auto env_id = state["info:env_id"_];
    auto player_env_id = state["info:players.env_id"_];
    auto player_id = state["info:players.id"_];
    auto obs = state["obs:raw"_];
    auto dyn = state["obs:dyn"_];
    auto reward = state["reward"_];
    auto done = state["done"_];
    auto player_done = state["info:players.done"_];
    EXPECT_EQ(env_id.Shape(0), batch);
    int total_num_players = 0;
    for (int i = 0; i < batch; ++i) {
      int eid = env_id[i];
      if (is_sync) {
        EXPECT_EQ(eid, i);
      }
      ++counter[eid];
      int num_players =
          max_num_players <= 1 ? 1 : counter[eid] % (max_num_players - 1) + 1;
      total_num_players += num_players;
    }
    EXPECT_EQ(player_env_id.Shape(0), total_num_players);
    if (is_sync) {
      EXPECT_EQ(total_num_players, batch);
    }
    for (int i = 0; i < total_num_players; ++i) {
      int eid = player_env_id[i];
      int num_players =
          max_num_players <= 1 ? 1 : counter[eid] % (max_num_players - 1) + 1;
      int id = player_id[i];
      int state0 = obs(i, 0);
      int state1 = obs(i, 1);
      // float rew = reward[i];
      bool done_flag = player_done[i];
      EXPECT_LT(id, num_players);
      // EXPECT_LT(-rew, num_players);  <-- error caused by float(arr) := int
      EXPECT_EQ(done_flag, counter[eid] >= length[eid]);
      ASSERT_EQ(state0, counter[eid]) << eid;
      EXPECT_LE(state1, max_num_players);
      if (is_sync) {
        EXPECT_EQ(eid, i);
      }
      // check dyn
      const Container<int>& c = dyn[i];
      EXPECT_EQ(c->Shape(0), eid + 1);
      auto* data = reinterpret_cast<int*>(c->Data());
      EXPECT_EQ(data[0], eid);  // checking all is too expensive
    }

    for (int i = 0; i < batch; ++i) {
      int eid = env_id[i];
      if (counter[eid] >= length[eid]) {
        EXPECT_TRUE(static_cast<bool>(done[i]));
        counter[eid] = -1;
      } else {
        EXPECT_FALSE(static_cast<bool>(done[i]));
      }
    }
    // construct action
    DummyAction action;
    action["env_id"_] = env_id;
    action["players.env_id"_] = player_env_id;
    action["list_action"_] = list_action;
    action["players.action"_] = player_id;
    action["players.id"_] = player_id;
    // send
    envpool.Send(action);
  }
  std::chrono::duration<double> dur = std::chrono::system_clock::now() - start;
  double t = dur.count();
  double fps = (total_iter * batch) / t;
  LOG(INFO) << "time(s): " << t << ", FPS: " << fps;
}

TEST(DummyEnvPoolTest, SmokeSynchronousSinglePlayer) {
  Runner(1, 1, 3, 32, 1, 1);
}

TEST(DummyEnvPoolTest, SmokeAsynchronousSinglePlayer) {
  Runner(8, 4, 3, 64, 2, 1);
}

TEST(DummyEnvPoolTest, SmokeAsynchronousMultiPlayer) {
  Runner(8, 4, 3, 64, 2, 4);
}

TEST(DummyEnvPoolTest, SinglePlayer) {
  Runner(1, 1, 20, 100000, 1, 1);
  Runner(3, 1, 20, 100000, 1, 1);
  Runner(3, 1, 20, 100000, 3, 1);
  Runner(9, 4, 20, 100000, 1, 1);
  Runner(9, 4, 30, 100000, 4, 1);
  Runner(9, 4, 30, 100000, 9, 1);
  Runner(10, 10, 25, 100000, 0, 1);
}

TEST(DummyEnvPoolTest, MultiPlayers) {
  Runner(1, 1, 20, 100000, 1, 10);
  Runner(3, 1, 20, 100000, 1, 10);
  Runner(3, 1, 20, 100000, 3, 10);
  Runner(9, 4, 30, 100000, 1, 6);
  Runner(9, 4, 30, 100000, 4, 6);
  Runner(9, 4, 30, 100000, 9, 6);
  Runner(10, 10, 25, 100000, 0, 9);
}
