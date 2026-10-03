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

#include "envpool/core/action_buffer_queue.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <queue>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "ThreadPool.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "envpool/core/dict.h"
#include "envpool/core/spec.h"

using ActionSlice = typename ActionBufferQueue::ActionSlice;

namespace {

class InspectableActionBufferQueue : public ActionBufferQueue {
 public:
  using ActionBufferQueue::ActionBufferQueue;

  uint64_t Allocated() const { return alloc_ptr_.load(); }
  uint64_t Dequeued() const { return done_ptr_.load(); }
  ActionSlice Slot(std::size_t index) const { return queue_[index]; }
};

void ExpectSameAction(const ActionSlice& actual, const ActionSlice& expected) {
  EXPECT_EQ(actual.env_id, expected.env_id);
  EXPECT_EQ(actual.order, expected.order);
  EXPECT_EQ(actual.force_reset, expected.force_reset);
}

void ExpectShutdownWakesConsumers(ActionBufferQueue* queue,
                                  std::atomic<int>* stop) {
  constexpr std::size_t kNumConsumers = 8;
  const ActionSlice untouched{.env_id = -7, .order = 91, .force_reset = true};
  std::array<ActionSlice, kNumConsumers> outputs;
  outputs.fill(untouched);
  std::array<bool, kNumConsumers> dequeued{};
  std::atomic<std::size_t> started{0};
  std::vector<std::thread> consumers;
  consumers.reserve(kNumConsumers);
  for (std::size_t i = 0; i < kNumConsumers; ++i) {
    consumers.emplace_back([&, i] {
      started.fetch_add(1);
      dequeued[i] = queue->DequeueOrStop(&outputs[i], *stop);
    });
  }
  while (started.load() != kNumConsumers) {
    std::this_thread::yield();
  }
  stop->store(1, std::memory_order_release);
  queue->WakeForShutdown(kNumConsumers);
  for (auto& consumer : consumers) {
    consumer.join();
  }
  // Keep the queue, stop flag, and output storage alive until every join.
  for (std::size_t i = 0; i < kNumConsumers; ++i) {
    SCOPED_TRACE(i);
    EXPECT_FALSE(dequeued[i]);
    ExpectSameAction(outputs[i], untouched);
  }
}

}  // namespace

TEST(ActionBufferQueueTest, ShutdownWakesEmptyQueueWithoutAdvancingCursors) {
  // More consumers than ring slots must not require enqueueing fake actions.
  InspectableActionBufferQueue queue(1);
  std::atomic<int> stop{0};
  ExpectShutdownWakesConsumers(&queue, &stop);
  EXPECT_EQ(queue.Allocated(), 0);
  EXPECT_EQ(queue.Dequeued(), 0);
  EXPECT_EQ(queue.SizeApprox(), 0);
}

TEST(ActionBufferQueueTest, ShutdownWakePreservesPendingAction) {
  InspectableActionBufferQueue queue(1);
  const ActionSlice action{.env_id = 0, .order = 37, .force_reset = true};
  queue.EnqueueBulk({action});
  const ActionSlice unused_slot = queue.Slot(1);
  // Stop before starting consumers so even the real item's permit cannot
  // advance the dequeue cursor or read its payload.
  std::atomic<int> stop{1};
  ExpectShutdownWakesConsumers(&queue, &stop);
  EXPECT_EQ(queue.Allocated(), 1);
  EXPECT_EQ(queue.Dequeued(), 0);
  EXPECT_EQ(queue.SizeApprox(), 1);
  ExpectSameAction(queue.Slot(0), action);
  ExpectSameAction(queue.Slot(1), unused_slot);
}

TEST(ActionBufferQueueTest, StopAwareDequeuePreservesActiveDelivery) {
  InspectableActionBufferQueue queue(3);
  std::atomic<int> stop{0};
  // Traverse the ring more than once using normal live-consumer permits.
  for (int batch = 0; batch < 4; ++batch) {
    std::vector<ActionSlice> actions;
    for (int env_id = 0; env_id < 3; ++env_id) {
      actions.push_back(ActionSlice{.env_id = env_id,
                                    .order = batch * 3 + env_id,
                                    .force_reset = (batch + env_id) % 2 == 0});
    }
    queue.EnqueueBulk(actions);
    EXPECT_EQ(queue.SizeApprox(), actions.size());
    for (const auto& expected : actions) {
      ActionSlice actual{};
      ASSERT_TRUE(queue.DequeueOrStop(&actual, stop));
      ExpectSameAction(actual, expected);
    }
    EXPECT_EQ(queue.SizeApprox(), 0);
  }
  EXPECT_EQ(queue.Allocated(), 12);
  EXPECT_EQ(queue.Dequeued(), 12);
}

TEST(ActionBufferQueueTest, Concurrent) {
  std::size_t num_envs = 1000;
  ActionBufferQueue queue(num_envs);
  std::srand(std::time(nullptr));
  std::size_t mul = 2000;
  std::vector<ActionSlice> actions;
  actions.reserve(num_envs);
  // enqueue all envs
  for (std::size_t i = 0; i < num_envs; ++i) {
    actions.push_back(ActionSlice{
        .env_id = static_cast<int>(i), .order = -1, .force_reset = false});
  }
  queue.EnqueueBulk(actions);
  std::vector<std::atomic<std::size_t>> flag(mul);
  std::vector<std::size_t> env_num(mul);
  for (std::size_t m = 0; m < mul; ++m) {
    flag[m] = 1;
    env_num[m] = std::rand() % (num_envs - 1) + 1;
  }

  std::thread send([&] {
    for (std::size_t m = 0; m < mul; ++m) {
      while (flag[m] == 1) {
      }
      actions.clear();
      for (std::size_t i = 0; i < env_num[m]; ++i) {
        actions.push_back(ActionSlice{
            .env_id = static_cast<int>(i), .order = -1, .force_reset = false});
      }
      queue.EnqueueBulk(actions);
    }
  });
  std::thread recv([&] {
    for (std::size_t m = 0; m < mul; ++m) {
      for (std::size_t i = 0; i < env_num[m]; ++i) {
        queue.Dequeue();
      }
      flag[m] = 0;
    }
  });
  recv.join();
  send.join();
  EXPECT_EQ(queue.SizeApprox(), num_envs);
}
