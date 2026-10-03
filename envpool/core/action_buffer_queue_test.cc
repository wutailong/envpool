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

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <future>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using ActionSlice = typename ActionBufferQueue::ActionSlice;

namespace {

class SeededActionBufferQueue : public ActionBufferQueue {
 public:
  SeededActionBufferQueue(std::size_t num_envs, std::uint64_t position)
      : ActionBufferQueue(num_envs) {
    alloc_ptr_.store(position);
    done_ptr_.store(position);
  }
};

class InspectableActionBufferQueue : public ActionBufferQueue {
 public:
  using ActionBufferQueue::ActionBufferQueue;

  std::uint64_t Allocated() const {
    return alloc_ptr_.load(std::memory_order_relaxed);
  }

  std::uint64_t Dequeued() const {
    return done_ptr_.load(std::memory_order_relaxed);
  }

  ActionSlice Slot(std::size_t index) const { return queue_[index]; }
};

// ActionBufferQueue relies on its caller to bound outstanding actions. Reserve
// an entire batch before enqueueing, and return credits only after Dequeue has
// copied its payload. In particular, SizeApprox is not a capacity reservation.
class QueueCredits {
 public:
  explicit QueueCredits(std::size_t capacity) : available_(capacity) {}

  void Acquire(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [&] { return available_ >= count; });
    available_ -= count;
  }

  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++available_;
    }
    ready_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::size_t available_;
};

ActionSlice MakeAction(int order, int num_envs) {
  return ActionSlice{.env_id = order % num_envs,
                     .order = order,
                     .force_reset = order % 3 == 0};
}

void ExpectAction(const ActionSlice& actual, int order, int num_envs) {
  const auto expected = MakeAction(order, num_envs);
  EXPECT_EQ(actual.env_id, expected.env_id) << "order " << order;
  EXPECT_EQ(actual.order, expected.order);
  EXPECT_EQ(actual.force_reset, expected.force_reset) << "order " << order;
}

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

TEST(ActionBufferQueueTest, SmallRingWraparoundPreservesPayloadsAndCounters) {
  for (const int num_envs : {1, 3}) {
    SCOPED_TRACE(num_envs);
    InspectableActionBufferQueue queue(num_envs);
    const int capacity = 2 * num_envs;
    int next_order = 0;
    EXPECT_EQ(queue.SizeApprox(), 0);
    for (int cycle = 0; cycle < 1024; ++cycle) {
      const int count = 1 + cycle % capacity;
      std::vector<ActionSlice> actions;
      for (int i = 0; i < count; ++i) {
        actions.push_back(MakeAction(next_order + i, num_envs));
      }
      queue.EnqueueBulk(actions);
      EXPECT_EQ(queue.SizeApprox(), count);
      EXPECT_EQ(queue.Allocated(), next_order + count);
      EXPECT_EQ(queue.Dequeued(), next_order);
      for (int i = 0; i < count; ++i) {
        ExpectAction(queue.Dequeue(), next_order + i, num_envs);
        EXPECT_EQ(queue.SizeApprox(), count - i - 1);
      }
      next_order += count;
      EXPECT_EQ(queue.Allocated(), next_order);
      EXPECT_EQ(queue.Dequeued(), next_order);
    }
    queue.EnqueueBulk({});
    EXPECT_EQ(queue.SizeApprox(), 0);
    EXPECT_EQ(queue.Allocated(), next_order);
    EXPECT_EQ(queue.Dequeued(), next_order);
  }
}

TEST(ActionBufferQueueTest, BoundedMultipleProducersAndConsumersExactlyOnce) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kActionsPerProducer = 1024;
  constexpr int kTotal = kProducers * kActionsPerProducer;
  for (const int num_envs : {1, 3, 4}) {
    SCOPED_TRACE(num_envs);
    InspectableActionBufferQueue queue(num_envs);
    const int capacity = 2 * num_envs;
    QueueCredits credits(capacity);
    std::promise<void> start;
    const auto started = start.get_future().share();
    std::vector<std::vector<ActionSlice>> received(kConsumers);
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    for (int consumer = 0; consumer < kConsumers; ++consumer) {
      consumers.emplace_back([&, consumer] {
        auto& actions = received[consumer];
        actions.reserve(kTotal / kConsumers);
        started.wait();
        for (int i = 0; i < kTotal / kConsumers; ++i) {
          actions.push_back(queue.Dequeue());
          if ((i + consumer) % 17 == 0) {
            std::this_thread::yield();
          }
          credits.Release();
        }
      });
    }
    for (int producer = 0; producer < kProducers; ++producer) {
      producers.emplace_back([&, producer] {
        started.wait();
        int sent = 0;
        while (sent < kActionsPerProducer) {
          const int count =
              std::min(1 + (sent + producer) % std::min(capacity, 3),
                       kActionsPerProducer - sent);
          std::vector<ActionSlice> actions;
          for (int i = 0; i < count; ++i) {
            actions.push_back(MakeAction(
                producer * kActionsPerProducer + sent + i, num_envs));
          }
          credits.Acquire(count);
          if ((sent + producer) % 11 == 0) {
            std::this_thread::yield();
          }
          queue.EnqueueBulk(actions);
          sent += count;
        }
      });
    }
    start.set_value();
    for (auto& producer : producers) {
      producer.join();
    }
    for (auto& consumer : consumers) {
      consumer.join();
    }

    // All worker state is inspected after joining. Every consumer has a fixed
    // amount of work, so corrupt payloads cannot control termination.
    std::vector<ActionSlice> all_actions;
    for (const auto& actions : received) {
      all_actions.insert(all_actions.end(), actions.begin(), actions.end());
    }
    ASSERT_EQ(all_actions.size(), kTotal);
    std::sort(all_actions.begin(), all_actions.end(),
              [](const ActionSlice& lhs, const ActionSlice& rhs) {
                return lhs.order < rhs.order;
              });
    for (int i = 0; i < kTotal; ++i) {
      ExpectAction(all_actions[i], i, num_envs);
    }
    EXPECT_EQ(queue.SizeApprox(), 0);
    EXPECT_EQ(queue.Allocated(), kTotal);
    EXPECT_EQ(queue.Dequeued(), kTotal);
    // Destruction happens only after all producers and consumers are joined.
  }
}

TEST(ActionBufferQueueTest, DelayedConsumerKeepsPayloadAcrossRingReuse) {
  constexpr int kTotal = 1024;
  InspectableActionBufferQueue queue(1);
  queue.EnqueueBulk({MakeAction(0, 1), MakeAction(1, 1)});
  std::promise<void> copied;
  auto first_copied = copied.get_future();
  std::promise<void> finish;
  auto may_finish = finish.get_future();
  ActionSlice delayed{};
  std::vector<ActionSlice> completed(kTotal);
  std::atomic<int> next_completion{0};
  std::thread slow_consumer([&] {
    delayed = queue.Dequeue();
    copied.set_value();
    may_finish.wait();
    completed[next_completion.fetch_add(1)] = delayed;
  });
  first_copied.wait();

  // Hold the first consumer after its copy, not inside Dequeue. The other
  // consumer completes subsequent work first while the two-slot ring wraps.
  // At most two actions are outstanding, including the delayed consumer's.
  for (int i = 1; i < kTotal; ++i) {
    if (i > 1) {
      queue.EnqueueBulk({MakeAction(i, 1)});
    }
    completed[next_completion.fetch_add(1)] = queue.Dequeue();
  }
  finish.set_value();
  slow_consumer.join();

  EXPECT_EQ(next_completion.load(), kTotal);
  for (int i = 0; i < kTotal - 1; ++i) {
    ExpectAction(completed[i], i + 1, 1);
  }
  ExpectAction(completed.back(), 0, 1);
  ExpectAction(delayed, 0, 1);
  EXPECT_EQ(queue.SizeApprox(), 0);
  EXPECT_EQ(queue.Allocated(), kTotal);
  EXPECT_EQ(queue.Dequeued(), kTotal);
}

TEST(ActionBufferQueueTest, ConcurrentProducerBulksRemainContiguous) {
  constexpr int kNumEnvs = 5;
  constexpr std::array<int, 3> kBatchSizes{1, 2, 4};
  constexpr std::array<int, 3> kBatchStarts{0, 1, 3};
  constexpr int kPerRound = 7;
  InspectableActionBufferQueue queue(kNumEnvs);
  for (int round = 0; round < 257; ++round) {
    SCOPED_TRACE(round);
    const int first = round * kPerRound;
    std::promise<void> start;
    const auto started = start.get_future().share();
    std::vector<std::thread> producers;
    for (int producer = 0; producer < 3; ++producer) {
      producers.emplace_back([&, producer] {
        std::vector<ActionSlice> actions;
        for (int i = 0; i < kBatchSizes[producer]; ++i) {
          actions.push_back(
              MakeAction(first + kBatchStarts[producer] + i, kNumEnvs));
        }
        started.wait();
        queue.EnqueueBulk(actions);
      });
    }
    start.set_value();
    for (auto& producer : producers) {
      producer.join();
    }
    // Seven total actions fit in the ten-slot ring. Each round is fully
    // drained before refilling; seven-step advances visit every ring offset.
    EXPECT_EQ(queue.SizeApprox(), kPerRound);
    EXPECT_EQ(queue.Allocated(), first + kPerRound);
    EXPECT_EQ(queue.Dequeued(), first);
    std::array<ActionSlice, kPerRound> actual;
    for (auto& action : actual) {
      action = queue.Dequeue();
    }
    std::array<int, 3> seen{};
    for (int pos = 0; pos < kPerRound;) {
      ASSERT_GE(actual[pos].order, first);
      ASSERT_LT(actual[pos].order, first + kPerRound);
      const int batch_start = actual[pos].order - first;
      const auto found =
          std::find(kBatchStarts.begin(), kBatchStarts.end(), batch_start);
      ASSERT_NE(found, kBatchStarts.end());
      const auto producer = found - kBatchStarts.begin();
      const int count = kBatchSizes[producer];
      ASSERT_LE(pos + count, kPerRound);
      ++seen[producer];
      for (int i = 0; i < count; ++i) {
        ExpectAction(actual[pos + i], first + batch_start + i, kNumEnvs);
      }
      pos += count;
    }
    for (const int count : seen) {
      EXPECT_EQ(count, 1);
    }
    EXPECT_EQ(queue.SizeApprox(), 0);
    EXPECT_EQ(queue.Allocated(), first + kPerRound);
    EXPECT_EQ(queue.Dequeued(), first + kPerRound);
  }
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

TEST(ActionBufferQueueTest, BulkWrapPreservesOrderWithNonPowerOfTwoCapacity) {
  // Three environments give a six-entry ring. Mixed bulk sizes cross every
  // ring position, including exact-end and full-capacity writes.
  constexpr std::size_t kCapacity = 6;
  ActionBufferQueue queue(kCapacity / 2);
  std::queue<ActionSlice> expected;
  int next = 0;
  auto enqueue = [&](std::size_t count) {
    std::vector<ActionSlice> actions;
    for (std::size_t i = 0; i < count; ++i) {
      ActionSlice action{
          .env_id = next, .order = 1000 - next, .force_reset = next % 3 == 0};
      ++next;
      actions.push_back(action);
      expected.push(action);
    }
    queue.EnqueueBulk(actions);
    EXPECT_EQ(queue.SizeApprox(), expected.size());
  };
  auto dequeue = [&](std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      const auto actual = queue.Dequeue();
      const auto action = expected.front();
      expected.pop();
      EXPECT_EQ(actual.env_id, action.env_id);
      EXPECT_EQ(actual.order, action.order);
      EXPECT_EQ(actual.force_reset, action.force_reset);
      EXPECT_EQ(queue.SizeApprox(), expected.size());
    }
  };

  enqueue(5);
  dequeue(4);
  enqueue(4);  // Starts at index 5 and wraps while one old entry is live.
  dequeue(3);
  enqueue(4);  // Starts at index 3 and fills the ring exactly.
  dequeue(kCapacity);
  for (std::size_t count : {0U, 6U, 1U, 5U, 2U, 4U, 3U, 6U, 0U}) {
    enqueue(count);
    dequeue(count);
  }
  EXPECT_TRUE(expected.empty());
  EXPECT_EQ(queue.SizeApprox(), 0U);
}

TEST(ActionBufferQueueTest, ConcurrentConsumersPreserveEveryWrappedAction) {
  constexpr std::size_t kConsumers = 3;
  constexpr std::size_t kBatch = 5;
  ActionBufferQueue queue(3);  // Capacity six, deliberately not a power of two.
  for (int round = 0; round < 18; ++round) {
    const int first = round * static_cast<int>(kBatch);
    std::vector<ActionSlice> actions;
    for (std::size_t i = 0; i < kBatch; ++i) {
      const int id = first + static_cast<int>(i);
      actions.push_back(ActionSlice{
          .env_id = id, .order = 1000 + id, .force_reset = id % 2 == 0});
    }
    // Drain each bounded batch before refilling, respecting queue capacity.
    // Five-entry batches advance the starting position around the whole ring.
    queue.EnqueueBulk(actions);
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::array<std::future<std::vector<ActionSlice>>, kConsumers> consumers;
    for (std::size_t consumer = 0; consumer < kConsumers; ++consumer) {
      consumers[consumer] = std::async(std::launch::async, [&, consumer]() {
        std::vector<ActionSlice> received;
        const std::size_t count = consumer < 2 ? 2 : 1;
        ready.wait();
        for (std::size_t i = 0; i < count; ++i) {
          received.push_back(queue.Dequeue());
        }
        return received;
      });
    }
    start.set_value();
    std::array<std::vector<ActionSlice>, kConsumers> received;
    for (std::size_t consumer = 0; consumer < kConsumers; ++consumer) {
      received[consumer] = consumers[consumer].get();
    }
    // Join every async consumer before assertions can leave the test early.
    std::array<int, kBatch> seen{};
    for (const auto& actions_received : received) {
      int previous = first - 1;
      for (const auto& action : actions_received) {
        EXPECT_GT(action.env_id, previous);
        previous = action.env_id;
        EXPECT_EQ(action.order, 1000 + action.env_id);
        EXPECT_EQ(action.force_reset, action.env_id % 2 == 0);
        const int index = action.env_id - first;
        ASSERT_GE(index, 0);
        ASSERT_LT(index, static_cast<int>(kBatch));
        ++seen[index];
      }
    }
    for (int count : seen) {
      EXPECT_EQ(count, 1);
    }
    EXPECT_EQ(queue.SizeApprox(), 0U);
  }
}

TEST(ActionBufferQueueTest, BulkWrapMatchesUnsignedCounterRollover) {
  constexpr std::size_t kCapacity = 6;
  constexpr std::uint64_t kStart =
      std::numeric_limits<std::uint64_t>::max() - 2;
  SeededActionBufferQueue queue(kCapacity / 2, kStart);
  EXPECT_EQ(queue.SizeApprox(), 0U);
  int next = 0;
  // The first batch crosses UINT64_MAX using distinct ring slots 1, 2, 3, 0.
  // Later batches verify that normal wrapping resumes at the new ticket.
  for (std::size_t count : {4U, 6U, 5U, 6U}) {
    std::vector<ActionSlice> actions;
    for (std::size_t i = 0; i < count; ++i) {
      actions.push_back(ActionSlice{
          .env_id = next, .order = 1000 - next, .force_reset = next % 2 == 0});
      ++next;
    }
    queue.EnqueueBulk(actions);
    EXPECT_EQ(queue.SizeApprox(), count);
    for (std::size_t i = 0; i < actions.size(); ++i) {
      const auto actual = queue.Dequeue();
      EXPECT_EQ(actual.env_id, actions[i].env_id);
      EXPECT_EQ(actual.order, actions[i].order);
      EXPECT_EQ(actual.force_reset, actions[i].force_reset);
      EXPECT_EQ(queue.SizeApprox(), count - i - 1);
    }
  }
  EXPECT_EQ(queue.SizeApprox(), 0U);
}

TEST(ActionBufferQueueTest, EmptyBulkIsSafeWithZeroCapacity) {
  ActionBufferQueue queue(0);
  EXPECT_EQ(queue.SizeApprox(), 0U);
  EXPECT_NO_THROW(queue.EnqueueBulk({}));
  EXPECT_EQ(queue.SizeApprox(), 0U);
  EXPECT_NO_THROW(queue.EnqueueBulk({}));
  EXPECT_EQ(queue.SizeApprox(), 0U);
}
