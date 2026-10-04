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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "envpool/core/circular_buffer.h"
#include "envpool/core/state_buffer_queue.h"

namespace {

// These tests execute only the repaired candidate. Every test has a watchdog,
// including its joins, so a regression fails the process instead of hanging.
class StockCloseTest : public ::testing::Test {
 protected:
  StockCloseTest()
      : watchdog_([this] {
          std::unique_lock<std::mutex> lock(mutex_);
          if (!condition_.wait_for(lock, std::chrono::seconds(30),
                                   [this] { return finished_; })) {
            std::fputs("stock-close test exceeded its 30-second deadline\n",
                       stderr);
            std::abort();
          }
        }) {}

  ~StockCloseTest() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
    }
    condition_.notify_all();
    watchdog_.join();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool finished_{false};
  std::thread watchdog_;
};

class Gate {
 public:
  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrivals_;
    condition_.notify_all();
    condition_.wait(lock, [this] { return open_; });
  }

  void WaitFor(std::size_t arrivals) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(10),
                             [&] { return arrivals_ >= arrivals; })) {
      std::fputs("stock-close gate was not reached\n", stderr);
      std::abort();
    }
  }

  void Open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t arrivals_{0};
  bool open_{false};
};

template <typename T>
T Await(std::future<T>* result) {
  if (result->wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    std::fputs("stock-close result was not ready\n", stderr);
    std::abort();
  }
  return result->get();
}

template <typename T>
class InspectableBuffer : public CircularBuffer<T> {
 public:
  using CircularBuffer<T>::CircularBuffer;
  uint64_t Head() const { return this->head_.load(); }
  uint64_t Tail() const { return this->tail_.load(); }
};

struct Counts {
  std::atomic<int> made{0};
  std::atomic<int> destroyed{0};
};

struct Tracked {
  explicit Tracked(Counts* counts) : counts(counts) { ++counts->made; }
  ~Tracked() { ++counts->destroyed; }
  Counts* counts;
};

// Candidate-only lifecycle simulation. The producer loop, mutex scope, stop,
// wake and joins match StateBufferQueue; gates control otherwise rare timing.
// Payload ownership replaces expensive arrays without changing stock capacity.
class CreatorModel {
 public:
  explicit CreatorModel(std::size_t capacity) : stock(capacity) {}
  ~CreatorModel() {
    Stop();
    Join();
  }

  void Start(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      creators.emplace_back([this] {
        before_loop();
        while (!stop) {
          auto buffer = std::make_unique<Tracked>(&counts);
          during_allocation();
          {
            std::scoped_lock lock(put_mutex);
            if (!stock.PutOrStop(std::move(buffer), stop)) {
              break;
            }
          }
        }
      });
    }
  }

  void Stop() {
    if (!stop.exchange(true)) {
      stock.WakeProducersForStop(creators.size());
    }
  }

  void Join() {
    for (auto& creator : creators) {
      if (creator.joinable()) {
        creator.join();
      }
    }
  }

  Counts counts;
  InspectableBuffer<std::unique_ptr<Tracked>> stock;
  std::mutex put_mutex;
  std::vector<std::thread> creators;
  std::atomic<bool> stop{false};
  std::function<void()> before_loop = [] {};
  std::function<void()> during_allocation = [] {};
};

TEST_F(StockCloseTest, NormalFifoAcrossPositiveCapacities) {
  for (std::size_t capacity : {1, 2, 6}) {
    InspectableBuffer<int> buffer(capacity);
    std::atomic<bool> stop{false};
    int expected = 0;
    for (int round = 0; round < 20; ++round) {
      for (std::size_t i = 0; i < capacity; ++i) {
        EXPECT_TRUE(buffer.PutOrStop(expected + i, stop));
      }
      for (std::size_t i = 0; i < capacity; ++i) {
        int value = -1;
        if (i % 2 == 0) {
          value = buffer.Get();
        } else {
          EXPECT_TRUE(buffer.TryGet(&value));
        }
        EXPECT_EQ(value, expected++);
      }
      int value = -1;
      EXPECT_FALSE(buffer.TryGet(&value));
    }
    // The existing ordinary Put path still interoperates with both reads.
    buffer.Put(expected);
    EXPECT_EQ(buffer.Get(), expected);
    EXPECT_EQ(buffer.Head(), buffer.Tail());
  }
}

TEST_F(StockCloseTest, FullStockCancellationDoesNotMoveOrPublish) {
  InspectableBuffer<std::unique_ptr<int>> buffer(1);
  buffer.Put(std::make_unique<int>(7));
  std::atomic<bool> stop{false};
  std::promise<void> entered;
  auto started = entered.get_future();
  auto owner = std::make_unique<int>(9);
  auto producer = std::async(std::launch::async, [&] {
    entered.set_value();
    return buffer.PutOrStop(std::move(owner), stop);
  });
  Await(&started);
  // With a full stock no producer can succeed. This does not rely on sleeps
  // for correctness; Stop covers both already-blocked and late wait entry.
  EXPECT_EQ(producer.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  stop = true;
  buffer.WakeProducersForStop(1);
  EXPECT_FALSE(Await(&producer));
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(*owner, 9);
  EXPECT_EQ(buffer.Tail(), 1);
  EXPECT_EQ(*buffer.Get(), 7);
  std::unique_ptr<int> value;
  EXPECT_FALSE(buffer.TryGet(&value));
  EXPECT_EQ(buffer.Head(), 1);
}

TEST_F(StockCloseTest, SerializedCreatorsAllCancelWithoutConsumer) {
  constexpr std::size_t kCreators = 4;
  CreatorModel model(1);
  model.stock.Put(std::make_unique<Tracked>(&model.counts));
  Gate allocated;
  model.during_allocation = [&] { allocated.ArriveAndWait(); };
  model.Start(kCreators);
  allocated.WaitFor(kCreators);
  allocated.Open();
  model.Stop();
  model.Join();
  EXPECT_EQ(model.stock.Tail(), 1);
  EXPECT_EQ(model.counts.made, kCreators + 1);
  EXPECT_EQ(model.counts.destroyed, kCreators);
  auto value = model.stock.Get();
  value.reset();
  EXPECT_EQ(model.counts.destroyed, kCreators + 1);
  EXPECT_FALSE(model.stock.TryGet(&value));
}

TEST_F(StockCloseTest, LateCreatorsExitWithoutAllocating) {
  CreatorModel model(2);
  Gate before_loop;
  model.before_loop = [&] { before_loop.ArriveAndWait(); };
  model.Start(4);
  before_loop.WaitFor(4);
  model.Stop();
  before_loop.Open();
  model.Join();
  EXPECT_EQ(model.counts.made, 0);
  EXPECT_EQ(model.stock.Tail(), 0);
  std::unique_ptr<Tracked> value;
  EXPECT_FALSE(model.stock.TryGet(&value));
}

TEST_F(StockCloseTest, CurrentAllocationsRemainOwnedThroughJoin) {
  CreatorModel model(2);
  Gate allocated;
  model.during_allocation = [&] { allocated.ArriveAndWait(); };
  model.Start(4);
  allocated.WaitFor(4);
  model.Stop();
  EXPECT_EQ(model.counts.made, 4);
  EXPECT_EQ(model.counts.destroyed, 0);
  auto joined = std::async(std::launch::async, [&] { model.Join(); });
  EXPECT_EQ(joined.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  allocated.Open();
  Await(&joined);
  EXPECT_EQ(model.counts.destroyed, 4);
  EXPECT_EQ(model.stock.Tail(), 0);
  std::unique_ptr<Tracked> value;
  EXPECT_FALSE(model.stock.TryGet(&value));
}

TEST_F(StockCloseTest, TerminalWakePreservesDrainAndReturnedOwnership) {
  Counts counts;
  std::unique_ptr<Tracked> retained;
  {
    InspectableBuffer<std::unique_ptr<Tracked>> buffer(2);
    std::atomic<bool> stop{false};
    EXPECT_TRUE(buffer.PutOrStop(std::make_unique<Tracked>(&counts), stop));
    EXPECT_TRUE(buffer.PutOrStop(std::make_unique<Tracked>(&counts), stop));
    stop = true;
    buffer.WakeProducersForStop(4);
    EXPECT_TRUE(buffer.TryGet(&retained));
    auto other = buffer.Get();
    other.reset();
    std::unique_ptr<Tracked> extra;
    EXPECT_FALSE(buffer.TryGet(&extra));
    EXPECT_EQ(buffer.Head(), 2);
    EXPECT_EQ(buffer.Tail(), 2);
    EXPECT_EQ(counts.destroyed, 1);
  }
  // A normally received owning value survives destruction of its queue.
  EXPECT_EQ(counts.destroyed, 1);
  retained.reset();
  EXPECT_EQ(counts.destroyed, 2);
}

struct GatedAssignment {
  int value{0};
  Gate* admitted{nullptr};

  GatedAssignment& operator=(GatedAssignment&& other) {
    if (other.admitted != nullptr) {
      other.admitted->ArriveAndWait();
    }
    value = other.value;
    return *this;
  }

  GatedAssignment() = default;
  GatedAssignment(int value, Gate* gate) : value(value), admitted(gate) {}
  GatedAssignment(GatedAssignment&&) = default;
};

TEST_F(StockCloseTest, RealPreStopAdmissionMayFinishAfterStop) {
  InspectableBuffer<GatedAssignment> buffer(1);
  std::atomic<bool> stop{false};
  Gate admitted;
  auto producer = std::async(std::launch::async, [&] {
    return buffer.PutOrStop(GatedAssignment(17, &admitted), stop);
  });
  admitted.WaitFor(1);
  // Slot assignment proves the producer passed its post-wait stop check
  // using a genuine free-slot permit, before terminal wakes were posted.
  stop = true;
  buffer.WakeProducersForStop(1);
  admitted.Open();
  EXPECT_TRUE(Await(&producer));
  EXPECT_EQ(buffer.Get().value, 17);
  GatedAssignment value;
  EXPECT_FALSE(buffer.TryGet(&value));
  EXPECT_EQ(buffer.Head(), 1);
  EXPECT_EQ(buffer.Tail(), 1);
}

TEST_F(StockCloseTest, StateQueueImmediateScopeExit) {
  for (std::size_t batch : {1, 2, 4}) {
    for (int iteration = 0; iteration < 8; ++iteration) {
      StateBufferQueue queue(
          batch, batch * 3, 1,
          std::vector<ShapeSpec>{ShapeSpec(sizeof(int), {1})});
    }
  }
}

TEST_F(StockCloseTest, StateQueueClosesAfterCompletedNormalReceives) {
  for (int iteration = 0; iteration < 20; ++iteration) {
    StateBufferQueue queue(1, 1, 1,
                           std::vector<ShapeSpec>{ShapeSpec(sizeof(int), {1})});
    for (int step = 0; step < 8; ++step) {
      auto slice = queue.Allocate(1);
      slice.arr[0] = step;
      slice.done_write();
      auto output = queue.Wait();
      ASSERT_EQ(output.size(), 1);
      EXPECT_EQ(*reinterpret_cast<int*>(output[0].Data()), step);
    }
  }
}

}  // namespace
