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

// Standalone test: compile this translation unit with the generated include
// overlay BEFORE the source include root. Run through the helper's bounded
// "run" command. Never link this instrumented queue into production clients.

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace shutdown_admission_test {
void AfterAdmission();
}  // namespace shutdown_admission_test

#include "envpool/core/action_buffer_queue.h"

namespace shutdown_admission_test {
namespace {

struct AdmissionGate {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t calls = 0;
  bool released = false;
};

// Set before thread creation and cleared only after every consumer joins.
AdmissionGate* active_gate = nullptr;

}  // namespace

// Called only by the generated test-only header, after its false stop check
// and before DequeueReady reserves or reads any action slot.
void AfterAdmission() {
  std::unique_lock<std::mutex> lock(active_gate->mutex);
  ++active_gate->calls;
  active_gate->changed.notify_all();
  active_gate->changed.wait(lock, [] { return active_gate->released; });
}

}  // namespace shutdown_admission_test

namespace {

using ActionSlice = ActionBufferQueue::ActionSlice;

class InspectableQueue : public ActionBufferQueue {
 public:
  using ActionBufferQueue::ActionBufferQueue;

  uint64_t Allocated() const { return alloc_ptr_.load(); }
  uint64_t Dequeued() const { return done_ptr_.load(); }
  ActionSlice Slot(std::size_t index) const { return queue_[index]; }
};

bool SameAction(const ActionSlice& lhs, const ActionSlice& rhs) {
  return lhs.env_id == rhs.env_id && lhs.order == rhs.order &&
         lhs.force_reset == rhs.force_reset;
}

}  // namespace

int main() {
  constexpr std::size_t kIdleConsumers = 7;
  const ActionSlice payload{0, 37, true};
  const ActionSlice untouched{-7, 91, false};
  shutdown_admission_test::AdmissionGate gate;
  shutdown_admission_test::active_gate = &gate;
  InspectableQueue queue(1);  // Two slots, eight consumers, one real action.
  std::atomic<int> stop{0};
  ActionSlice admitted_output = untouched;
  ActionSlice next_output = untouched;
  bool admitted_result = false;
  bool next_result = true;
  std::array<ActionSlice, kIdleConsumers> idle_outputs;
  idle_outputs.fill(untouched);
  std::array<bool, kIdleConsumers> idle_results;
  idle_results.fill(true);
  std::mutex started_mutex;
  std::condition_variable started_changed;
  std::size_t started = 0;
  int failures = 0;
  const auto expect = [&](bool condition, const char* message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  // The producer is quiescent before any consumer starts or stop is published.
  queue.EnqueueBulk({payload});
  const ActionSlice unused_slot = queue.Slot(1);
  std::thread admitted([&] {
    admitted_result = queue.DequeueOrStop(&admitted_output, stop);
    next_result = queue.DequeueOrStop(&next_output, stop);
  });
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.changed.wait(lock, [&] { return gate.calls != 0; });
  }

  // The real permit is now held by the admitted consumer, but no slot has
  // been reserved or read. Idle consumers have no real permits available.
  std::vector<std::thread> idle;
  idle.reserve(kIdleConsumers);
  for (std::size_t i = 0; i < kIdleConsumers; ++i) {
    idle.emplace_back([&, i] {
      {
        std::lock_guard<std::mutex> lock(started_mutex);
        ++started;
      }
      started_changed.notify_one();
      idle_results[i] = queue.DequeueOrStop(&idle_outputs[i], stop);
    });
  }
  {
    std::unique_lock<std::mutex> lock(started_mutex);
    started_changed.wait(lock, [&] { return started == kIdleConsumers; });
  }
  stop.store(1, std::memory_order_release);
  queue.WakeForShutdown(kIdleConsumers + 1);
  for (auto& consumer : idle) {
    consumer.join();
  }

  expect(queue.Allocated() == 1, "shutdown changed the allocation cursor");
  expect(queue.Dequeued() == 0, "idle consumers advanced the dequeue cursor");
  expect(queue.SizeApprox() == 1, "pending action disappeared before release");
  expect(SameAction(queue.Slot(0), payload), "shutdown changed real payload");
  expect(SameAction(queue.Slot(1), unused_slot),
         "shutdown changed unused slot");
  for (std::size_t i = 0; i < kIdleConsumers; ++i) {
    expect(!idle_results[i], "idle consumer reported a phantom action");
    expect(SameAction(idle_outputs[i], untouched), "idle output was modified");
  }

  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    expect(gate.calls == 1, "more than the real consumer passed admission");
    gate.released = true;
  }
  gate.changed.notify_all();
  admitted.join();
  // Queue, stop, outputs, and both gates stay alive through all consumer joins.
  shutdown_admission_test::active_gate = nullptr;

  expect(admitted_result, "already-admitted dequeue did not succeed");
  expect(SameAction(admitted_output, payload), "admitted payload was changed");
  expect(!next_result, "next dequeue reported a phantom action");
  expect(SameAction(next_output, untouched),
         "stopped next output was modified");
  expect(queue.Allocated() == 1, "allocation cursor changed after release");
  expect(queue.Dequeued() == 1, "dequeue cursor did not advance exactly once");
  expect(queue.SizeApprox() == 0,
         "admitted action was not consumed exactly once");
  expect(gate.calls == 1, "next dequeue crossed the stop admission boundary");
  expect(SameAction(queue.Slot(0), payload), "real slot was overwritten");
  expect(SameAction(queue.Slot(1), unused_slot), "unused slot was overwritten");
  if (failures != 0) {
    return 1;
  }
  std::cout
      << "PASS: admitted action survives stop; idle and next dequeues stop"
      << std::endl;
  return 0;
}
