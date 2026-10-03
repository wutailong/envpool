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
#include <cstddef>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/core/env_spec.h"
#include "envpool/core/spec.h"
#include "envpool/core/state_buffer.h"
#include "envpool/core/state_buffer_queue.h"

namespace {

struct PayloadCounts {
  std::atomic<int> created{0};
  std::atomic<int> destroyed{0};
  std::atomic<int> finished{0};
};

// Each test creates exactly one payload. Neither this recorder nor the tests
// retain an owning copy of its inner Array storage.
struct PayloadProbe {
  explicit PayloadProbe(std::shared_ptr<PayloadCounts> counters)
      : counts(std::move(counters)), resume(release.get_future()) {}

  ~PayloadProbe() {
    // Negative-control cleanup, after every pool, output Array and extracted
    // Container is gone. A zero destruction count proves the recorded TArray
    // is still live: its custom deleter has no other inner-storage owners.
    // Delete the TArray itself, never the already-freed outer Container slot.
    // Assertions run before this fallback, so it cannot hide a cleanup failure.
    if (counts->created.load() == 1 && counts->destroyed.load() == 0) {
      delete recorded;
    }
  }

  Container<int> MakePayload() {
    Array storage(::Spec<int>({1}), reinterpret_cast<char*>(new int[1]{42}),
                  [counters = counts](char* data) {
                    delete[] reinterpret_cast<int*>(data);
                    ++counters->destroyed;
                  });
    auto payload = std::make_unique<TArray<int>>(std::move(storage));
    recorded = payload.get();
    ++counts->created;
    return payload;
  }

  std::shared_ptr<PayloadCounts> counts;
  std::promise<void> created;
  std::promise<void> release;
  std::future<void> resume;
  TArray<int>* recorded{nullptr};
};

class ContainerEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("probe"_.Bind(std::shared_ptr<PayloadProbe>()));
  }

  template <typename Config>
  static decltype(auto) StateSpec(const Config&) {
    return MakeDict("obs"_.Bind(Spec<Container<int>>({}, Spec<int>({1}))));
  }

  template <typename Config>
  static decltype(auto) ActionSpec(const Config&) {
    return MakeDict();
  }
};

using ContainerEnvSpec = EnvSpec<ContainerEnvFns>;

class ContainerEnv : public Env<ContainerEnvSpec> {
 public:
  ContainerEnv(const Spec& spec, int env_id)
      : Env<ContainerEnvSpec>(spec, env_id), probe_(spec.config["probe"_]) {}

  void Reset() override {
    auto state = Allocate();
    state["reward"_].Fill(0.0F);
    Container<int>& payload = state["obs"_];
    payload = probe_->MakePayload();
    probe_->created.set_value();
    probe_->resume.wait();
    ++probe_->counts->finished;
  }

  bool IsDone() override { return false; }

 private:
  std::shared_ptr<PayloadProbe> probe_;
};

using ContainerPool = AsyncEnvPool<ContainerEnv>;

ContainerEnvSpec MakeSpec(const std::shared_ptr<PayloadProbe>& probe,
                          int num_envs = 1) {
  auto config = ContainerEnvSpec::kDefaultConfig;
  config["num_envs"_] = num_envs;
  config["batch_size"_] = num_envs;
  config["num_threads"_] = 1;
  config["probe"_] = probe;
  return ContainerEnvSpec(config);
}

Array OneEnvId() {
  Array ids(Spec<int>({1}));
  ids[0] = 0;
  return ids;
}

}  // namespace

TEST(ContainerOutputTest, QueuedOutputReclaimedOnPoolDestruction) {
  // Also cover a partly populated batch, whose unused slots were never handed
  // to Env::Allocate. No Recv call is needed for normal pool shutdown.
  for (int num_envs : {1, 2}) {
    SCOPED_TRACE(num_envs);
    auto counts = std::make_shared<PayloadCounts>();
    auto probe = std::make_shared<PayloadProbe>(counts);
    auto created = probe->created.get_future();
    auto pool = std::make_unique<ContainerPool>(MakeSpec(probe, num_envs));
    pool->Reset(OneEnvId());
    const bool payload_created =
        created.wait_for(std::chrono::seconds(10)) == std::future_status::ready;

    // Transfer the sole pool owner. Only the test gate is touched concurrently
    // with destruction; there are no concurrent calls to the public pool API.
    std::thread destroyer([pool = std::move(pool)]() mutable { pool.reset(); });
    probe->release.set_value();
    destroyer.join();
    // Always release the worker and join destruction before fatal assertions.
    ASSERT_TRUE(payload_created);
    ASSERT_EQ(counts->created.load(), 1);
    EXPECT_EQ(counts->finished.load(), 1);
    EXPECT_EQ(counts->destroyed.load(), 1);
  }
}

TEST(ContainerOutputTest, ReceivedOutputLivesUntilLastArrayOwner) {
  auto counts = std::make_shared<PayloadCounts>();
  auto probe = std::make_shared<PayloadProbe>(counts);
  probe->release.set_value();
  auto pool = std::make_unique<ContainerPool>(MakeSpec(probe));
  pool->Reset(OneEnvId());
  auto output = pool->Recv();
  pool.reset();
  ASSERT_EQ(counts->created.load(), 1);
  ASSERT_EQ(counts->destroyed.load(), 0);
  {
    ContainerEnv::State state(output);
    ASSERT_EQ(state["obs"_].size, 1);
    Container<int>& payload = state["obs"_][0];
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(static_cast<int>((*payload)[0]), 42);
  }
  // These are owning Array copies, not non-owning Slice/operator[] views.
  auto last_owner = output;
  output.clear();
  EXPECT_EQ(counts->destroyed.load(), 0);
  last_owner.clear();
  EXPECT_EQ(counts->destroyed.load(), 1);
}

TEST(ContainerOutputTest, ExtractedPayloadOwnsItsStorage) {
  auto counts = std::make_shared<PayloadCounts>();
  auto probe = std::make_shared<PayloadProbe>(counts);
  probe->release.set_value();
  Container<int> extracted;
  {
    auto pool = std::make_unique<ContainerPool>(MakeSpec(probe));
    pool->Reset(OneEnvId());
    auto output = pool->Recv();
    pool.reset();
    ASSERT_EQ(counts->created.load(), 1);
    ASSERT_EQ(counts->destroyed.load(), 0);
    {
      ContainerEnv::State state(output);
      ASSERT_EQ(state["obs"_].size, 1);
      Container<int>& payload = state["obs"_][0];
      extracted = std::move(payload);
      // Leave the moved-from null slot alive for its outer-owner destructor,
      // matching the typed-backing Python conversion contract.
    }
  }
  ASSERT_EQ(counts->destroyed.load(), 0);
  ASSERT_NE(extracted, nullptr);
  EXPECT_EQ(static_cast<int>((*extracted)[0]), 42);
  extracted.reset();
  EXPECT_EQ(counts->destroyed.load(), 1);
}

namespace {

Container<int> MakeCountedPayload(
    const std::shared_ptr<PayloadCounts>& counts) {
  Array storage(Spec<int>({1}), reinterpret_cast<char*>(new int[1]{42}),
                [counts](char* data) {
                  delete[] reinterpret_cast<int*>(data);
                  ++counts->destroyed;
                });
  auto payload = std::make_unique<TArray<int>>(std::move(storage));
  ++counts->created;
  return payload;
}

void FillPayloads(const Array& array,
                  const std::shared_ptr<PayloadCounts>& counts) {
  auto* slots = static_cast<Container<int>*>(array.Data());
  for (std::size_t i = 0; i < array.size; ++i) {
    slots[i] = MakeCountedPayload(counts);
  }
}

// These lower-level cases explicitly opt into typed ownership. They do not
// imply a destructor contract for the legacy ShapeSpec-only queue overload.
void CheckMixedShapePartialBatch(bool receive) {
  auto counts = std::make_shared<PayloadCounts>();
  const auto typed_specs =
      std::make_tuple(Spec<Container<int>>({2, 3}, Spec<int>({1})),
                      Spec<Container<int>>({-1, 2, 2}, Spec<int>({1})),
                      Spec<Container<int>>({0, 2}, Spec<int>({1})),
                      Spec<Container<int>>({-1, 0, 3}, Spec<int>({1})));
  auto erased_specs = std::apply(
      [](const auto&... specs) { return std::vector<ShapeSpec>{specs...}; },
      typed_specs);
  std::vector<Array> output;
  {
    StateBufferQueue queue(3, 3, 3, erased_specs,
                           MakeStateArrayFactories(typed_specs));
    // One empty-player env, one two-player env, and an unused batch slot.
    // Shared fields still contain six payloads per allocated environment.
    for (int players : {0, 2}) {
      auto slice = queue.Allocate(players);
      for (const auto& array : slice.arr) {
        FillPayloads(array, counts);
      }
      slice.done_write();
    }
    if (receive) {
      output = queue.Wait(1);
    }
  }
  // Queue destruction joined every background producer before assertions.
  ASSERT_EQ(counts->created.load(), 20);
  if (receive) {
    ASSERT_EQ(counts->destroyed.load(), 0);
    ASSERT_EQ(output.size(), 4);
    EXPECT_EQ(output[0].Shape(), (std::vector<std::size_t>{2, 2, 3}));
    EXPECT_EQ(output[1].Shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_EQ(output[2].Shape(), (std::vector<std::size_t>{2, 0, 2}));
    EXPECT_EQ(output[3].Shape(), (std::vector<std::size_t>{2, 0, 3}));
    auto last_owner = output;
    output.clear();
    EXPECT_EQ(counts->destroyed.load(), 0);
    last_owner.clear();
  }
  EXPECT_EQ(counts->destroyed.load(), 20);
}

// StateArrayFactory is a function pointer, so the callback has one scoped
// context. The pointer is installed before thread creation and cleared only
// after queue destruction joins all producers; gtest runs these cases serially.
struct FactoryGate {
  explicit FactoryGate(bool allow_first) : allow_first(allow_first) {}

  Array Make(const ShapeSpec& spec) {
    if (std::this_thread::get_id() == foreground) {
      ++foreground_calls;
    } else {
      std::unique_lock<std::mutex> lock(mutex);
      const int visit = ++visits[std::this_thread::get_id()];
      ++background_entries;
      // Returning to the same producer's factory proves its prior Put ended.
      // No consumer has run when WaitUntilReady observes this condition.
      if (visit > 1) {
        ++completed_puts;
      }
      condition.notify_all();
      condition.wait(lock, [&] { return open || (allow_first && visit == 1); });
    }
    return StateArrayFactoryHelper<Container<int>>::Make(spec);
  }

  bool WaitUntilReady(bool require_stock) {
    std::unique_lock<std::mutex> lock(mutex);
    return condition.wait_for(lock, std::chrono::seconds(10), [&] {
      return require_stock ? completed_puts > 0 : background_entries > 0;
    });
  }

  void Open() {
    std::lock_guard<std::mutex> lock(mutex);
    open = true;
    condition.notify_all();
  }

  const std::thread::id foreground{std::this_thread::get_id()};
  const bool allow_first;
  std::atomic<int> foreground_calls{0};
  std::mutex mutex;
  std::condition_variable condition;
  std::map<std::thread::id, int> visits;
  int background_entries{0};
  int completed_puts{0};
  bool open{false};
};

FactoryGate* active_factory_gate = nullptr;

Array GatedContainerFactory(const ShapeSpec& spec) {
  return active_factory_gate->Make(spec);
}

void CheckReplacementRecipe(bool use_stock) {
  // Public constructor arguments (batch=1, envs=1) currently create six ring
  // entries. Seven allocations revisit the first replaced entry.
  constexpr int kQueueSlots = 6;
  auto counts = std::make_shared<PayloadCounts>();
  FactoryGate gate(use_stock);
  active_factory_gate = &gate;
  auto queue = std::make_unique<StateBufferQueue>(
      1, 1, 1, std::vector<ShapeSpec>{Spec<Container<int>>({}, Spec<int>({1}))},
      std::vector<StateArrayFactory>{GatedContainerFactory});
  const bool ready = gate.WaitUntilReady(use_stock);
  const bool initial_recipe = gate.foreground_calls.load() == kQueueSlots;
  bool replacement_recipe = false;
  int first_wait_calls = -1;
  if (ready && initial_recipe) {
    for (int iteration = 0; iteration <= kQueueSlots; ++iteration) {
      auto slice = queue->Allocate(1);
      FillPayloads(slice.arr[0], counts);
      slice.done_write();
      auto output = queue->Wait();
      output.clear();
      if (iteration == 0) {
        first_wait_calls = gate.foreground_calls.load();
        replacement_recipe =
            first_wait_calls == kQueueSlots + (use_stock ? 0 : 1);
        // Do not access replacement Container slots if the factory was lost.
        if (!replacement_recipe) {
          break;
        }
      }
    }
  }
  const int final_foreground_calls = gate.foreground_calls.load();
  // Releasing every producer before destroying the queue also makes timeout
  // and failed-recipe paths safe. No assertion can skip this release/join.
  gate.Open();
  queue.reset();
  active_factory_gate = nullptr;
  ASSERT_TRUE(ready);
  ASSERT_TRUE(initial_recipe);
  ASSERT_TRUE(replacement_recipe)
      << "first Wait factory calls=" << first_wait_calls;
  EXPECT_EQ(counts->created.load(), kQueueSlots + 1);
  EXPECT_EQ(counts->destroyed.load(), kQueueSlots + 1);
  if (!use_stock) {
    EXPECT_EQ(final_foreground_calls, 2 * kQueueSlots + 1);
  }
}

}  // namespace

TEST(ContainerOutputTest, TruncateReclaimsOriginalCapacity) {
  auto counts = std::make_shared<PayloadCounts>();
  std::vector<Array> owners;
  {
    TArray<Container<int>> full(StateArrayFactoryHelper<Container<int>>::Make(
        Spec<Container<int>>({3, 2}, Spec<int>({1}))));
    FillPayloads(full, counts);
    auto truncated = full.Truncate(1);
    EXPECT_EQ(truncated.size, 2);
    owners.emplace_back(static_cast<const Array&>(truncated));
  }
  EXPECT_EQ(counts->created.load(), 6);
  EXPECT_EQ(counts->destroyed.load(), 0);
  owners[0].TruncateInPlace(0);
  EXPECT_EQ(owners[0].size, 0);
  owners.clear();
  EXPECT_EQ(counts->destroyed.load(), 6);
}

TEST(ContainerOutputTest, LastStrongOwnerReclaimsWithWeakObserverRetained) {
  auto counts = std::make_shared<PayloadCounts>();
  std::weak_ptr<char> observer;
  {
    auto owner = StateArrayFactoryHelper<Container<int>>::Make(
        Spec<Container<int>>({1}, Spec<int>({1})));
    FillPayloads(owner, counts);
    observer = owner.SharedPtr();
    {
      auto another_owner = owner;
      EXPECT_EQ(another_owner.size, 1);
      EXPECT_FALSE(observer.expired());
    }
    EXPECT_EQ(counts->destroyed.load(), 0);
  }
  EXPECT_TRUE(observer.expired());
  EXPECT_EQ(counts->destroyed.load(), 1);
  observer.reset();
  EXPECT_EQ(counts->destroyed.load(), 1);
}

TEST(ContainerOutputTest, MixedShapePartialBatchRetainsReturnedPayloads) {
  CheckMixedShapePartialBatch(true);
}

TEST(ContainerOutputTest, MixedShapePartialBatchReclaimsUnconsumedPayloads) {
  CheckMixedShapePartialBatch(false);
}

TEST(ContainerOutputTest, StockReplacementPreservesTypedRecipe) {
  CheckReplacementRecipe(true);
}

TEST(ContainerOutputTest, FallbackReplacementPreservesTypedRecipe) {
  CheckReplacementRecipe(false);
}
