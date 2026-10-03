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
#include <future>
#include <initializer_list>
#include <memory>
#include <thread>
#include <utility>

#include "envpool/core/array.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/core/env_spec.h"
#include "envpool/core/spec.h"

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
      // Leave the moved-from null slot alive for any outer-owner destructor.
      // Python's explicit slot destruction is a separate conversion contract.
    }
  }
  ASSERT_EQ(counts->destroyed.load(), 0);
  ASSERT_NE(extracted, nullptr);
  EXPECT_EQ(static_cast<int>((*extracted)[0]), 42);
  extracted.reset();
  EXPECT_EQ(counts->destroyed.load(), 1);
}
