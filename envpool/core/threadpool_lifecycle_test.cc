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
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "ThreadPool.h"

namespace {

// Ordinary lifecycle checks only; no thread-launch failure injection.
class ThreadPoolLifecycleTest : public ::testing::TestWithParam<std::size_t> {
 protected:
  ThreadPoolLifecycleTest()
      : watchdog_([this] {
          std::unique_lock<std::mutex> lock(mutex_);
          if (!condition_.wait_for(lock, std::chrono::seconds(30),
                                   [this] { return finished_; })) {
            std::fputs("threadpool lifecycle exceeded its 30-second deadline\n",
                       stderr);
            std::abort();
          }
        }) {}

  ~ThreadPoolLifecycleTest() override {
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

TEST_P(ThreadPoolLifecycleTest, CompletesTaskResults) {
  ThreadPool pool(GetParam());
  std::vector<std::future<int>> results;
  constexpr int kTasks = 32;
  results.reserve(kTasks);
  for (int i = 0; i < kTasks; ++i) {
    results.emplace_back(pool.enqueue([i] { return i * i; }));
  }
  for (int i = 0; i < kTasks; ++i) {
    ASSERT_EQ(results[i].wait_for(std::chrono::seconds(10)),
              std::future_status::ready);
    EXPECT_EQ(results[i].get(), i * i);
  }
}

TEST_P(ThreadPoolLifecycleTest, DestructorCompletesSubmittedTasks) {
  std::atomic<int> completed{0};
  std::vector<std::future<void>> results;
  constexpr int kTasks = 32;
  results.reserve(kTasks);
  {
    ThreadPool pool(GetParam());
    for (int i = 0; i < kTasks; ++i) {
      results.emplace_back(pool.enqueue([&completed] { ++completed; }));
    }
    // Leave scope without waiting on task futures. Destruction joins workers
    // after queued work has drained, so every result must now be ready.
  }
  EXPECT_EQ(completed.load(), kTasks);
  for (auto& result : results) {
    ASSERT_EQ(result.wait_for(std::chrono::seconds(0)),
              std::future_status::ready);
    result.get();
  }
}

INSTANTIATE_TEST_SUITE_P(OneAndSeveralWorkers, ThreadPoolLifecycleTest,
                         ::testing::Values(1U, 3U));

}  // namespace
