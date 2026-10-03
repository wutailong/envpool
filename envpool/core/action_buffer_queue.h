/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_CORE_ACTION_BUFFER_QUEUE_H_
#define ENVPOOL_CORE_ACTION_BUFFER_QUEUE_H_

#ifndef MOODYCAMEL_DELETE_FUNCTION
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include <atomic>
#include <cassert>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "lightweightsemaphore.h"

/**
 * Lock-free action buffer queue.
 */
class ActionBufferQueue {
 public:
  struct ActionSlice {
    int env_id;
    int order;
    bool force_reset;
  };

 protected:
  std::atomic<uint64_t> alloc_ptr_, done_ptr_;
  std::size_t queue_size_;
  std::vector<ActionSlice> queue_;
  moodycamel::LightweightSemaphore sem_, sem_enqueue_, sem_dequeue_;

 public:
  explicit ActionBufferQueue(std::size_t num_envs)
      : alloc_ptr_(0),
        done_ptr_(0),
        queue_size_(num_envs * 2),
        queue_(queue_size_),
        sem_(0),
        sem_enqueue_(1),
        sem_dequeue_(1) {}

  void EnqueueBulk(const std::vector<ActionSlice>& action) {
    // ensure only one enqueue_bulk happens at any time
    while (!sem_enqueue_.wait()) {
    }
    uint64_t pos = alloc_ptr_.fetch_add(action.size());
    for (std::size_t i = 0; i < action.size(); ++i) {
      queue_[(pos + i) % queue_size_] = action[i];
    }
    sem_.signal(action.size());
    sem_enqueue_.signal(1);
  }

  ActionSlice Dequeue() {
    while (!sem_.wait()) {
    }
    return DequeueReady();
  }

  // Shutdown wake permits do not represent queued actions. Check the caller's
  // stop flag after acquiring a permit, before reserving or reading a slot.
  bool DequeueOrStop(ActionSlice* action, const std::atomic<int>& stop) {
    while (!sem_.wait()) {
    }
    if (stop.load(std::memory_order_acquire) != 0) {
      return false;
    }
    *action = DequeueReady();
    return true;
  }

  // Terminal shutdown only: publish stop first, quiesce producers, and use
  // DequeueOrStop for every consumer. Wake at least all live consumers. No
  // payload writes can race pending consumers.
  void WakeForShutdown(std::size_t num_consumers) {
    sem_.signal(num_consumers);
  }

  std::size_t SizeApprox() {
    return static_cast<std::size_t>(alloc_ptr_ - done_ptr_);
  }

 private:
  ActionSlice DequeueReady() {
    while (!sem_dequeue_.wait()) {
    }
    auto ptr = done_ptr_.fetch_add(1);
    auto ret = queue_[ptr % queue_size_];
    sem_dequeue_.signal(1);
    return ret;
  }
};

#endif  // ENVPOOL_CORE_ACTION_BUFFER_QUEUE_H_
