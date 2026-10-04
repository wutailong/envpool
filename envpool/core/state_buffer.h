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

#ifndef ENVPOOL_CORE_STATE_BUFFER_H_
#define ENVPOOL_CORE_STATE_BUFFER_H_

#ifndef MOODYCAMEL_DELETE_FUNCTION
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/dict.h"
#include "envpool/core/spec.h"
#include "lightweightsemaphore.h"

// Preserve element lifetimes before state specs are erased to ShapeSpec.
// The legacy ShapeSpec-only path remains raw storage with manual lifetimes.
using StateArrayFactory = Array (*)(const ShapeSpec&);

template <typename Dtype>
struct StateArrayFactoryHelper {
  static constexpr bool kIsContainer = false;
  static Array Make(const ShapeSpec& spec) { return Array(spec); }
};

template <typename Dtype>
struct StateArrayFactoryHelper<Container<Dtype>> {
  static constexpr bool kIsContainer = true;
  static Array Make(const ShapeSpec& spec) {
    // Expose the existing protected owner/shape constructor only here. The
    // fieldless bridge moves its Array base into the return value.
    struct OwnedArray final : Array {
      OwnedArray(std::shared_ptr<char> owner, std::vector<std::size_t>&& shape,
                 std::size_t element_size)
          : Array(std::move(owner), std::move(shape), element_size) {}
    };
    auto shape = spec.Shape();
    // Every slot is live and null, including unused original capacity. The
    // alias keeps typed delete[] at final strong ownership, even with weak
    // observers, without another control block or a second Shape() copy.
    std::shared_ptr<Container<Dtype>[]> owner(
        new Container<Dtype>[Prod(shape.data(), shape.size())]);
    auto* data = reinterpret_cast<char*>(owner.get());
    return OwnedArray(std::shared_ptr<char>(owner, data), std::move(shape),
                      spec.element_size);
  }
};

template <typename... Spec>
std::vector<StateArrayFactory> MakeStateArrayFactories(
    const std::tuple<Spec...>& /*specs*/) {
  if constexpr ((StateArrayFactoryHelper<typename Spec::dtype>::kIsContainer ||
                 ...)) {
    return {&StateArrayFactoryHelper<typename Spec::dtype>::Make...};
  }
  // Primitive-only pools keep the existing MakeArray allocation path.
  return {};
}

inline std::vector<Array> MakeStateArrays(
    const std::vector<ShapeSpec>& specs,
    const std::vector<StateArrayFactory>& factories) {
  if (factories.empty()) {
    return MakeArray(specs);
  }
  CHECK_EQ(factories.size(), specs.size());
  std::vector<Array> arrays;
  arrays.reserve(specs.size());
  for (std::size_t i = 0; i < specs.size(); ++i) {
    arrays.emplace_back(factories[i](specs[i]));
  }
  return arrays;
}

/**
 * Buffer of a batch of states, which is used as an intermediate storage device
 * for the environments to write their state outputs of each step.
 * There's a quota for how many envs' results are stored in this buffer,
 * which is controlled by the batch argments in the constructor.
 */
class StateBuffer {
 protected:
  std::size_t batch_;
  std::size_t max_num_players_;
  std::vector<Array> arrays_;
  std::vector<bool> is_player_state_;
  std::atomic<uint64_t> offsets_{0};
  std::atomic<std::size_t> alloc_count_{0};
  std::atomic<std::size_t> done_count_{0};
  moodycamel::LightweightSemaphore sem_;

 public:
  /**
   * Return type of StateBuffer.Allocate is a slice of each state arrays that
   * can be written by the caller. When writing is done, the caller should
   * invoke done write.
   */
  struct WritableSlice {
    std::vector<Array> arr;
    std::function<void()> done_write;
  };

  /**
   * Create a StateBuffer instance with the player_specs and shared_specs
   * provided.
   */
  StateBuffer(std::size_t batch, std::size_t max_num_players,
              const std::vector<ShapeSpec>& specs,
              std::vector<bool> is_player_state)
      : batch_(batch),
        max_num_players_(max_num_players),
        arrays_(MakeArray(specs)),
        is_player_state_(std::move(is_player_state)) {}

  // Opt into typed ownership with factories matching the original spec tuple.
  StateBuffer(std::size_t batch, std::size_t max_num_players,
              const std::vector<ShapeSpec>& specs,
              std::vector<bool> is_player_state,
              const std::vector<StateArrayFactory>& factories)
      : batch_(batch),
        max_num_players_(max_num_players),
        arrays_(MakeStateArrays(specs, factories)),
        is_player_state_(std::move(is_player_state)) {}

  /**
   * Tries to allocate a piece of memory without lock.
   * If this buffer runs out of quota, an out_of_range exception is thrown.
   * Externally, caller has to catch the exception and handle accordingly.
   */
  WritableSlice Allocate(std::size_t num_players, int order = -1) {
    DCHECK_LE(num_players, max_num_players_);
    std::size_t alloc_count = alloc_count_.fetch_add(1);
    if (alloc_count < batch_) {
      // Make a increment atomically on two uint32_t simultaneously
      // This avoids lock
      uint64_t increment = static_cast<uint64_t>(num_players) << 32 | 1;
      uint64_t offsets = offsets_.fetch_add(increment);
      uint32_t player_offset = offsets >> 32;
      uint32_t shared_offset = offsets;
      DCHECK_LE((std::size_t)shared_offset + 1, batch_);
      DCHECK_LE((std::size_t)(player_offset + num_players),
                batch_ * max_num_players_);
      if (order != -1 && max_num_players_ == 1) {
        // single player with sync setting: return ordered data
        player_offset = shared_offset = order;
      }
      std::vector<Array> state;
      state.reserve(arrays_.size());
      for (std::size_t i = 0; i < arrays_.size(); ++i) {
        const Array& a = arrays_[i];
        if (is_player_state_[i]) {
          state.emplace_back(
              a.Slice(player_offset, player_offset + num_players));
        } else {
          state.emplace_back(a[shared_offset]);
        }
      }
      return WritableSlice{.arr = std::move(state),
                           .done_write = [this]() { Done(); }};
    }
    DLOG(INFO) << "Allocation failed, continue to the next block of memory";
    throw std::out_of_range("StateBuffer out of storage");
  }

  [[nodiscard]] std::pair<uint32_t, uint32_t> Offsets() const {
    uint32_t player_offset = offsets_ >> 32;
    uint32_t shared_offset = offsets_;
    return {player_offset, shared_offset};
  }

  /**
   * When the allocated memory has been filled, the user of the memory will
   * call this callback to notify StateBuffer that its part has been written.
   */
  void Done(std::size_t num = 1) {
    // Another writer may publish the final count and retire this buffer as
    // soon as we contribute. Read shared metadata before that publication.
    const std::size_t batch = batch_;
    std::size_t done_count = done_count_.fetch_add(num);
    if (done_count + num == batch) {
      sem_.signal();
    }
  }

  /**
   * Blocks until the entire buffer is ready, aka, all quota has been
   * distributed out, and all user has called done.
   */
  std::vector<Array> Wait(std::size_t additional_done_count = 0) {
    if (additional_done_count > 0) {
      Done(additional_done_count);
    }
    while (!sem_.wait()) {
    }
    // when things are all done, compact the buffer.
    uint64_t offsets = offsets_;
    uint32_t player_offset = (offsets >> 32);
    uint32_t shared_offset = offsets;
    DCHECK_EQ((std::size_t)shared_offset, batch_ - additional_done_count);
    // Wait consumes this one-shot buffer. Transfer the owning arrays rather
    // than allocating a second vector and copying each array's shape.
    for (std::size_t i = 0; i < arrays_.size(); ++i) {
      arrays_[i].TruncateInPlace(is_player_state_[i] ? player_offset
                                                     : shared_offset);
    }
    return std::move(arrays_);
  }
};

#endif  // ENVPOOL_CORE_STATE_BUFFER_H_
