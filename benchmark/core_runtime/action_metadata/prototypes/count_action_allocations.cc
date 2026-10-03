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

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include "envpool/classic_control/cartpole.h"
thread_local bool count_enabled = false;
thread_local std::size_t calls = 0, bytes = 0;
void* operator new(std::size_t n) {
  if (count_enabled) {
    ++calls;
    bytes += n;
  }
  if (auto p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
int main() {
  auto config = classic_control::CartPoleEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["num_threads"_] = 1;
  classic_control::CartPoleEnvSpec spec(config);
  StateBufferQueue queue(1, 1, 1, spec.state_spec.AllValues<ShapeSpec>());
  classic_control::CartPoleEnv env(spec, 0);
  auto actions = std::make_shared<std::vector<Array>>();
  for (int i = 0; i < 3; ++i) actions->emplace_back(Spec<int>({1}));
  env.SetAction(actions, 0);
  std::size_t env_calls = 0, env_bytes = 0, wait_calls = 0, wait_bytes = 0;
  std::size_t reset_steps = 0, step_steps = 0, reset_calls = 0, reset_bytes = 0,
              step_calls = 0, step_bytes = 0;
  for (int i = 0; i < 10010; ++i) {
    const bool measure = i >= 10;
    const bool reset = i == 0 || env.IsDone();
    calls = bytes = 0;
    count_enabled = measure;
    env.EnvStep(&queue, 0, reset, i == 0);
    count_enabled = false;
    env_calls += calls;
    env_bytes += bytes;
    if (measure && reset) {
      ++reset_steps;
      reset_calls += calls;
      reset_bytes += bytes;
    }
    if (measure && !reset) {
      ++step_steps;
      step_calls += calls;
      step_bytes += bytes;
    }
    calls = bytes = 0;
    count_enabled = measure;
    auto results = queue.Wait();
    count_enabled = false;
    wait_calls += calls;
    wait_bytes += bytes;
  }
  std::printf(
      "{\"reset_steps\":%zu,\"reset_alloc_calls\":%zu,\"reset_alloc_bytes\":%"
      "zu,\"nonreset_steps\":%zu,\"nonreset_alloc_calls\":%zu,\"nonreset_alloc_"
      "bytes\":%zu}\n",
      reset_steps, reset_calls, reset_bytes, step_steps, step_calls,
      step_bytes);
  std::size_t compact_calls = 0, compact_bytes = 0;
  auto raw_specs = spec.state_spec.AllValues<ShapeSpec>();
  std::vector<bool> player;
  std::vector<ShapeSpec> batched;
  for (auto shape : raw_specs) {
    bool is_player = !shape.shape.empty() && shape.shape[0] == -1;
    player.push_back(is_player);
    if (is_player) {
      shape.shape[0] = 4;
      batched.push_back(shape);
    } else {
      batched.push_back(shape.Batch(4));
    }
  }
  for (int i = 0; i < 10000; ++i) {
    StateBuffer buffer(4, 1, batched, player);
    for (int j = 0; j < 4; ++j) {
      auto slice = buffer.Allocate(1);
      slice.done_write();
    }
    calls = bytes = 0;
    count_enabled = true;
    auto result = buffer.Wait();
    count_enabled = false;
    compact_calls += calls;
    compact_bytes += bytes;
  }
  std::printf(
      "{\"compact_steps\":10000,\"wait_metadata_alloc_calls\":%zu,\"wait_"
      "metadata_alloc_bytes\":%zu}\n",
      compact_calls, compact_bytes);
  std::printf(
      "{\"steps\":10000,\"env_step_alloc_calls\":%zu,\"env_step_alloc_bytes\":%"
      "zu,\"recv_alloc_calls\":%zu,\"recv_alloc_bytes\":%zu}\n",
      env_calls, env_bytes, wait_calls, wait_bytes);
}
