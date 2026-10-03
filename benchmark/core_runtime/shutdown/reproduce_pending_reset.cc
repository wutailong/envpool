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

#include "envpool/dummy/dummy_envpool.h"

TEST(ShutdownRegression, PendingActionWithMoreWorkersThanEnvironments) {
  auto config = dummy::DummyEnvSpec::kDefaultConfig;
  config["num_envs"_] = 1;
  config["batch_size"_] = 1;
  config["num_threads"_] = 2;
  config["max_num_players"_] = 1;
  config["seed"_] = 42;
  dummy::DummyEnvSpec spec(config);
  Array ids(Spec<int>({1}));
  ids[0] = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    dummy::DummyEnvPool pool(spec);
    pool.Reset(ids);
  }
}
