/*
 * Copyright 2026 Garena Online Private Limited
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

// Measurement-only opaque consumer; ownership escapes the converting TU.
#include "conversion_timing_consumer.h"

#include "envpool/core/array.h"
extern "C" __attribute__((noinline)) std::uint64_t ConversionTimingConsume(
    const Array& array) {
  const auto* data = static_cast<const int*>(array.Data());
  return static_cast<std::uint64_t>(array.size) * 1000003ULL +
         static_cast<std::uint64_t>(array.ndim) * 10007ULL +
         static_cast<std::uint32_t>(data[0]) * 101ULL +
         static_cast<std::uint32_t>(data[array.size - 1]) * 1009ULL;
}
