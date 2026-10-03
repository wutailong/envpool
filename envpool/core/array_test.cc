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

#include "envpool/core/array.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "envpool/core/spec.h"

TEST(ArrayTest, ViewsPreserveHighDimensionalShapesAndData) {
  ShapeSpec spec(sizeof(int), {2, 3, 4, 5, 6});
  Array array(spec);

  auto view = array(1, 2);
  EXPECT_EQ(view.Shape(), std::vector<std::size_t>({4, 5, 6}));
  view(3, 4, 5) = 17;
  EXPECT_EQ(*reinterpret_cast<int*>(array(1, 2, 3, 4, 5).Data()), 17);

  auto slice = array.Slice(1, 2);
  EXPECT_EQ(slice.Shape(), std::vector<std::size_t>({1, 3, 4, 5, 6}));
  slice(0, 2, 3, 4, 5) = 23;
  EXPECT_EQ(*reinterpret_cast<int*>(array(1, 2, 3, 4, 5).Data()), 23);
}

TEST(ArrayTest, TruncateAndSharedPtrKeepOwnedStorageAlive) {
  ShapeSpec spec(sizeof(int), {4});
  int delete_count = 0;
  std::shared_ptr<char> shared;

  {
    Array truncated;
    {
      auto* data = new char[4 * sizeof(int)];
      Array parent(spec, data, [&delete_count](char* ptr) {
        ++delete_count;
        delete[] ptr;
      });
      truncated = parent.Truncate(2);
      shared = truncated.SharedPtr();
      EXPECT_EQ(shared.get(), truncated.Data());
    }
    EXPECT_EQ(delete_count, 0);
  }

  EXPECT_EQ(delete_count, 0);
  shared.reset();
  EXPECT_EQ(delete_count, 1);
}

TEST(ArrayTest, SharedPtrPreservesNonOwningDataPointer) {
  ShapeSpec spec(sizeof(int), {2});
  std::array<int, 2> data{};
  Array array(spec, reinterpret_cast<char*>(data.data()));

  auto shared = array.SharedPtr();
  EXPECT_EQ(shared.get(), reinterpret_cast<char*>(data.data()));
}

TEST(ArrayTest, ViewsDoNotExtendBackingStorageLifetime) {
  ShapeSpec spec(sizeof(int), {2, 2});
  int delete_count = 0;

  {
    Array indexed_view;
    Array slice;
    {
      auto* data = new char[4 * sizeof(int)];
      Array parent(spec, data, [&delete_count](char* ptr) {
        ++delete_count;
        delete[] ptr;
      });
      indexed_view = parent(1);
      slice = parent.Slice(0, 1);
    }
    EXPECT_EQ(delete_count, 1);
  }

  EXPECT_EQ(delete_count, 1);
}

TEST(ArrayTest, TypedTruncateSharesData) {
  Spec<int> spec(std::vector<int>({4}));
  TArray<int> array(spec);

  const TArray<int>& original = array;
  auto truncated = original.Truncate(2);
  truncated[1] = 31;

  EXPECT_EQ(array.Shape(), std::vector<std::size_t>({4}));
  EXPECT_EQ(array.size, 4U);
  EXPECT_EQ(truncated.Shape(), std::vector<std::size_t>({2}));
  EXPECT_EQ(truncated.size, 2U);
  EXPECT_EQ(static_cast<int>(array[1]), 31);
}

TEST(ArrayTest, TruncateInPlacePreservesHighDimensionalShapeAndData) {
  ShapeSpec spec(sizeof(int), {4, 3, 2, 5, 2, 3});
  Array array(spec);
  auto* data = reinterpret_cast<int*>(array.Data());
  for (std::size_t i = 0; i < array.size; ++i) {
    data[i] = static_cast<int>(i);
  }

  array.TruncateInPlace(3);
  EXPECT_EQ(array.Shape(), std::vector<std::size_t>({3, 3, 2, 5, 2, 3}));
  EXPECT_EQ(array.ndim, 6U);
  EXPECT_EQ(array.element_size, sizeof(int));
  EXPECT_EQ(array.size, 540U);
  EXPECT_EQ(array.Data(), data);
  for (std::size_t i = 0; i < array.size; ++i) {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
  EXPECT_EQ(*reinterpret_cast<int*>(array(2, 2, 1, 4, 1, 2).Data()), 539);

  array.TruncateInPlace(1);
  EXPECT_EQ(array.Shape(), std::vector<std::size_t>({1, 3, 2, 5, 2, 3}));
  EXPECT_EQ(array.size, 180U);
  EXPECT_EQ(array.Data(), data);
  EXPECT_EQ(*reinterpret_cast<int*>(array(0, 2, 1, 4, 1, 2).Data()), 179);

  array.TruncateInPlace(0);
  EXPECT_EQ(array.Shape(), std::vector<std::size_t>({0, 3, 2, 5, 2, 3}));
  EXPECT_EQ(array.size, 0U);
  EXPECT_EQ(array.Data(), data);
}

TEST(ArrayTest, TruncateInPlaceHandlesEmptyDimensions) {
  Array empty_first(ShapeSpec(sizeof(int), {0, 3, 4}));
  void* empty_data = empty_first.Data();
  empty_first.TruncateInPlace(0);
  empty_first.TruncateInPlace(0);
  EXPECT_EQ(empty_first.Shape(), std::vector<std::size_t>({0, 3, 4}));
  EXPECT_EQ(empty_first.ndim, 3U);
  EXPECT_EQ(empty_first.size, 0U);
  EXPECT_EQ(empty_first.Data(), empty_data);

  Array empty_inner(ShapeSpec(sizeof(int), {4, 0, 3}));
  empty_inner.TruncateInPlace(2);
  EXPECT_EQ(empty_inner.Shape(), std::vector<std::size_t>({2, 0, 3}));
  EXPECT_EQ(empty_inner.size, 0U);
}

TEST(ArrayTest, TruncateInPlaceRetainsSharedOwnershipAndCustomDeleter) {
  ShapeSpec spec(sizeof(int), {4, 2});
  int delete_count = 0;
  std::shared_ptr<char> shared;

  {
    Array retained;
    {
      auto* data = new char[8 * sizeof(int)];
      Array array(spec, data, [&delete_count](char* ptr) {
        ++delete_count;
        delete[] ptr;
      });
      {
        const Array& original = array;
        Array sibling = original.Truncate(4);
        array.TruncateInPlace(2);
        EXPECT_EQ(array.Data(), data);
        EXPECT_EQ(array.Shape(), std::vector<std::size_t>({2, 2}));
        EXPECT_EQ(sibling.Shape(), std::vector<std::size_t>({4, 2}));
        EXPECT_EQ(sibling.size, 8U);
        sibling(1, 1) = 37;
      }
      ASSERT_EQ(delete_count, 0);
      retained = std::move(array);
    }
    ASSERT_EQ(delete_count, 0);
    EXPECT_EQ(retained.size, 4U);
    EXPECT_EQ(*reinterpret_cast<int*>(retained(1, 1).Data()), 37);
    shared = retained.SharedPtr();
    EXPECT_EQ(shared.get(), retained.Data());
  }

  EXPECT_EQ(delete_count, 0);
  shared.reset();
  EXPECT_EQ(delete_count, 1);
}
