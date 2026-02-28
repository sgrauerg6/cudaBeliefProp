/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file BnchmrksMtrx.h
 * @author Scott Grauer-Gray
 * @brief Declares class for matrix to use and evaluate benchmarks
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef BNCHMRKS_MTRX_H_
#define BNCHMRKS_MTRX_H_

#include <chrono>
#include <algorithm>
#include <vector>
#include <random>
#include <iterator>

namespace benchmarks {
  const std::string_view kSumSqrDiffOptSingThreadOutputMtrx{
    "Sum squared difference between optimized and single-thread CPU results"};
};

template <typename T>
class BnchmrksMtrx {
public:
  explicit BnchmrksMtrx() {}

  explicit BnchmrksMtrx(size_t width, size_t height, T* data) :
    width_(width), height_(height)
  {
    mtrix_data_ =
      std::vector<T>(
        data,
        data + (width_ * height_));
  }

  void InitMtxWRandData(size_t width, size_t height) {
    width_ = width;
    height_ = height;
    mtrix_data_ = std::vector<T>(width_*height_);
    SetRandData();
  }

  void SetRandData() {
    //steady_clock provides a non-deterministic seed
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 mersenne_engine(seed); //Mersenne Twister engine

    //Define the distribution to be values from -999 to 999
    std::uniform_real_distribution dist(-999.0, 999.0);

    //Use std::generate to fill the vector
    //A lambda function is used to bind the distribution and engine
    auto generator = [&]() { return dist(mersenne_engine); };
    std::generate(mtrix_data_.begin(), mtrix_data_.end(), generator);
  }

  float GetSumSqrDiff(const BnchmrksMtrx<T>& mtrx_comp) const {
    //set max sum of squared difference to one-fifth of max float
    //value to prevent overflow
    constexpr float MAX_DIFF{
      std::numeric_limits<float>::max() / 5};
    float sum_sqr_diff{0.0};
    for (size_t i=0; i < width_*height_; i++) {
      sum_sqr_diff +=
        (((float)mtrix_data_[i] - (float)mtrx_comp.mtrix_data_[i]) *
         ((float)mtrix_data_[i] - (float)mtrx_comp.mtrix_data_[i]));
      if (sum_sqr_diff >= MAX_DIFF) {
        return MAX_DIFF;
      }
    }
    
    //return computed sum of square different across all elements in matrices
    //that are compared
    return sum_sqr_diff;
  }

  size_t Width() const {
    return width_;
  }

  size_t Height() const {
    return height_;
  }

  const T* Data() const {
    return mtrix_data_.data();
  }

private:
  size_t width_{0};
  size_t height_{0};
  std::vector<T> mtrix_data_;
};

#endif //BNCHMRKS_MATRX_H_