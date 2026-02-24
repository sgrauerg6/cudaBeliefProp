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

#include <vector>

template <typename T>
class BnchmrksMtrx {
public:
  explicit BnchmrksMtrx(size_t width, size_t height, T* mtrx_data) :
    width_(width), height_(height)
  {
    //copy input matrix data to vector
    mtrix_data_ = std::vector<T>(mtrx_data, mtrx_data + (width_*height_));
  }

  float GetSumSqrDiff(const BnchmrksMtrx& mtrx_comp) const {
    float sum_sqr_diff{0.0};
    //assuming that matrix to compare has same width and height
    for (size_t i=0; i < width_*height_; i++) {
      sum_sqr_diff +=
        ((float)matrix_data_[i] - (float)mtrx_comp.mtrix_data_[i]) *
        ((float)matrix_data_[i] - (float)mtrx_comp.mtrix_data_[i]);
    }
    
    //return computed sum of square different across all elements in matrices
    //that are compared
    return sum_sqr_diff;
  } 

private:
  size_t width_;
  size_t height_;
  std::vector<T> mtrix_data_;
};

#endif //BNCHMRKS_MATRX_H_