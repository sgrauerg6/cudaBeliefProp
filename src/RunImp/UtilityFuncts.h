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
 * @file UtilityFuncts.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef UTILITY_FUNCTS_H_
#define UTILITY_FUNCTS_H_

#include "RunEval/RunTypeConstraints.h"

#ifdef OPTIMIZED_CUDA_RUN
//added in front of function header to indicate that function is device function to be processed on GPU
#define ARCHITECTURE_ADDITION __device__
#else
#define ARCHITECTURE_ADDITION
#endif //OPTIMIZED_CUDA_RUN

/**
 * @brief Namespace with utility functions for bp implementation. 
 */
namespace util_functs {

/**
 * @brief T is input type, U is output type
 * 
 * @tparam T 
 * @tparam U 
 * @param data 
 * @return ARCHITECTURE_ADDITION 
 */
template<RunData_t T, RunData_t U>
ARCHITECTURE_ADDITION inline U ConvertValToDifferentDataTypeIfNeeded(T data) {
  return data; //by default assume same data type and just return data
}

template<RunData_t T>
ARCHITECTURE_ADDITION inline T ZeroVal() {
  return (T)0.0;
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T GetMin(T val1, T val2) {
  return ((val1 < val2) ? val1 : val2);
}

template<typename T>
requires std::is_arithmetic_v<T>
ARCHITECTURE_ADDITION inline T GetMax(T val1, T val2) {
  return ((val1 > val2) ? val1 : val2);
}

/**
 * @brief Checks if the current point is within the image bounds
 * Assumed that input x/y vals are above zero since their unsigned int so no need for >= 0 check
 * 
 * @param x_val 
 * @param y_val 
 * @param width 
 * @param height 
 * @return ARCHITECTURE_ADDITION 
 */
ARCHITECTURE_ADDITION inline bool WithinImageBounds(
  unsigned int x_val, unsigned int y_val,
  unsigned int width, unsigned int height)
{
  return ((x_val < width) && (y_val < height));
}

/**
 * @brief Inline function to check if data is aligned at x_val_data_start for
 * SIMD loads/stores that require alignment
 * 
 * @param x_val_data_start 
 * @param simd_data_size 
 * @param num_data_align_width 
 * @param divPaddedChBoardWidthForAlign 
 * @return true 
 * @return false 
 */
inline bool MemoryAlignedAtDataStart(
  unsigned int x_val_data_start,
  unsigned int simd_data_size,
  unsigned int num_data_align_width,
  unsigned int divPaddedChBoardWidthForAlign)
{
  //assuming that the padded checkerboard width divides evenly by beliefprop::NUM_DATA_ALIGN_WIDTH (if that's not the case it's a bug)
  return (((x_val_data_start % simd_data_size) == 0) && ((num_data_align_width % divPaddedChBoardWidthForAlign) == 0));
}

};

#endif //UTILITY_FUNCTS_H_