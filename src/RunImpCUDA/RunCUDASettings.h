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
 * @file RunCUDASettings.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_CUDA_SETTINGS_H_
#define RUN_CUDA_SETTINGS_H_

#include <vector>
#include <array>

//set data type used for half-precision with CUDA
#ifdef USE_BFLOAT16_FOR_HALF_PRECISION
#include <cuda_bf16.h>
using halftype = __nv_bfloat16;
#else
#include <cuda_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION

/**
 * @brief Namespace with constants for parallel parameters default and options
 * to use in run optimization
 */
namespace run_cuda {

/** @brief Parallel parameter options to run to retrieve optimized parallel parameters in CUDA implementation
 *  Parallel parameter corresponds to thread block dimensions in CUDA implementation */
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  {16, 1}, {32, 4}, {64, 4}};
#else
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  {16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4},
  {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS

/** @brief Default thread block dimensions (which is what parallel parameters
 *  corresponds to in CUDA implementation) */
constexpr std::array<unsigned int, 2> kParallelParamsDefault{{32, 4}};

};

#endif /* RUN_CUDA_SETTINGS_H_ */
