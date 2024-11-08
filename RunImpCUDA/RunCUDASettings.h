/*
 * RunCUDASettings.h
 *
 *  Created on: Jan 27, 2024
 *      Author: scott
 */

#ifndef RUNCUDASETTINGS_H_
#define RUNCUDASETTINGS_H_

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

namespace run_cuda {

//parallel parameter options to run to retrieve optimized parallel parameters in CUDA implementation
//parallel parameter corresponds to thread block dimensions in CUDA implementation
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{{16, 1}, {32, 4}, {64, 4}};
#else
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{{16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4}, {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS
constexpr std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{32, 4}};

};

#endif /* RUNCUDASETTINGS_H_ */
