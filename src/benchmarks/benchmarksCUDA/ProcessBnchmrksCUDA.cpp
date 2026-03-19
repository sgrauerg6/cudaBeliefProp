/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file ProcessBnchmrksCUDA.cpp
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#include "ProcessBnchmrksCUDA.h"
#include "KernelBnchmrksCUDA.cu"

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN, typename U>
std::optional<DetailedTimings<benchmarks::Runtime_Type>> ProcessBnchmrksCUDA<T, ACCELERATION, BENCHMARK_RUN, U>::TwoDMatricesBnchmrk(
  const unsigned int mat_w_h,
  const U* mat_input_0,
  const U* mat_input_1,
  U* mat_result) const
{
  if (run_cuda::ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return {};
  }

  //set to prefer L1 cache for now since no shared memory is used
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //setup execution parameters
  const auto kernel_thread_block_dims =
    this->parallel_params_.OptParamsForKernel({0, 0});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  //kernel run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
  const dim3 grid{
    (unsigned int)ceil((float)mat_w_h / (float)threads.x),
    (unsigned int)ceil((float)mat_w_h / (float)threads.y)};

  auto add_mat_start_time = std::chrono::system_clock::now();
  //process matrix addition on GPU using CUDA
  benchmarks_cuda::TwoDMatricesBnchmrk<T, BENCHMARK_RUN> <<<grid, threads>>> (
    mat_w_h, mat_w_h, mat_input_0, mat_input_1, mat_result);
  cudaDeviceSynchronize();
  auto end_mat_start_time = std::chrono::system_clock::now();

  if (run_cuda::ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
    return {};
  }

  DetailedTimings add_mat_timing(benchmarks::kTimingNames);
  add_mat_timing.AddTiming(benchmarks::Runtime_Type::kTotalBnchmrkNoTransfer,
    end_mat_start_time - add_mat_start_time);

  return add_mat_timing;
}

template class ProcessBnchmrksCUDA<float, run_environment::AccSetting::kCUDA, benchmarks::BenchmarkRun::kGemm, float>;
template class ProcessBnchmrksCUDA<double, run_environment::AccSetting::kCUDA, benchmarks::BenchmarkRun::kGemm, double>;
template class ProcessBnchmrksCUDA<halftype, run_environment::AccSetting::kCUDA, benchmarks::BenchmarkRun::kGemm, halftype>;