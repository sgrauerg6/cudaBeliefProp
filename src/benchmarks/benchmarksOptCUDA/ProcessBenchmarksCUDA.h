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
 * @file ProcessBenchmarksCUDA.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BENCHMARKS_CUDA_H_
#define PROCESS_BENCHMARKS_CUDA_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "benchmarksRunProcessing/ProcessBenchmarks.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "KernelBenchmarksCUDA.cu"

template<RunData_t T, run_environment::AccSetting ACCELERATION>
class ProcessBenchmarksCUDA : public ProcessBenchmarks<T, ACCELERATION> {
public:

private:
  /**
   * @brief Function to run add matrices benchmark on device<br>
   * 
   * @param mat_w_h
   * @param mat_addend_0
   * @param mat_addend_1
   * @param mat_sum
   * @return Status of "no error" if successful, "error" status otherwise
   */
  run_eval::Status AddMatrices(
    const unsigned int mat_w_h,
    const T* mat_addend_0,
    const T* mat_addend_1,
    T* mat_sum) const override
  {
    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
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

    //initialize the data the the "bottom" of the image pyramid
    benchmarks_cuda::addMatrices<T> <<<grid, threads>>> (
      mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    cudaDeviceSynchronize();

    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }

    return run_eval::Status::kNoError;
  }
};

#endif //PROCESS_BENCHMARKS_CUDA_H_