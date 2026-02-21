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
 * @file KernelBenchmarksCUDA.cu
 * @author Scott Grauer-Gray
 * @brief Kernel functions to run benchmarks on the GPU using CUDA
 * 
 * @copyright Copyright (c) 2026
 */

#include "KernelFilterCUDA.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include "RunImp/UtilityFuncts.h"

/**
 * @brief Namespace to define global kernel functions for benchmark functions
 * using CUDA
 */
namespace benchmarks_cuda {

  /**
   * @brief CUDA kernel to sum two matrices element-by-element
   * 
   * @param matrix_0
   * @param matrix_1
   * @param mtrx_width
   * @param mtrx_height
   * @param matrix_sum
   */
  template <RunData_t T>
  __global__ void addMatrices(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const T* matrix_0, const T* matrix_1,
    T* matrix_sum)
  {
    //get x and y indices corresponding to current CUDA thread
    const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

    matrix_sum[y_val * mtrx_width + x_val] =
      matrix_0[y_val * mtrx_width + x_val] + 
      matrix_1[y_val * mtrx_width + x_val];
  }
};
