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
 * @file KernelBnchmrksCUDA.cu
 * @author Scott Grauer-Gray
 * @brief Kernel functions to run benchmarks on the GPU using CUDA
 * 
 * @copyright Copyright (c) 2026
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Namespace to define kernel functions for benchmark functions
 * using Metal
 */
//namespace benchmarks_metal {

  /**
   * @brief Metal kernel to run GEMM benchmark
   * 
   * @param matrix_0
   * @param matrix_1
   * @param mtrx_width
   * @param mtrx_height
   * @param matrix_result
   */
  //template <RunData_t T, benchmarks::BenchmarkRun BENCHMARK_RUN>
  kernel void TwoDMatricesBnchmrkFloat(
    constant unsigned int& mtrx_width [[buffer(0)]],
    constant unsigned int& mtrx_height [[buffer(1)]],
    device const float* matrix_0 [[buffer(2)]],
    device const float* matrix_1 [[buffer(3)]],
    device float* matrix_result [[buffer(4)]],
    uint2 grid_pos [[thread_position_in_grid]])
  {
    //get x and y indices
    const unsigned int x_val = grid_pos.x;
    const unsigned int y_val = grid_pos.y;
    if ((x_val < mtrx_width) && (y_val < mtrx_height)) {
        const unsigned int val_idx = y_val*mtrx_width + x_val;
        float sum = 0.0f;
        size_t curr_matrix_input0_idx{y_val * mtrx_width};
        size_t curr_matrix_input1_idx{x_val};
        //compute dot product of the corresponding matrix_0 row and matrix_1 column
        for (size_t i = 0; i < mtrx_width; ++i) {
/*#if defined(USE_FUSED_MULTIPLY_ADD)
          sum =
            fma(
              matrix_0[curr_matrix_input0_idx],
              matrix_1[curr_matrix_input1_idx],
              sum);
#else*/
          sum +=
            matrix_0[curr_matrix_input0_idx] *
            matrix_1[curr_matrix_input1_idx];
//#endif //USE_FUSED_MULTIPLY_ADD
          curr_matrix_input0_idx += 1;
          curr_matrix_input1_idx += mtrx_width;
        }
        matrix_result[val_idx] = sum;
    }
  }
//};
