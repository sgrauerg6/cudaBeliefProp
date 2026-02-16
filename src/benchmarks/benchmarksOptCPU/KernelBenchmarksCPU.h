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
 * @file KernelBenchmarksCPU.h
 * @author Scott Grauer-Gray
 * @brief This header declares the kernel functions to run optimized benchmarks
 * on CPU
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef KERNEL_BENCHMARKS_CPU_H
#define KERNEL_BENCHMARKS_CPU_H

#ifndef __APPLE__
#include <omp.h>
#else
#include <dispatch/dispatch.h>
#endif //__APPLE__

#include <math.h>
#include <iostream>
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/UtilityFuncts.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunImpCPU/SIMDProcessing.h"

//#define DONT_USE_GRAND_CENTRAL_DISPATCH

/**
 * @brief Namespace to define global kernel functions for optimized benchmark
 * processing on the CPU using OpenMP and SIMD vectorization.
 */
namespace benchmarks_cpu
{
  /**
   * @brief Sum two matrices element-by-element using parallelism but not
   * SIMD instructions
   * 
   * @param matrix_0
   * @param matrix_1
   * @param mtrx_width
   * @param mtrx_height
   * @param matrix_sum
   */
  template <RunData_t T>
  void AddMatricesNoPackedInstructions(
    const T* matrix_0, const T* matrix_1,
    unsigned int mtrx_width, unsigned int mtrx_height,
    T* matrix_sum)
  {
    for (unsigned int y=0; y < mtrx_height; y++)
    {
#if !defined(__APPLE__) || defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
      #pragma omp parallel for
      for (unsigned int x=0; x < mtrx_width; x++)
#else
      // Get a global concurrent queue (system-managed thread pool)
      dispatch_queue_t concurrent_queue =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

      // dispatch_apply submits each iteration as a task to the queue
      dispatch_apply(mtrx_width,
                     concurrent_queue,
                     ^(size_t x)
#endif //__APPLE__
      {
        const unsigned int val_idx = y*mtrx_width + x;
        matrix_sum[val_idx] = matrix_0[val_idx] + matrix_1[val_idx];
      }
    }
  }
};

#endif //KERNEL_BENCHMARKS_CPU_H