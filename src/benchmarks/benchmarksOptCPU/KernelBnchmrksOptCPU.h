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
 * @file KernelBnchmrksOptCPU.h
 * @author Scott Grauer-Gray
 * @brief This header declares the kernel functions to run optimized benchmarks
 * on CPU
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef KERNEL_BNCHMRKS_OPT_CPU_H
#define KERNEL_BNCHMRKS_OPT_CPU_H

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
#include "benchmarksRunProcessing/BnchmrksConstsEnumsAliases.h"

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
   * @param mtrx_width
   * @param mtrx_height
   * @param matrix_addend_0
   * @param matrix_addend_1
   * @param matrix_sum
   */
  template <RunData_t T, benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkNoPackedInstructions(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const T* matrix_addend_0, const T* matrix_addend_1,
    T* matrix_sum);
  
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_addend_0, const float* matrix_addend_1,
    float* matrix_sum);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_addend_0, const double* matrix_addend_1,
    double* matrix_sum);

#if defined(COMPILING_FOR_ARM)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float16_t* matrix_addend_0, const float16_t* matrix_addend_1,
    float16_t* matrix_sum);
#endif //COMPILING_FOR_ARM

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_addend_0, const float* matrix_addend_1,
    float* matrix_sum);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_addend_0, const double* matrix_addend_1,
    double* matrix_sum);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const short* matrix_addend_0, const short* matrix_addend_1,
    short* matrix_sum);

#if defined(FLOAT16_VECTORIZATION)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const _Float16* matrix_addend_0, const _Float16* matrix_addend_1,
    _Float16* matrix_sum);
#endif //FLOAT16_VECTORIZATION

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_addend_0, const float* matrix_addend_1,
    float* matrix_sum);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_addend_0, const double* matrix_addend_1,
    double* matrix_sum);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const short* matrix_addend_0, const short* matrix_addend_1,
    short* matrix_sum);

#if defined(FLOAT16_VECTORIZATION)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const _Float16* matrix_addend_0, const _Float16* matrix_addend_1,
    _Float16* matrix_sum);
#endif //FLOAT16_VECTORIZATION

  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkSIMD(
  unsigned int mtrx_width, unsigned int mtrx_height,
    const T* matrix_addend_0, const T* matrix_addend_1,
    T* matrix_sum);
};

//headers to include differ depending on architecture and CPU vectorization setting
#if defined(COMPILING_FOR_ARM)

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
#include "KernelBnchmrksOptCPU_NEON.h"
#endif //CPU_VECTORIZATION_DEFINE == NEON_DEFINE

#else

#if ((CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE) || (CPU_VECTORIZATION_DEFINE == AVX_256_F16_DEFINE))
#include "KernelBnchmrksOptCPU_AVX256.h"
#elif ((CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE) || (CPU_VECTORIZATION_DEFINE == AVX_512_F16_DEFINE))
#include "KernelBnchmrksOptCPU_AVX256.h"
#include "KernelBnchmrksOptCPU_AVX512.h"
#endif //CPU_VECTORIZATION_DEFINE

#endif //COMPILING_FOR_ARM

template <RunData_t T, benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkNoPackedInstructions(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const T* matrix_addend_0, const T* matrix_addend_1,
  T* matrix_sum)
{
#if !defined(__APPLE__) || defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  #pragma omp parallel for
  for (unsigned int y=0; y < mtrx_height; y++)
#else
  //parallelize on apple processor using Grand Central Dispatch
  //get a global concurrent queue (system-managed thread pool)
  dispatch_queue_t concurrent_queue =
    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  //dispatch_apply submits each iteration as a task to the queue
  dispatch_apply(mtrx_height,
                 concurrent_queue,
                 ^(size_t y)
#endif //__APPLE__
  {
    for (unsigned int x=0; x < mtrx_width; x++)
    {
      const unsigned int val_idx = y*mtrx_width + x;
      if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kAddTwoD) {
        matrix_sum[val_idx] = matrix_addend_0[val_idx] + matrix_addend_1[val_idx];
      }
    }
  }
#if defined(__APPLE__) && !defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  );
#endif //__APPLE__
}

template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkSIMD(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const T* matrix_addend_0, const T* matrix_addend_1,
  T* matrix_sum)
{
  constexpr size_t simd_data_size{sizeof(U) / sizeof(T)};

  #if !defined(__APPLE__) || defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  #pragma omp parallel for
  for (unsigned int y = 0; y < mtrx_height; y++)
#else
  //parallelize on apple processor using Grand Central Dispatch
  //get a global concurrent queue (system-managed thread pool)
  dispatch_queue_t concurrent_queue =
    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

  //dispatch _apply submits each iteration as a task to the queue
  dispatch_apply(mtrx_height,
                 concurrent_queue,
                 ^(size_t y)
#endif //__APPLE__
  {
    //for now assuming that matrix width is multiple of simd data size
    for (unsigned int x_val = 0; x_val < mtrx_width; x_val += simd_data_size)
    {
      const unsigned int val_idx = y*mtrx_width + x_val;
      if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kAddTwoD) {
        simd_processing::StorePackedDataAligned<T, W>(
          val_idx,
          matrix_sum,
          simd_processing::AddVals<U, U, W>(
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_addend_0),
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_addend_1)));
      }
    }
  }
#if defined(__APPLE__) && !defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  );
#endif //__APPLE__
}

#endif //KERNEL_BNCHMRKS_OPT_CPU_H