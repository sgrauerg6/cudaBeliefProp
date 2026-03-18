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

#include <cmath>
#include <math.h>
#include <iostream>
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/UtilityFuncts.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunImpCPU/SIMDProcessing.h"
#include "benchmarksRunProcessing/BnchmrksConstsEnumsAliases.h"

//#define DONT_USE_GRAND_CENTRAL_DISPATCH
#define GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT

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
   * @param matrix_input_0
   * @param matrix_input_1
   * @param matrix_result
   */
  template <RunData_t T, benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkNoPackedInstructions(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const T* matrix_input_0, const T* matrix_input_1,
    T* matrix_result, unsigned int num_threads);
  
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_input_0, const float* matrix_input_1,
    float* matrix_result, unsigned int num_threads);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_input_0, const double* matrix_input_1,
    double* matrix_result, unsigned int num_threads);

#if defined(COMPILING_FOR_ARM)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsNEON(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float16_t* matrix_input_0, const float16_t* matrix_input_1,
    float16_t* matrix_result, unsigned int num_threads);
#endif //COMPILING_FOR_ARM

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_input_0, const float* matrix_input_1,
    float* matrix_result, unsigned int num_threads);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_input_0, const double* matrix_input_1,
    double* matrix_result, unsigned int num_threads);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const short* matrix_input_0, const short* matrix_input_1,
    short* matrix_result, unsigned int num_threads);

#if defined(FLOAT16_VECTORIZATION)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const _Float16* matrix_input_0, const _Float16* matrix_input_1,
    _Float16* matrix_result, unsigned int num_threads);
#endif //FLOAT16_VECTORIZATION

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const float* matrix_input_0, const float* matrix_input_1,
    float* matrix_result, unsigned int num_threads);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const double* matrix_input_0, const double* matrix_input_1,
    double* matrix_result, unsigned int num_threads);

  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const short* matrix_input_0, const short* matrix_input_1,
    short* matrix_result, unsigned int num_threads);

#if defined(FLOAT16_VECTORIZATION)
  template <benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkUseSIMDVectorsAVX256(
    unsigned int mtrx_width, unsigned int mtrx_height,
    const _Float16* matrix_input_0, const _Float16* matrix_input_1,
    _Float16* matrix_result, unsigned int num_threads);
#endif //FLOAT16_VECTORIZATION

  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, benchmarks::BenchmarkRun BENCHMARK_RUN>
  void TwoDMatricesBnchmrkSIMD(
  unsigned int mtrx_width, unsigned int mtrx_height,
    const T* matrix_input_0, const T* matrix_input_1,
    T* matrix_result, unsigned int num_threads);
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
  const T* matrix_input_0, const T* matrix_input_1,
  T* matrix_result, unsigned int num_threads)
{
#if !defined(__APPLE__) || defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  #pragma omp parallel for
  for (unsigned int y=0; y < mtrx_height; y++)
  {
#else
  //parallelize on apple processor using Grand Central Dispatch
  //get a global concurrent queue (system-managed thread pool)
  dispatch_queue_t concurrent_queue =
    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

#if defined(GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT)
  //std::cout << "NUM THREADS: " << num_threads << std::endl;
  //dispatch _apply submits each iteration as a task to the queue
  const unsigned int num_rows_per_thread{mtrx_height / num_threads};
  dispatch_apply(num_threads,
                 concurrent_queue,
                 ^(size_t thread_num)
  {
  for (unsigned int y_thread = 0; y_thread < num_rows_per_thread; y_thread++)
  {
    unsigned int y = (num_rows_per_thread * thread_num) + y_thread;
#else
  //dispatch_apply submits each iteration as a task to the queue
  dispatch_apply(mtrx_height,
                 concurrent_queue,
                 ^(size_t y)
  {
#endif //GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT
#endif //__APPLE__
    for (unsigned int x=0; x < mtrx_width; x++)
    {
      const unsigned int val_idx = y*mtrx_width + x;
      if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kAddTwoD) {
        matrix_result[val_idx] = matrix_input_0[val_idx] + matrix_input_1[val_idx];
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kDivideTwoD) {
        matrix_result[val_idx] = matrix_input_0[val_idx] / matrix_input_1[val_idx];
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kCopyTwoD) {
        matrix_result[val_idx] = matrix_input_0[val_idx];
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kGemm) {
        T sum{0};
        size_t curr_matrix_input0_idx{y * mtrx_width};
        size_t curr_matrix_input1_idx{x};
        for (unsigned int k = 0; k < mtrx_width; k++) {
#if defined(USE_FUSED_MULTIPLY_ADD)
          sum = std::fma(
            matrix_input_0[curr_matrix_input0_idx],
            matrix_input_1[curr_matrix_input1_idx],
            sum);
#else
          sum += 
            matrix_input_0[curr_matrix_input0_idx] *
            matrix_input_1[curr_matrix_input1_idx];
#endif //USE_FUSED_MULTIPLY_ADD
          curr_matrix_input0_idx += 1;
          curr_matrix_input1_idx += mtrx_width;
        }
        matrix_result[val_idx] = sum;
      }
    }
  }
#if defined(__APPLE__) && !defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
#if defined(GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT)
  }
#endif //GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT
  );
#endif //__APPLE__
}

template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkSIMD(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const T* matrix_input_0, const T* matrix_input_1,
  T* matrix_result, unsigned int num_threads)
{
  constexpr size_t simd_data_size{sizeof(U) / sizeof(T)};

  #if !defined(__APPLE__) || defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
  #pragma omp parallel for
  for (unsigned int y = 0; y < mtrx_height; y++)
  {
#else
  //parallelize on apple processor using Grand Central Dispatch
  //get a global concurrent queue (system-managed thread pool)
  dispatch_queue_t concurrent_queue =
    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

#if defined(GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT)
  //std::cout << "NUM THREADS: " << num_threads << std::endl;
  //dispatch _apply submits each iteration as a task to the queue
  const unsigned int num_rows_per_thread{mtrx_height / num_threads};
  dispatch_apply(num_threads,
                 concurrent_queue,
                 ^(size_t thread_num)
  {
  for (unsigned int y_thread = 0; y_thread < num_rows_per_thread; y_thread++)
  {
    unsigned int y = (num_rows_per_thread * thread_num) + y_thread;
#else
  //dispatch _apply submits each iteration as a task to the queue
  dispatch_apply(mtrx_height,
                 concurrent_queue,
                 ^(size_t y)
  {
#endif //GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT
#endif //__APPLE__
    //for now assuming that matrix width is multiple of simd data size
    for (unsigned int x_val = 0; x_val < mtrx_width; x_val += simd_data_size)
    {
      const unsigned int val_idx = y*mtrx_width + x_val;
      if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kAddTwoD) {
        simd_processing::StorePackedDataAligned<T, W>(
          val_idx,
          matrix_result,
          simd_processing::AddVals<U, U, W>(
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_input_0),
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_input_1)));
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kDivideTwoD) {
        simd_processing::StorePackedDataAligned<T, W>(
          val_idx,
          matrix_result,
          simd_processing::DivideVals<U, U, W>(
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_input_0),
            simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_input_1)));
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kCopyTwoD) {
        simd_processing::StorePackedDataAligned<T, W>(
          val_idx,
          matrix_result,
          simd_processing::LoadPackedDataAligned<T, U>(val_idx, matrix_input_0));
      }
      else if constexpr (BENCHMARK_RUN == benchmarks::BenchmarkRun::kGemm) {
        W sum = simd_processing::createSIMDVectorSameData<W>(0.0f);
        size_t curr_matrix_input0_idx{y * mtrx_width};
        size_t curr_matrix_input1_idx{x_val};
        for (unsigned int k = 0; k < mtrx_width; k++) {
          W addend_0 = simd_processing::createSIMDVectorSameData<W>(
            util_functs::ConvertValToDifferentDataTypeIfNeeded<T, float>(
              matrix_input_0[curr_matrix_input0_idx]));
          U addend_1 = simd_processing::LoadPackedDataAligned<T, U>(
            curr_matrix_input1_idx, matrix_input_1);
#if defined(USE_FUSED_MULTIPLY_ADD)
          sum = simd_processing::FusedMultAddVals<W, U, W, W>(
                  addend_0, addend_1, sum);
#else
          //const auto product = simd_processing::MultVals<W, U, W>(addend_0, addend_1);
          sum = simd_processing::AddVals<W, W, W>(
            sum,
            simd_processing::MultVals<W, U, W>(addend_0, addend_1));
#endif //USE_FUSED_MULTIPLY_ADD
          curr_matrix_input0_idx += 1;
          curr_matrix_input1_idx += mtrx_width;
        }
        simd_processing::StorePackedDataAligned<T, W>(
          val_idx,
          matrix_result,
          sum);
      }
    }
  }
#if defined(__APPLE__) && !defined(DONT_USE_GRAND_CENTRAL_DISPATCH)
#if defined(GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT)
  }
#endif //GRAND_CENTRAL_DISPATCH_SPECIFY_THREAD_COUNT
  );
#endif //__APPLE__
}

#endif //KERNEL_BNCHMRKS_OPT_CPU_H