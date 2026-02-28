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
 * @file KernelBnchmrksOptCPU_AVX512.h
 * @author Scott Grauer-Gray
 * @brief Defines functions used in processing benchmarks that are
 * specific to implementation with AVX512 vectorization
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef KERNEL_BNCHMRKS_OPT_CPU_AVX512_H_
#define KERNEL_BNCHMRKS_OPT_CPU_AVX512_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>
#include "BpSharedFuncts/SharedBpProcessingFuncts.h"
#include "RunImpCPU/AVX512TemplateSpFuncts.h"
#include "RunImpCPU/RunCPUSettings.h"

template <benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const float* matrix_addend_0, const float* matrix_addend_1,
  float* matrix_sum)
{
  TwoDMatricesBnchmrkSIMD<float, __m512, float, __m512, BENCHMARK_RUN>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

template <benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const double* matrix_addend_0, const double* matrix_addend_1,
  double* matrix_sum)
{
  TwoDMatricesBnchmrkSIMD<double, __m512d, double, __m512d, BENCHMARK_RUN>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

template <benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const short* matrix_addend_0, const short* matrix_addend_1,
  short* matrix_sum)
{
  TwoDMatricesBnchmrkSIMD<short, __m256i, float, __m512, BENCHMARK_RUN>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

#if defined(FLOAT16_VECTORIZATION)

template <benchmarks::BenchmarkRun BENCHMARK_RUN>
void benchmarks_cpu::TwoDMatricesBnchmrkUseSIMDVectorsAVX512(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const _Float16* matrix_addend_0, const _Float16* matrix_addend_1,
  _Float16* matrix_sum)
{
  TwoDMatricesBnchmrkSIMD<_Float16, __m512h, _Float16, __m512h, BENCHMARK_RUN>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

#endif //FLOAT16_VECTORIZATION

#endif /* KERNEL_BNCHMRKS_OPT_CPU_AVX512_H_ */
