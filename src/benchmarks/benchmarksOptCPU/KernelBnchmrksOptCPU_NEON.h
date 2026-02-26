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
 * @file KernelBnchmrksOptCPU_NEON.h
 * @author Scott Grauer-Gray
 * @brief Defines functions used in processing benchmarks that are
 * specific to implementation with NEON vectorization
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef KERNEL_BNCHMRKS_OPT_CPU_NEON_H_
#define KERNEL_BNCHMRKS_OPT_CPU_NEON_H_

//this is only used when processing using an ARM CPU with NEON instructions
#include <arm_neon.h>
#include "RunImpCPU/NEONTemplateSpFuncts.h"

//using inline since otherwise get duplicate symbol error
inline void benchmarks_cpu::AddMatricesUseSIMDVectorsNEON(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const float* matrix_addend_0, const float* matrix_addend_1,
  float* matrix_sum)
{
  AddMatricesSIMD<float, float32x4_t, float, float32x4_t>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

inline void benchmarks_cpu::AddMatricesUseSIMDVectorsNEON(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const double* matrix_addend_0, const double* matrix_addend_1,
  double* matrix_sum)
{
  AddMatricesSIMD<double, float64x2_t, double, float64x2_t>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

inline void benchmarks_cpu::AddMatricesUseSIMDVectorsNEON(
  unsigned int mtrx_width, unsigned int mtrx_height,
  const float16_t* matrix_addend_0, const float16_t* matrix_addend_1,
  float16_t* matrix_sum)
{
  AddMatricesSIMD<float16_t, float16x4_t, float, float32x4_t>(mtrx_width, mtrx_height,
    matrix_addend_0, matrix_addend_1, matrix_sum);
}

#endif /* KERNEL_BNCHMRKS_OPT_CPU_NEON_H_ */
