/*
 * RunCUDASettings.h
 *
 *  Created on: Jan 27, 2024
 *      Author: scott
 */

#ifndef RUNCUDASETTINGS_H_
#define RUNCUDASETTINGS_H_

//set data type used for half-precision with CUDA
#ifdef USE_BFLOAT16_FOR_HALF_PRECISION
#include <cuda_bf16.h>
using halftype = __nv_bfloat16;
#else
#include <cuda_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION

#endif /* RUNCUDASETTINGS_H_ */
