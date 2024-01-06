/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//This class defines parameters for the cuda implementation for disparity map estimation for a pair of stereo images

#ifndef BP_STEREO_CUDA_PARAMETERS_H
#define BP_STEREO_CUDA_PARAMETERS_H

#include "bpStereoParameters.h"
#include "bpRunSettings.h"
#include "bpStructsAndEnums.h"
#include <vector>

//determine whether or not to support CUDA half-precision
//comment out if not supporting CUDA half-precision
//remove (or don't use) capability for half precision if using GPU with compute capability under 5.3
//half precision currently only supported on CPU if using GPU with compute capability under 5.3
#define CUDA_HALF_SUPPORT

//uncomment to use bfloat16 data type for half precision in CUDA (only supported in compute capability sm_80 and later)
//#define USE_BFLOAT16_FOR_HALF_PRECISION

#ifdef CUDA_HALF_SUPPORT
//set data type used for half-precision with CUDA
#ifdef USE_BFLOAT16_FOR_HALF_PRECISION
#include <cuda_bf16.h>
using halftype = __nv_bfloat16;
#else
#include <cuda_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION
#endif //CUDA_HALF_SUPPORT

#define USE_SHARED_MEMORY 0
#define DISP_INDEX_START_REG_LOCAL_MEM 0

//uncomment to use template specialization functions that check if "val to normalize" in
//message processing is valid when processing in CUDA half precision...have found with many
//levels (maybe around 7) that value can become invalid due to lesser precision using half
//type
//off by default since may make processing take a little longer and complicates code a little
//#define CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

#endif // BP_STEREO_CUDA_PARAMETERS_H
