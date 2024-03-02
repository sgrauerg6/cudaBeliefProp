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

#include "RunImpCUDA/RunCUDASettings.h"

#define USE_SHARED_MEMORY 0
#define DISP_INDEX_START_REG_LOCAL_MEM 0

//uncomment to use template specialization functions that check if "val to normalize" in
//message processing is valid when processing in CUDA half precision...have found with many
//levels (maybe around 7) that value can become invalid due to lesser precision using half
//type
//off by default since may make processing take a little longer and complicates code a little
//#define CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

#endif // BP_STEREO_CUDA_PARAMETERS_H
