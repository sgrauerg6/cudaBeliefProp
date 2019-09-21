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

//remove (or don't use) capability for half precision if using GPU with compute capability under 5.3
//half precision currently only supported on CPU if using GPU with compute capability under 5.3
#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))
#include <cuda_fp16.h>
#endif
#include <cuda_fp16.h>

//defines the width and height of the thread block used for
//image filtering (applying the Guassian filter in smoothImageHost)
#define BLOCK_SIZE_WIDTH_FILTER_IMAGES 16
#define BLOCK_SIZE_HEIGHT_FILTER_IMAGES 16

//defines the width and height of the thread block used for
//each kernal function when running BP (right now same thread
//block dimensions are used for each kernal function when running
//kernal function in runBpStereoHost.cu, though this could be
//changed)
#define BLOCK_SIZE_WIDTH_BP 32
#define BLOCK_SIZE_HEIGHT_BP 4

#ifdef USE_SHARED_MEMORY_FROM_PYTHON
#define USE_SHARED_MEMORY USE_SHARED_MEMORY_FROM_PYTHON
#else
#define USE_SHARED_MEMORY 0
#endif


#ifdef DISP_INDEX_START_REG_LOCAL_MEM_FROM_PYTHON
#define DISP_INDEX_START_REG_LOCAL_MEM DISP_INDEX_START_REG_LOCAL_MEM_FROM_PYTHON
#else
#define DISP_INDEX_START_REG_LOCAL_MEM 0
#endif


#endif // BP_STEREO_CUDA_PARAMETERS_H
