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
#include <cuda_fp16.h>

namespace bp_cuda_params
{
	//defines the default width and height of the thread block used for
	//kernel functions when running BP
	constexpr unsigned int DEFAULT_BLOCK_SIZE_WIDTH_BP{32};
	constexpr unsigned int DEFAULT_BLOCK_SIZE_HEIGHT_BP{4};

	enum CudaKernel { BLUR_IMAGES, DATA_COSTS_AT_LEVEL, INIT_MESSAGE_VALS, BP_AT_LEVEL,
	                  COPY_AT_LEVEL, OUTPUT_DISP };
    constexpr unsigned int NUM_KERNELS{6u};

    //structure containing CUDA parameters including thread block dimensions
	//to use at each BP level
	struct CudaParameters {
		CudaParameters(unsigned int numLevels) {
			setThreadBlockDims({bp_cuda_params::DEFAULT_BLOCK_SIZE_WIDTH_BP, bp_cuda_params::DEFAULT_BLOCK_SIZE_HEIGHT_BP}, numLevels);
		  };
		//std::vector<std::array<unsigned int, 2>> blockDimsXY_;
		void setThreadBlockDims(const std::array<unsigned int, 2>& tbDims, unsigned int numLevels) {
			blockDimsXYEachKernel_[BLUR_IMAGES] = {tbDims};
			blockDimsXYEachKernel_[DATA_COSTS_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels, tbDims);
			blockDimsXYEachKernel_[INIT_MESSAGE_VALS] = {tbDims};
			blockDimsXYEachKernel_[BP_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels, tbDims);
			blockDimsXYEachKernel_[COPY_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels, tbDims);
			blockDimsXYEachKernel_[OUTPUT_DISP] = {tbDims};
		}
		std::array<std::vector<std::array<unsigned int, 2>>, NUM_KERNELS> blockDimsXYEachKernel_;
		bool useSharedMemory_{false};
    };
}

#define USE_SHARED_MEMORY 0
#define DISP_INDEX_START_REG_LOCAL_MEM 0

//uncomment to use template specialization functions that check if "val to normalize" in
//message processing is valid when processing in CUDA half precision...have found with many
//levels (maybe around 7) that value can become invalid due to lesser precision using half
//type
//off by default since may make processing take a little longer and complicates code a little
//#define CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

#endif // BP_STEREO_CUDA_PARAMETERS_H
