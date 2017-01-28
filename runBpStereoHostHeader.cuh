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

//This function declares the host functions to run the CUDA implementation of Stereo estimation using BP

#ifndef RUN_BP_STEREO_HOST_HEADER_CUH
#define RUN_BP_STEREO_HOST_HEADER_CUH

#include "bpStereoCudaParameters.cuh"

//include for the kernal functions to be run on the GPU
#include "kernalBpStereo.cu"


texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard2NotDecInKern;

texture<float, 1, cudaReadModeElementType> messageUPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageDPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageLPrevTexStereoCheckerboard1NotDecInKern;
texture<float, 1, cudaReadModeElementType> messageRPrevTexStereoCheckerboard1NotDecInKern;

//functions directed related to running BP to retrieve the movement between the images

//set the current BP settings in the host in constant memory on the device
__host__ void setBPSettingInConstMem(BPsettings& currentBPSettings);


//run the given number of iterations of BP at the current level using the given message values in global device memory
__host__ void runBPAtCurrentLevel(int& numIterationsAtLevel, int& widthLevelInt, int& heightLevelInt, size_t& dataTexOffset,
		float*& messageUDeviceCheckerboard1, float*& messageDDeviceCheckerboard1, float*& messageLDeviceCheckerboard1, 
		float*& messageRDeviceCheckerboard1, float*& messageUDeviceCheckerboard2, float*& messageDDeviceCheckerboard2, float*& messageLDeviceCheckerboard2, 
		float*& messageRDeviceCheckerboard2, dim3& grid, dim3& threads, int& numBytesDataAndMessageSetInCheckerboardAtLevel);

//run the given number of iterations of BP at the current level using the given message values in global device memory without using textures
__host__ void runBPAtCurrentLevelNoTextures(int& numIterationsAtLevel, int& widthLevelActualIntegerSize, int& heightLevelActualIntegerSize, 
	float*& messageUDeviceCheckerboard1, float*& messageDDeviceCheckerboard1, float*& messageLDeviceCheckerboard1, 
	float*& messageRDeviceCheckerboard1, float*& messageUDeviceCheckerboard2, float*& messageDDeviceCheckerboard2, float*& messageLDeviceCheckerboard2, 
	float*& messageRDeviceCheckerboard2, dim3& grid, dim3& threads, int& numBytesDataAndMessageSetInCheckerboardAtLevel,
	float*& dataCostDeviceCheckerboard1, float*& dataCostDeviceCheckerboard2, int& offsetDataLevel); 

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
__host__ void copyMessageValuesToNextLevelDown(int& widthLevelInt, int& heightLevelInt,
		float*& messageUDeviceCheckerboard1CopyFrom, float*& messageDDeviceCheckerboard1CopyFrom, float*& messageLDeviceCheckerboard1CopyFrom, 
		float*& messageRDeviceCheckerboard1CopyFrom, float*& messageUDeviceCheckerboard2CopyFrom, float*& messageDDeviceCheckerboard2CopyFrom, 
		float*& messageLDeviceCheckerboard2CopyFrom, float*& messageRDeviceCheckerboard2CopyFrom, float*& messageUDeviceCheckerboard1CopyTo, 
		float*& messageDDeviceCheckerboard1CopyTo, float*& messageLDeviceCheckerboard1CopyTo, float*& messageRDeviceCheckerboard1CopyTo, 
		float*& messageUDeviceCheckerboard2CopyTo, float*& messageDDeviceCheckerboard2CopyTo, float*& messageLDeviceCheckerboard2CopyTo, 
		float*& messageRDeviceCheckerboard2CopyTo, int& numBytesDataAndMessageSetInCheckerboardAtLevel, dim3& grid, dim3& threads);

//initialize the data cost at each pixel for each disparity value
__host__ void initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& dataCostDeviceCheckerboard1, float*& dataCostDeviceCheckerboard2, BPsettings& algSettings);


//initialize the message values for every pixel at every disparity to DEFAULT_INITIAL_MESSAGE_VAL (value is 0.0f unless changed)
__host__ void initializeMessageValsToDefault(float*& messageUDeviceSet0Checkerboard1, float*& messageDDeviceSet0Checkerboard1, float*& messageLDeviceSet0Checkerboard1, float*& messageRDeviceSet0Checkerboard1,
												float*& messageUDeviceSet0Checkerboard2, float*& messageDDeviceSet0Checkerboard2, float*& messageLDeviceSet0Checkerboard2, float*& messageRDeviceSet0Checkerboard2,
												int widthOfCheckerboard, int heightOfCheckerboard, int numPossibleMovements);


//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
__host__ void runBeliefPropStereoCUDA(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& resultingDisparityMapDevice, BPsettings& algSettings);

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
