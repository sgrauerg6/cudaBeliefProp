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

//Defines the functions to run the CUDA implementation of 2-D Stereo estimation using BP

#include "runBpStereoHostHeader.cuh"
#include <chrono>
#include <cuda_fp16.h>

#define RUN_DETAILED_TIMING

double timeCopyDataKernelTotalTime = 0.0;
double timeBpItersKernelTotalTime = 0.0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
__host__ void runBPAtCurrentLevel(int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
	T* messageUDeviceCheckerboard1, T* messageDDeviceCheckerboard1, T* messageLDeviceCheckerboard1,
	T* messageRDeviceCheckerboard1, T* messageUDeviceCheckerboard2, T* messageDDeviceCheckerboard2, T* messageLDeviceCheckerboard2,
	T* messageRDeviceCheckerboard2, T* dataCostDeviceCheckerboard1,
	T* dataCostDeviceCheckerboard2)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSize);
	grid.x = (unsigned int) ceil(
			(float) (widthCheckerboard) / (float) threads.x); //only updating half at a time
	grid.y = (unsigned int) ceil((float) heightLevelActualIntegerSize / (float) threads.y);

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < numIterationsAtLevel; iterationNum++)
	{
		int checkboardPartUpdate = CHECKERBOARD_PART_2;

		if ((iterationNum % 2) == 0)
		{
			checkboardPartUpdate = CHECKERBOARD_PART_2;
		}
		else
		{
			checkboardPartUpdate = CHECKERBOARD_PART_1;
		}

		(cudaDeviceSynchronize());

#ifdef RUN_DETAILED_TIMING
		auto timeBpItersKernelStart = std::chrono::system_clock::now();
#endif

		runBPIterationUsingCheckerboardUpdatesNoTextures<T><<<grid, threads>>>(
				dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
				messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
				messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
				messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
				messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
				widthLevelActualIntegerSize, heightLevelActualIntegerSize,
				checkboardPartUpdate);

		(cudaDeviceSynchronize());

		/*printDataAndMessageValsAtPoint<T>(50, 22, dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
				messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
								messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
								messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
								messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
								messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
												messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
												messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
												messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
						getCheckerboardWidth<T>(widthLevelActualIntegerSize), heightLevelActualIntegerSize, 0);*/

#ifdef RUN_DETAILED_TIMING

		auto timeBpItersKernelEnd = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = timeBpItersKernelEnd
				- timeBpItersKernelStart;

		timeBpItersKernelTotalTime += diff.count();

#endif

	}

	gpuErrchk( cudaPeekAtLastError() );
}

/*template<>
__host__ void runBPAtCurrentLevel<half2>(int numIterationsAtLevel, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
		half2* messageUDeviceCheckerboard1, half2* messageDDeviceCheckerboard1, half2* messageLDeviceCheckerboard1,
		half2* messageRDeviceCheckerboard1, half2* messageUDeviceCheckerboard2, half2* messageDDeviceCheckerboard2, half2* messageLDeviceCheckerboard2,
		half2* messageRDeviceCheckerboard2, half2* dataCostDeviceCheckerboard1,
		half2* dataCostDeviceCheckerboard2)
{
	runBPAtCurrentLevel<half>(numIterationsAtLevel, widthLevelActualIntegerSize, heightLevelActualIntegerSize,
		(half*)messageUDeviceCheckerboard1, (half*)messageDDeviceCheckerboard1, (half*)messageLDeviceCheckerboard1,
		(half*)messageRDeviceCheckerboard1, (half*)messageUDeviceCheckerboard2, (half*)messageDDeviceCheckerboard2, (half*)messageLDeviceCheckerboard2,
		(half*)messageRDeviceCheckerboard2, (half*)dataCostDeviceCheckerboard1,
		(half*)dataCostDeviceCheckerboard2);
}*/


//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
__host__ void copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
	int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
	T* messageUDeviceCheckerboard1CopyFrom, T* messageDDeviceCheckerboard1CopyFrom, T* messageLDeviceCheckerboard1CopyFrom,
	T* messageRDeviceCheckerboard1CopyFrom, T* messageUDeviceCheckerboard2CopyFrom, T* messageDDeviceCheckerboard2CopyFrom,
	T* messageLDeviceCheckerboard2CopyFrom, T* messageRDeviceCheckerboard2CopyFrom, T** messageUDeviceCheckerboard1CopyTo,
	T** messageDDeviceCheckerboard1CopyTo, T** messageLDeviceCheckerboard1CopyTo, T** messageRDeviceCheckerboard1CopyTo,
	T** messageUDeviceCheckerboard2CopyTo, T** messageDDeviceCheckerboard2CopyTo, T** messageLDeviceCheckerboard2CopyTo,
	T** messageRDeviceCheckerboard2CopyTo)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSizeNextLevel);
	grid.x = (unsigned int)ceil((float)(widthCheckerboard / 2.0f) / (float)threads.x);
	grid.y = (unsigned int)ceil((float)(heightLevelActualIntegerSizeNextLevel / 2.0f) / (float)threads.y);

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

	//update the number of bytes needed to store each set
	int numBytesDataAndMessageSetInCheckerboardAtLevel = widthCheckerboard * heightLevelActualIntegerSizeNextLevel * totalPossibleMovements * sizeof(T);

	//allocate space in the GPU for the message values in the checkerboard set to copy to
	(cudaMalloc((void**) messageUDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageDDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageLDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageRDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));

	(cudaMalloc((void**) messageUDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageDDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageLDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) messageRDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));

#endif

	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeCopyDataKernelStart = std::chrono::system_clock::now();

#endif

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures<T> <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			*messageUDeviceCheckerboard1CopyTo, *messageDDeviceCheckerboard1CopyTo, *messageLDeviceCheckerboard1CopyTo,
			*messageRDeviceCheckerboard1CopyTo, *messageUDeviceCheckerboard2CopyTo, *messageDDeviceCheckerboard2CopyTo, *messageLDeviceCheckerboard2CopyTo,
			*messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			widthLevelActualIntegerSizeNextLevel, heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_1);

	( cudaDeviceSynchronize() );

	copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures<T> <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			*messageUDeviceCheckerboard1CopyTo, *messageDDeviceCheckerboard1CopyTo, *messageLDeviceCheckerboard1CopyTo,
			*messageRDeviceCheckerboard1CopyTo, *messageUDeviceCheckerboard2CopyTo, *messageDDeviceCheckerboard2CopyTo, *messageLDeviceCheckerboard2CopyTo,
			*messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			widthLevelActualIntegerSizeNextLevel, heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_2);

	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeCopyDataKernelEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = timeCopyDataKernelEnd-timeCopyDataKernelStart;

	timeCopyDataKernelTotalTime += diff.count();

#endif

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	//free the now-copied from computed data of the completed level
	cudaFree(messageUDeviceCheckerboard1CopyFrom);
	cudaFree(messageDDeviceCheckerboard1CopyFrom);
	cudaFree(messageLDeviceCheckerboard1CopyFrom);
	cudaFree(messageRDeviceCheckerboard1CopyFrom);

	cudaFree(messageUDeviceCheckerboard2CopyFrom);
	cudaFree(messageDDeviceCheckerboard2CopyFrom);
	cudaFree(messageLDeviceCheckerboard2CopyFrom);
	cudaFree(messageRDeviceCheckerboard2CopyFrom);

#endif
	gpuErrchk( cudaPeekAtLastError() );
}

template<>
__host__ void copyMessageValuesToNextLevelDown<half2>(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
	int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
	half2* messageUDeviceCheckerboard1CopyFrom, half2* messageDDeviceCheckerboard1CopyFrom, half2* messageLDeviceCheckerboard1CopyFrom,
	half2* messageRDeviceCheckerboard1CopyFrom, half2* messageUDeviceCheckerboard2CopyFrom, half2* messageDDeviceCheckerboard2CopyFrom,
	half2* messageLDeviceCheckerboard2CopyFrom, half2* messageRDeviceCheckerboard2CopyFrom, half2** messageUDeviceCheckerboard1CopyTo,
	half2** messageDDeviceCheckerboard1CopyTo, half2** messageLDeviceCheckerboard1CopyTo, half2** messageRDeviceCheckerboard1CopyTo,
	half2** messageUDeviceCheckerboard2CopyTo, half2** messageDDeviceCheckerboard2CopyTo, half2** messageLDeviceCheckerboard2CopyTo,
	half2** messageRDeviceCheckerboard2CopyTo)
{
	copyMessageValuesToNextLevelDown<half>(widthLevelActualIntegerSizePrevLevel, heightLevelActualIntegerSizePrevLevel,
		widthLevelActualIntegerSizeNextLevel, heightLevelActualIntegerSizeNextLevel,
		(half*)messageUDeviceCheckerboard1CopyFrom, (half*)messageDDeviceCheckerboard1CopyFrom, (half*)messageLDeviceCheckerboard1CopyFrom,
		(half*)messageRDeviceCheckerboard1CopyFrom, (half*)messageUDeviceCheckerboard2CopyFrom, (half*)messageDDeviceCheckerboard2CopyFrom,
		(half*)messageLDeviceCheckerboard2CopyFrom, (half*)messageRDeviceCheckerboard2CopyFrom, (half**)messageUDeviceCheckerboard1CopyTo,
		(half**)messageDDeviceCheckerboard1CopyTo, (half**)messageLDeviceCheckerboard1CopyTo, (half**)messageRDeviceCheckerboard1CopyTo,
		(half**)messageUDeviceCheckerboard2CopyTo, (half**)messageDDeviceCheckerboard2CopyTo, (half**)messageLDeviceCheckerboard2CopyTo,
		(half**)messageRDeviceCheckerboard2CopyTo);
}


//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T>
__host__ void initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, T* dataCostDeviceCheckerboard1, T* dataCostDeviceCheckerboard2, BPsettings& algSettings)
{
	//setup execution parameters
	//the thread size remains constant throughout but the grid size is adjusted based on the current level/kernal to run
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//kernal run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
	grid.x = (unsigned int)ceil((float)algSettings.widthImages / (float)threads.x);
	grid.y = (unsigned int)ceil((float)algSettings.heightImages / (float)threads.y);

	//initialize the data the the "bottom" of the image pyramid
	initializeBottomLevelDataStereo<T><<<grid, threads>>>(image1PixelsDevice,
			image2PixelsDevice, dataCostDeviceCheckerboard1,
			dataCostDeviceCheckerboard2, algSettings.widthImages,
			algSettings.heightImages);

	( cudaDeviceSynchronize() );
}

/*template<>
__host__ void initializeDataCosts<half2>(float*& image1PixelsDevice, float*& image2PixelsDevice, half2* dataCostDeviceCheckerboard1, half2* dataCostDeviceCheckerboard2, BPsettings& algSettings)
{
	initializeDataCosts<half>(image1PixelsDevice, image2PixelsDevice, (half*)dataCostDeviceCheckerboard1, (half*)dataCostDeviceCheckerboard2, algSettings);
}*/


//initialize the message values with no previous message values...all message values are set to 0
template<typename T>
__host__ void initializeMessageValsToDefault(T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
												  T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
												  int widthLevel, int heightLevel, int numPossibleMovements)
{
	int widthOfCheckerboard = getCheckerboardWidth<T>(widthLevel);

	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightLevel / (float)threads.y));

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	initializeMessageValsToDefault<T> <<< grid, threads >>> (messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1,
												messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
												messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, widthOfCheckerboard, heightLevel);

	cudaDeviceSynchronize();
}


/*(template<>
__host__ void initializeMessageValsToDefault<half2>(half2* messageUDeviceSet0Checkerboard1, half2* messageDDeviceSet0Checkerboard1, half2* messageLDeviceSet0Checkerboard1, half2* messageRDeviceSet0Checkerboard1,
		half2* messageUDeviceSet0Checkerboard2, half2* messageDDeviceSet0Checkerboard2, half2* messageLDeviceSet0Checkerboard2, half2* messageRDeviceSet0Checkerboard2,
		int widthLevel, int heightLevel, int numPossibleMovements)
{
	initializeMessageValsToDefault<half>((half*)messageUDeviceSet0Checkerboard1, (half*)messageDDeviceSet0Checkerboard1, (half*)messageLDeviceSet0Checkerboard1, (half*)messageRDeviceSet0Checkerboard1,
			(half*)messageUDeviceSet0Checkerboard2, (half*)messageDDeviceSet0Checkerboard2, (half*)messageLDeviceSet0Checkerboard2, (half*)messageRDeviceSet0Checkerboard2,
			widthLevel, heightLevel, numPossibleMovements);
}*/

template<typename T>
__host__ void initializeDataCurrentLevel(T* dataCostDeviceCheckerboard1,
		T* dataCostDeviceCheckerboard2, int prev_level_offset_level,
		int offsetLevel, int widthLevelActualIntegerSize,
		int heightLevelActualIntegerSize, int prevWidthLevelActualIntegerSize,
		int prevHeightLevelActualIntegerSize)
{
	int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSize);

	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
	//the four-connected neighbors are in the other checkerboard
	grid.x = (unsigned int) ceil(
			((float) widthCheckerboard) / (float) threads.x);
	grid.y = (unsigned int) ceil(
			(float) heightLevelActualIntegerSize / (float) threads.y);

	size_t offsetNum = 0;

	initializeCurrentLevelDataStereoNoTextures<T> <<<grid, threads>>>(
			&dataCostDeviceCheckerboard1[prev_level_offset_level],
			&dataCostDeviceCheckerboard2[prev_level_offset_level],
			&dataCostDeviceCheckerboard1[offsetLevel],
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_1, ((int) offsetNum / sizeof(float)));

	(cudaDeviceSynchronize());

	initializeCurrentLevelDataStereoNoTextures<T> <<<grid, threads>>>(
			&dataCostDeviceCheckerboard1[prev_level_offset_level],
			&dataCostDeviceCheckerboard2[prev_level_offset_level],
			&dataCostDeviceCheckerboard2[offsetLevel],
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_2, ((int) offsetNum / sizeof(float)));

	(cudaDeviceSynchronize());
}

/*template<>
__host__ void initializeDataCurrentLevel<half2>(half2* dataCostDeviceCheckerboard1,
		half2* dataCostDeviceCheckerboard2, int prev_level_offset_level,
		int offsetLevel, int widthLevelActualIntegerSize,
		int heightLevelActualIntegerSize, int prevWidthLevelActualIntegerSize,
		int prevHeightLevelActualIntegerSize)
{
	initializeDataCurrentLevel<half>((half*)dataCostDeviceCheckerboard1,
			(half*)dataCostDeviceCheckerboard2, prev_level_offset_level,
			offsetLevel, widthLevelActualIntegerSize,
			heightLevelActualIntegerSize, prevWidthLevelActualIntegerSize,
			prevHeightLevelActualIntegerSize);
}*/

template<typename T>
__host__ void retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1, T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2, T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int) ceil((float) widthLevel / (float) threads.x);
	grid.y = (unsigned int) ceil((float) heightLevel / (float) threads.y);

	if (currentCheckerboardSet == 0)
	{
		retrieveOutputDisparityCheckerboardStereoNoTextures<T> <<<grid, threads>>>(
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, resultingDisparityMapDevice,
				widthLevel, heightLevel);
	}
	else
	{
		retrieveOutputDisparityCheckerboardStereoNoTextures<T> <<<grid, threads>>>(
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, resultingDisparityMapDevice,
				widthLevel, heightLevel);
	}

	(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
}

/*template<>
__host__ void retrieveOutputDisparity<half2>(half2* dataCostDeviceCurrentLevelCheckerboard1, half2* dataCostDeviceCurrentLevelCheckerboard2,
		half2* messageUDeviceSet0Checkerboard1, half2* messageDDeviceSet0Checkerboard1, half2* messageLDeviceSet0Checkerboard1, half2* messageRDeviceSet0Checkerboard1,
		half2* messageUDeviceSet0Checkerboard2, half2* messageDDeviceSet0Checkerboard2, half2* messageLDeviceSet0Checkerboard2, half2* messageRDeviceSet0Checkerboard2,
		half2* messageUDeviceSet1Checkerboard1, half2* messageDDeviceSet1Checkerboard1, half2* messageLDeviceSet1Checkerboard1, half2* messageRDeviceSet1Checkerboard1,
		half2* messageUDeviceSet1Checkerboard2, half2* messageDDeviceSet1Checkerboard2, half2* messageLDeviceSet1Checkerboard2, half2* messageRDeviceSet1Checkerboard2,
		float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet)
{
	retrieveOutputDisparity<half>((half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
			(half*)messageUDeviceSet0Checkerboard1, (half*)messageDDeviceSet0Checkerboard1, (half*)messageLDeviceSet0Checkerboard1, (half*)messageRDeviceSet0Checkerboard1,
			(half*)messageUDeviceSet0Checkerboard2, (half*)messageDDeviceSet0Checkerboard2, (half*)messageLDeviceSet0Checkerboard2, (half*)messageRDeviceSet0Checkerboard2,
			(half*)messageUDeviceSet1Checkerboard1, (half*)messageDDeviceSet1Checkerboard1, (half*)messageLDeviceSet1Checkerboard1, (half*)messageRDeviceSet1Checkerboard1,
			(half*)messageUDeviceSet1Checkerboard2, (half*)messageDDeviceSet1Checkerboard2, (half*)messageLDeviceSet1Checkerboard2, (half*)messageRDeviceSet1Checkerboard2,
			resultingDisparityMapDevice, widthLevel, heightLevel, currentCheckerboardSet);
}*/

template<typename T>
__host__ void printDataAndMessageValsToPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1,
		T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2,
		T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1,
		T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2,
		T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		int widthCheckerboard, int heightLevel, int currentCheckerboardSet)
{
	dim3 threads(1, 1);
	dim3 grid;

	grid.x = 1;
	grid.y = 1;

	if (currentCheckerboardSet == 0) {
		printDataAndMessageValsToPoint<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, widthCheckerboard,
				heightLevel);
	} else {
		printDataAndMessageValsToPoint<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, widthCheckerboard,
				heightLevel);
	}
}

template<>
__host__ void printDataAndMessageValsToPoint<half2>(int xVal, int yVal, half2* dataCostDeviceCurrentLevelCheckerboard1, half2* dataCostDeviceCurrentLevelCheckerboard2,
		half2* messageUDeviceSet0Checkerboard1,
		half2* messageDDeviceSet0Checkerboard1,
		half2* messageLDeviceSet0Checkerboard1,
		half2* messageRDeviceSet0Checkerboard1,
		half2* messageUDeviceSet0Checkerboard2,
		half2* messageDDeviceSet0Checkerboard2,
		half2* messageLDeviceSet0Checkerboard2,
		half2* messageRDeviceSet0Checkerboard2, half2* messageUDeviceSet1Checkerboard1, half2* messageDDeviceSet1Checkerboard1,
		half2* messageLDeviceSet1Checkerboard1, half2* messageRDeviceSet1Checkerboard1,
		half2* messageUDeviceSet1Checkerboard2, half2* messageDDeviceSet1Checkerboard2,
		half2* messageLDeviceSet1Checkerboard2, half2* messageRDeviceSet1Checkerboard2, int widthCheckerboard,
		int heightLevel, int currentCheckerboardSet)
{
	printDataAndMessageValsToPoint<half>(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
			(half*) messageUDeviceSet0Checkerboard1,
			(half*) messageDDeviceSet0Checkerboard1,
			(half*) messageLDeviceSet0Checkerboard1,
			(half*) messageRDeviceSet0Checkerboard1,
			(half*) messageUDeviceSet0Checkerboard2,
			(half*) messageDDeviceSet0Checkerboard2,
			(half*) messageLDeviceSet0Checkerboard2,
			(half*) messageRDeviceSet0Checkerboard2,
			(half*) messageUDeviceSet1Checkerboard1,
			(half*) messageDDeviceSet1Checkerboard1,
			(half*) messageLDeviceSet1Checkerboard1,
			(half*) messageRDeviceSet1Checkerboard1,
			(half*) messageUDeviceSet1Checkerboard2,
			(half*) messageDDeviceSet1Checkerboard2,
			(half*) messageLDeviceSet1Checkerboard2,
			(half*) messageRDeviceSet1Checkerboard2, widthCheckerboard * 2,
			heightLevel, currentCheckerboardSet);
}


template<typename T>
__host__ void printDataAndMessageValsAtPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1,
		T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2,
		T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1,
		T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2,
		T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		int widthCheckerboard, int heightLevel, int currentCheckerboardSet)
{
	dim3 threads(1, 1);
	dim3 grid;

	grid.x = 1;
	grid.y = 1;

	if (currentCheckerboardSet == 0) {
		printDataAndMessageValsAtPoint<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet0Checkerboard1,
				messageDDeviceSet0Checkerboard1,
				messageLDeviceSet0Checkerboard1,
				messageRDeviceSet0Checkerboard1,
				messageUDeviceSet0Checkerboard2,
				messageDDeviceSet0Checkerboard2,
				messageLDeviceSet0Checkerboard2,
				messageRDeviceSet0Checkerboard2, widthCheckerboard,
				heightLevel);
	} else {
		printDataAndMessageValsAtPoint<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messageUDeviceSet1Checkerboard1,
				messageDDeviceSet1Checkerboard1,
				messageLDeviceSet1Checkerboard1,
				messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2,
				messageDDeviceSet1Checkerboard2,
				messageLDeviceSet1Checkerboard2,
				messageRDeviceSet1Checkerboard2, widthCheckerboard,
				heightLevel);
	}
}

template<>
__host__ void printDataAndMessageValsAtPoint<half2>(int xVal, int yVal, half2* dataCostDeviceCurrentLevelCheckerboard1, half2* dataCostDeviceCurrentLevelCheckerboard2,
		half2* messageUDeviceSet0Checkerboard1,
		half2* messageDDeviceSet0Checkerboard1,
		half2* messageLDeviceSet0Checkerboard1,
		half2* messageRDeviceSet0Checkerboard1,
		half2* messageUDeviceSet0Checkerboard2,
		half2* messageDDeviceSet0Checkerboard2,
		half2* messageLDeviceSet0Checkerboard2,
		half2* messageRDeviceSet0Checkerboard2, half2* messageUDeviceSet1Checkerboard1, half2* messageDDeviceSet1Checkerboard1,
		half2* messageLDeviceSet1Checkerboard1, half2* messageRDeviceSet1Checkerboard1,
		half2* messageUDeviceSet1Checkerboard2, half2* messageDDeviceSet1Checkerboard2,
		half2* messageLDeviceSet1Checkerboard2, half2* messageRDeviceSet1Checkerboard2, int widthCheckerboard,
		int heightLevel, int currentCheckerboardSet)
{
	printDataAndMessageValsAtPoint<half>(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
			(half*) messageUDeviceSet0Checkerboard1,
			(half*) messageDDeviceSet0Checkerboard1,
			(half*) messageLDeviceSet0Checkerboard1,
			(half*) messageRDeviceSet0Checkerboard1,
			(half*) messageUDeviceSet0Checkerboard2,
			(half*) messageDDeviceSet0Checkerboard2,
			(half*) messageLDeviceSet0Checkerboard2,
			(half*) messageRDeviceSet0Checkerboard2,
			(half*) messageUDeviceSet1Checkerboard1,
			(half*) messageDDeviceSet1Checkerboard1,
			(half*) messageLDeviceSet1Checkerboard1,
			(half*) messageRDeviceSet1Checkerboard1,
			(half*) messageUDeviceSet1Checkerboard2,
			(half*) messageDDeviceSet1Checkerboard2,
			(half*) messageLDeviceSet1Checkerboard2,
			(half*) messageRDeviceSet1Checkerboard2, widthCheckerboard * 2,
			heightLevel, currentCheckerboardSet);
}


//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
template<typename T>
__host__ void runBeliefPropStereoCUDA(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& resultingDisparityMapDevice, BPsettings& algSettings, DetailedTimings& timings)
{	

#ifdef RUN_DETAILED_TIMING

	timeCopyDataKernelTotalTime = 0.0;
	timeBpItersKernelTotalTime = 0.0;

#endif

	//retrieve the total number of possible movements; this is equal to the number of disparity values 
	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsConstMemStart = std::chrono::system_clock::now();

#endif

	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsConstMemEnd = std::chrono::system_clock::now();

	std::chrono::duration<double> diff = timeInitSettingsConstMemEnd-timeInitSettingsConstMemStart;
	double totalTimeInitSettingsConstMem = diff.count();

#endif

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	float widthLevel = (float)algSettings.widthImages;
	float heightLevel = (float)algSettings.heightImages;

	//store the "actual" integer size of the width and height of the level since it's not actually
	//possible to work with level with a decimal sizes...the portion of the last row/column is truncated
	//if the width/level size has a decimal
	int widthLevelActualIntegerSize = (int)roundf(widthLevel);
	int heightLevelActualIntegerSize = (int)roundf(heightLevel);

	int halfTotalDataAllLevels = 0;

	//compute "half" the total number of pixels in including every level of the "pyramid"
	//using "half" because the data is split in two using the checkerboard scheme
	for (int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
	{
		halfTotalDataAllLevels += (getCheckerboardWidth<T>(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * (totalPossibleMovements);
		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		widthLevelActualIntegerSize = (int)ceil(widthLevel);
		heightLevelActualIntegerSize = (int)ceil(heightLevel);
	}

	//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//each checkboard holds half of the data
	T* dataCostDeviceCheckerboard1; //checkerboard 1 includes the pixel in slot (0, 0)
	T* dataCostDeviceCheckerboard2;

	T* messageUDeviceCheckerboard1;
	T* messageDDeviceCheckerboard1;
	T* messageLDeviceCheckerboard1;
	T* messageRDeviceCheckerboard1;

	T* messageUDeviceCheckerboard2;
	T* messageDDeviceCheckerboard2;
	T* messageLDeviceCheckerboard2;
	T* messageRDeviceCheckerboard2;

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsMallocStart = std::chrono::system_clock::now();

#endif

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	printf("ALLOC ALL MEMORY\n");
	(cudaMalloc((void**) &dataCostDeviceCheckerboard1, 10*(halfTotalDataAllLevels)*sizeof(T)));
	dataCostDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[1*(halfTotalDataAllLevels)]);

	messageUDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[2*(halfTotalDataAllLevels)]);
	messageDDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[3*(halfTotalDataAllLevels)]);
	messageLDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[4*(halfTotalDataAllLevels)]);
	messageRDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[5*(halfTotalDataAllLevels)]);

	messageUDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[6*(halfTotalDataAllLevels)]);
	messageDDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[7*(halfTotalDataAllLevels)]);
	messageLDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[8*(halfTotalDataAllLevels)]);
	messageRDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[9*(halfTotalDataAllLevels)]);

#else

	(cudaMalloc((void**) &dataCostDeviceCheckerboard1, (halfTotalDataAllLevels)*sizeof(T)));
	(cudaMalloc((void**) &dataCostDeviceCheckerboard2, (halfTotalDataAllLevels)*sizeof(T)));

#endif

	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsMallocEnd = std::chrono::system_clock::now();

	diff = timeInitSettingsMallocEnd-timeInitSettingsMallocStart;
	double totalTimeInitSettingsMallocStart = diff.count();

	auto timeInitDataCostsStart = std::chrono::system_clock::now();

#endif

	//now go "back to" the bottom level to initialize the data costs starting at the bottom level and going up the pyramid
	widthLevel = (float)algSettings.widthImages;
	heightLevel = (float)algSettings.heightImages;

	widthLevelActualIntegerSize = (int)roundf(widthLevel);
	heightLevelActualIntegerSize = (int)roundf(heightLevel);

	//initialize the data cost at the bottom level 
	initializeDataCosts<T>(image1PixelsDevice, image2PixelsDevice, dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2, algSettings);

#ifdef RUN_DETAILED_TIMING

	auto timeInitDataCostsEnd = std::chrono::system_clock::now();
	diff = timeInitDataCostsEnd-timeInitDataCostsStart;

	double totalTimeGetDataCostsBottomLevel = diff.count();

#endif

	int offsetLevel = 0;

#ifdef RUN_DETAILED_TIMING

	auto timeInitDataCostsHigherLevelsStart = std::chrono::system_clock::now();

#endif

	//set the data costs at each level from the bottom level "up"
	for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
	{
		int prev_level_offset_level = offsetLevel;

		//width is half since each part of the checkboard contains half the values going across
		//retrieve offset where the data starts at the "current level"
		offsetLevel += (getCheckerboardWidth<T>(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * totalPossibleMovements;

		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
		int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

		widthLevelActualIntegerSize = (int)ceil(widthLevel);
		heightLevelActualIntegerSize = (int)ceil(heightLevel);
		int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSize);

		//printf("LevelNum: %d  Width: %d  Height: %d \n", levelNum, widthLevelActualIntegerSize, heightLevelActualIntegerSize);
		initializeDataCurrentLevel<T>(dataCostDeviceCheckerboard1,
				dataCostDeviceCheckerboard2, prev_level_offset_level,
				offsetLevel, widthLevelActualIntegerSize,
				heightLevelActualIntegerSize, prevWidthLevelActualIntegerSize,
				prevHeightLevelActualIntegerSize);
	}

#ifdef RUN_DETAILED_TIMING

	auto timeInitDataCostsHigherLevelsEnd = std::chrono::system_clock::now();
	diff = timeInitDataCostsHigherLevelsEnd-timeInitDataCostsHigherLevelsStart;

	double totalTimeGetDataCostsHigherLevels = diff.count();

#endif

	( cudaDeviceSynchronize() );

	//declare the space to pass the BP messages
	//need to have two "sets" of checkerboards because
	//the message values at the "higher" level in the image
	//pyramid need copied to a lower level without overwriting
	//values
	T* dataCostDeviceCurrentLevelCheckerboard1;
	T* dataCostDeviceCurrentLevelCheckerboard2;
	T* messageUDeviceSet0Checkerboard1;
	T* messageDDeviceSet0Checkerboard1;
	T* messageLDeviceSet0Checkerboard1;
	T* messageRDeviceSet0Checkerboard1;

	T* messageUDeviceSet0Checkerboard2;
	T* messageDDeviceSet0Checkerboard2;
	T* messageLDeviceSet0Checkerboard2;
	T* messageRDeviceSet0Checkerboard2;

	T* messageUDeviceSet1Checkerboard1;
	T* messageDDeviceSet1Checkerboard1;
	T* messageLDeviceSet1Checkerboard1;
	T* messageRDeviceSet1Checkerboard1;

	T* messageUDeviceSet1Checkerboard2;
	T* messageDDeviceSet1Checkerboard2;
	T* messageLDeviceSet1Checkerboard2;
	T* messageRDeviceSet1Checkerboard2;

#ifdef RUN_DETAILED_TIMING

	auto timeInitMessageValuesStart = std::chrono::system_clock::now();

#endif

	dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
	dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
	messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
	messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
	messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

	messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
	messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
	messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
	messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#else

	//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
	int numBytesDataAndMessageSetInCheckerboardAtLevel = (getCheckerboardWidth<T>(widthLevelActualIntegerSize)) * heightLevelActualIntegerSize * totalPossibleMovements * sizeof(T);

	//allocate the space for the message values in the first checkboard set at the current level
	(cudaMalloc((void**) &messageUDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));

	(cudaMalloc((void**) &messageUDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));

#endif

	auto timeInitMessageValuesKernelTimeStart = std::chrono::system_clock::now();

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault<T>(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
											messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
											widthLevelActualIntegerSize, heightLevelActualIntegerSize, totalPossibleMovements);

	gpuErrchk( cudaPeekAtLastError() );

	auto timeInitMessageValuesKernelTimeEnd = std::chrono::system_clock::now();
	diff = timeInitMessageValuesKernelTimeEnd-timeInitMessageValuesKernelTimeStart;

	double totalTimeInitMessageValuesKernelTime = diff.count();

	( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

#ifdef RUN_DETAILED_TIMING

	auto timeInitMessageValuesEnd = std::chrono::system_clock::now();
	diff = timeInitMessageValuesEnd-timeInitMessageValuesStart;

	double totalTimeInitMessageVals = diff.count();

#endif

	//alternate between checkerboard sets 0 and 1
	int currentCheckerboardSet = 0;

#ifdef RUN_DETAILED_TIMING

	double totalTimeBpIters = 0.0;
	double totalTimeCopyData = 0.0;

#endif


	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = algSettings.numLevels - 1; levelNum >= 0; levelNum--)
	{
		printf("LevelNum: %d\n", levelNum);
		printf("currentCheckerboardSet: %d\n", currentCheckerboardSet);
		gpuErrchk( cudaPeekAtLastError() );

		( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

		auto timeBpIterStart = std::chrono::system_clock::now();

#endif


		//printf("LevelNumBP: %d  Width: %f  Height: %f \n", levelNum, widthLevel, heightLevel);

		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		if (currentCheckerboardSet == 0)
		{
			runBPAtCurrentLevel<T>(algSettings.numIterations,
					widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet0Checkerboard1,
					messageDDeviceSet0Checkerboard1,
					messageLDeviceSet0Checkerboard1,
					messageRDeviceSet0Checkerboard1,
					messageUDeviceSet0Checkerboard2,
					messageDDeviceSet0Checkerboard2,
					messageLDeviceSet0Checkerboard2,
					messageRDeviceSet0Checkerboard2,
					dataCostDeviceCurrentLevelCheckerboard1,
					dataCostDeviceCurrentLevelCheckerboard2);
		}
		else
		{
			runBPAtCurrentLevel<T>(algSettings.numIterations,
					widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet1Checkerboard1,
					messageDDeviceSet1Checkerboard1,
					messageLDeviceSet1Checkerboard1,
					messageRDeviceSet1Checkerboard1,
					messageUDeviceSet1Checkerboard2,
					messageDDeviceSet1Checkerboard2,
					messageLDeviceSet1Checkerboard2,
					messageRDeviceSet1Checkerboard2,
					dataCostDeviceCurrentLevelCheckerboard1,
					dataCostDeviceCurrentLevelCheckerboard2);
		}

		(cudaDeviceSynchronize());

#ifdef RUN_DETAILED_TIMING

		auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd-timeBpIterStart;

		totalTimeBpIters += diff.count();
		
		auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

#endif

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level 
		if (levelNum > 0)
		{	
			int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
			int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

			//the "next level" down has double the width and height of the current level
			widthLevel *= 2.0f;
			heightLevel *= 2.0f;

			widthLevelActualIntegerSize = (int)ceil(widthLevel);
			heightLevelActualIntegerSize = (int)ceil(heightLevel);
			int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSize);

			offsetLevel -= widthCheckerboard * heightLevelActualIntegerSize * totalPossibleMovements;
			printf("OffsetLevel: %d\n", offsetLevel);

			dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
			dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

			//bind messages in the current checkerboard set to the texture to copy to the "other" checkerboard set at the next level 
			if (currentCheckerboardSet == 0)
			{

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				messageUDeviceSet1Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
				messageDDeviceSet1Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
				messageLDeviceSet1Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
				messageRDeviceSet1Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

				messageUDeviceSet1Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
				messageDDeviceSet1Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
				messageLDeviceSet1Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
				messageRDeviceSet1Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				copyMessageValuesToNextLevelDown<T>(
						prevWidthLevelActualIntegerSize,
						prevHeightLevelActualIntegerSize,
						widthLevelActualIntegerSize,
						heightLevelActualIntegerSize,
						messageUDeviceSet0Checkerboard1,
						messageDDeviceSet0Checkerboard1,
						messageLDeviceSet0Checkerboard1,
						messageRDeviceSet0Checkerboard1,
						messageUDeviceSet0Checkerboard2,
						messageDDeviceSet0Checkerboard2,
						messageLDeviceSet0Checkerboard2,
						messageRDeviceSet0Checkerboard2,
						(T**)&messageUDeviceSet1Checkerboard1,
						(T**)&messageDDeviceSet1Checkerboard1,
						(T**)&messageLDeviceSet1Checkerboard1,
						(T**)&messageRDeviceSet1Checkerboard1,
						(T**)&messageUDeviceSet1Checkerboard2,
						(T**)&messageDDeviceSet1Checkerboard2,
						(T**)&messageLDeviceSet1Checkerboard2,
						(T**)&messageRDeviceSet1Checkerboard2);

				currentCheckerboardSet = 1;
			}
			else
			{

#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

				messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
				messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
				messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
				messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

				messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
				messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
				messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
				messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				copyMessageValuesToNextLevelDown<T>(
						prevWidthLevelActualIntegerSize,
						prevHeightLevelActualIntegerSize,
						widthLevelActualIntegerSize,
						heightLevelActualIntegerSize,
						messageUDeviceSet1Checkerboard1,
						messageDDeviceSet1Checkerboard1,
						messageLDeviceSet1Checkerboard1,
						messageRDeviceSet1Checkerboard1,
						messageUDeviceSet1Checkerboard2,
						messageDDeviceSet1Checkerboard2,
						messageLDeviceSet1Checkerboard2,
						messageRDeviceSet1Checkerboard2,
						(T**)&messageUDeviceSet0Checkerboard1,
						(T**)&messageDDeviceSet0Checkerboard1,
						(T**)&messageLDeviceSet0Checkerboard1,
						(T**)&messageRDeviceSet0Checkerboard1,
						(T**)&messageUDeviceSet0Checkerboard2,
						(T**)&messageDDeviceSet0Checkerboard2,
						(T**)&messageLDeviceSet0Checkerboard2,
						(T**)&messageRDeviceSet0Checkerboard2);

				currentCheckerboardSet = 0;
			}
		}
		gpuErrchk( cudaPeekAtLastError() );

		//otherwise in "bottom level"; use message values and data costs to retrieve final movement values
		( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

		auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd-timeCopyMessageValuesStart;

		totalTimeCopyData += diff.count();

#endif
	}
	gpuErrchk( cudaPeekAtLastError() );

	//printf("Final  Width: %d  Height: %d \n", widthLevelActualIntegerSize, heightLevelActualIntegerSize);

#ifdef RUN_DETAILED_TIMING

	auto timeGetOutputDisparityStart = std::chrono::system_clock::now();
	gpuErrchk( cudaPeekAtLastError() );

#endif


	retrieveOutputDisparity<T>(dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
			messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
			messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
			messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, messageRDeviceSet1Checkerboard1,
			messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2,
			resultingDisparityMapDevice, widthLevel, heightLevel, currentCheckerboardSet);

	gpuErrchk( cudaPeekAtLastError() );

#ifdef RUN_DETAILED_TIMING

	auto timeGetOutputDisparityEnd = std::chrono::system_clock::now();
	diff = timeGetOutputDisparityEnd-timeGetOutputDisparityStart;

	double totalTimeGetOutputDisparity = diff.count();

	auto timeFinalUnbindFreeStart = std::chrono::system_clock::now();
	double totalTimeFinalUnbind = 0.0;

#endif

#ifdef RUN_DETAILED_TIMING

	auto timeFinalFreeStart = std::chrono::system_clock::now();

#endif
	gpuErrchk( cudaPeekAtLastError() );

#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

	printf("ALLOC MULT MEM SEGMENTS\n");

	//free the device storage for the message values used to retrieve the output movement values
	if (currentCheckerboardSet == 0)
	{
		//free device space allocated to message values
		cudaFree(messageUDeviceSet0Checkerboard1);
		cudaFree(messageDDeviceSet0Checkerboard1);
		cudaFree(messageLDeviceSet0Checkerboard1);
		cudaFree(messageRDeviceSet0Checkerboard1);

		cudaFree(messageUDeviceSet0Checkerboard2);
		cudaFree(messageDDeviceSet0Checkerboard2);
		cudaFree(messageLDeviceSet0Checkerboard2);
		cudaFree(messageRDeviceSet0Checkerboard2);
	}
	else
	{
		//free device space allocated to message values
		cudaFree(messageUDeviceSet1Checkerboard1);
		cudaFree(messageDDeviceSet1Checkerboard1);
		cudaFree(messageLDeviceSet1Checkerboard1);
		cudaFree(messageRDeviceSet1Checkerboard1);

		cudaFree(messageUDeviceSet1Checkerboard2);
		cudaFree(messageDDeviceSet1Checkerboard2);
		cudaFree(messageLDeviceSet1Checkerboard2);
		cudaFree(messageRDeviceSet1Checkerboard2);
	}

	//now free the allocated data space
	cudaFree(dataCostDeviceCheckerboard1);
	cudaFree(dataCostDeviceCheckerboard2);

#else

	printf("FREE ALL MEMORY\n");
	cudaFree(dataCostDeviceCheckerboard1);

#endif

	gpuErrchk( cudaPeekAtLastError() );
	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeFinalUnbindFreeEnd = std::chrono::system_clock::now();
	auto timeFinalFreeEnd = std::chrono::system_clock::now();

	diff = timeFinalUnbindFreeEnd-timeFinalUnbindFreeStart;
	double totalTimeFinalUnbindFree = diff.count();

	diff = timeFinalFreeEnd-timeFinalFreeStart;
	double totalTimeFinalFree = diff.count();

	double totalMemoryProcessingTime = totalTimeInitSettingsConstMem
			+ totalTimeInitSettingsMallocStart + totalTimeFinalUnbindFree
			+ (totalTimeInitMessageVals - totalTimeInitMessageValuesKernelTime)
			+ (totalTimeCopyData - timeCopyDataKernelTotalTime)
			+ (totalTimeBpIters - timeBpItersKernelTotalTime);
	double totalComputationProcessing = totalTimeGetDataCostsBottomLevel
			+ totalTimeGetDataCostsHigherLevels
			+ totalTimeInitMessageValuesKernelTime + timeCopyDataKernelTotalTime
			+ timeBpItersKernelTotalTime + totalTimeGetOutputDisparity;
	double totalTimed = totalTimeInitSettingsConstMem
			+ totalTimeInitSettingsMallocStart
			+ totalTimeGetDataCostsBottomLevel
			+ totalTimeGetDataCostsHigherLevels + totalTimeInitMessageVals
			+ totalTimeBpIters + totalTimeCopyData + totalTimeGetOutputDisparity
			+ totalTimeFinalUnbindFree;
	timings.totalTimeInitSettingsConstMem.push_back(
			totalTimeInitSettingsConstMem);
	timings.totalTimeInitSettingsMallocStart.push_back(
			totalTimeInitSettingsMallocStart);
	timings.totalTimeGetDataCostsBottomLevel.push_back(
			totalTimeGetDataCostsBottomLevel);
	timings.totalTimeGetDataCostsHigherLevels.push_back(
			totalTimeGetDataCostsHigherLevels);
	timings.totalTimeInitMessageVals.push_back(totalTimeInitMessageVals);
	timings.totalTimeInitMessageValuesKernelTime.push_back(totalTimeInitMessageValuesKernelTime);
	timings.totalTimeBpIters.push_back(totalTimeBpIters);
	timings.timeBpItersKernelTotalTime.push_back(timeBpItersKernelTotalTime);
	timings.totalTimeCopyData.push_back(totalTimeCopyData);
	timings.timeCopyDataKernelTotalTime.push_back(timeCopyDataKernelTotalTime);
	timings.totalTimeGetOutputDisparity.push_back(totalTimeGetOutputDisparity);
	timings.totalTimeFinalUnbindFree.push_back(totalTimeFinalUnbindFree);
	timings.totalTimeFinalUnbind.push_back(totalTimeFinalUnbind);
	timings.totalTimeFinalFree.push_back(totalTimeFinalFree);
	timings.totalTimed.push_back(totalTimed);
	timings.totalMemoryProcessingTime.push_back(totalMemoryProcessingTime);
	timings.totalComputationProcessing.push_back(totalComputationProcessing);
	timings.totNumTimings++;

#endif
}
