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

#include "ProcessCUDABP.h"
#include "kernalBpStereo.cu"

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

#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

template<>
int ProcessCUDABP<half2>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
			return (int)ceil(((ceil(((float)widthLevelActualIntegerSize) / 2.0)) / 2.0));
}

template<>
int ProcessCUDABP<half>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
		return (getCheckerboardWidth<half2>(widthLevelActualIntegerSize)) * 2;
}

#endif

template<typename T>
void ProcessCUDABP<T>::printDataAndMessageValsAtPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
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
		printDataAndMessageValsAtPointKernel<T> <<<grid, threads>>>(xVal, yVal,
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
		printDataAndMessageValsAtPointKernel<T> <<<grid, threads>>>(xVal, yVal,
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
void ProcessCUDABP<half2>::printDataAndMessageValsAtPoint(int xVal, int yVal, half2* dataCostDeviceCurrentLevelCheckerboard1, half2* dataCostDeviceCurrentLevelCheckerboard2,
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
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.printDataAndMessageValsAtPoint(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
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
void ProcessCUDABP<T>::printDataAndMessageValsToPoint(int xVal, int yVal, T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
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
		printDataAndMessageValsToPointKernel<T> <<<grid, threads>>>(xVal, yVal,
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
		printDataAndMessageValsToPointKernel<T> <<<grid, threads>>>(xVal, yVal,
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
void ProcessCUDABP<half2>::printDataAndMessageValsToPoint(int xVal, int yVal, half2* dataCostDeviceCurrentLevelCheckerboard1, half2* dataCostDeviceCurrentLevelCheckerboard2,
		half2* messageUDeviceSet0Checkerboard1, half2* messageDDeviceSet0Checkerboard1,
		half2* messageLDeviceSet0Checkerboard1, half2* messageRDeviceSet0Checkerboard1,
		half2* messageUDeviceSet0Checkerboard2, half2* messageDDeviceSet0Checkerboard2,
		half2* messageLDeviceSet0Checkerboard2, half2* messageRDeviceSet0Checkerboard2,
		half2* messageUDeviceSet1Checkerboard1, half2* messageDDeviceSet1Checkerboard1,
		half2* messageLDeviceSet1Checkerboard1, half2* messageRDeviceSet1Checkerboard1,
		half2* messageUDeviceSet1Checkerboard2, half2* messageDDeviceSet1Checkerboard2,
		half2* messageLDeviceSet1Checkerboard2, half2* messageRDeviceSet1Checkerboard2,
		int widthCheckerboard, int heightLevel, int currentCheckerboardSet)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.printDataAndMessageValsToPoint(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
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

//functions directed related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
void ProcessCUDABP<T>::runBPAtCurrentLevel(BPsettings& algSettings, int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
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
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
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
				checkboardPartUpdate, algSettings.disc_k_bp);

		(cudaDeviceSynchronize());

#ifdef RUN_DETAILED_TIMING

		auto timeBpItersKernelEnd = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = timeBpItersKernelEnd
				- timeBpItersKernelStart;

		timeBpItersKernelTotalTime += diff.count();

#endif

	}

	gpuErrchk( cudaPeekAtLastError() );
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
void ProcessCUDABP<T>::copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
	int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
	T* messageUDeviceCheckerboard1CopyFrom, T* messageDDeviceCheckerboard1CopyFrom, T* messageLDeviceCheckerboard1CopyFrom,
	T* messageRDeviceCheckerboard1CopyFrom, T* messageUDeviceCheckerboard2CopyFrom, T* messageDDeviceCheckerboard2CopyFrom,
	T* messageLDeviceCheckerboard2CopyFrom, T* messageRDeviceCheckerboard2CopyFrom, T* messageUDeviceCheckerboard1CopyTo,
	T* messageDDeviceCheckerboard1CopyTo, T* messageLDeviceCheckerboard1CopyTo, T* messageRDeviceCheckerboard1CopyTo,
	T* messageUDeviceCheckerboard2CopyTo, T* messageDDeviceCheckerboard2CopyTo, T* messageLDeviceCheckerboard2CopyTo,
	T* messageRDeviceCheckerboard2CopyTo)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	int widthCheckerboard = getCheckerboardWidth<T>(widthLevelActualIntegerSizeNextLevel);
	grid.x = (unsigned int)ceil((float)(widthCheckerboard / 2.0f) / (float)threads.x);
	grid.y = (unsigned int)ceil((float)(heightLevelActualIntegerSizeNextLevel / 2.0f) / (float)threads.y);

	( cudaDeviceSynchronize() );

#ifdef RUN_DETAILED_TIMING

	auto timeCopyDataKernelStart = std::chrono::system_clock::now();

#endif
	gpuErrchk( cudaPeekAtLastError() );

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures<T> <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo,
			messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo,
			messageRDeviceCheckerboard2CopyTo, getCheckerboardWidth<T>(widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			getCheckerboardWidth<T>(widthLevelActualIntegerSizeNextLevel), heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_1);

	( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures<T> <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
			messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
			messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
			messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo,
			messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo,
			messageRDeviceCheckerboard2CopyTo, getCheckerboardWidth<T>(widthLevelActualIntegerSizePrevLevel), (heightLevelActualIntegerSizePrevLevel),
			getCheckerboardWidth<T>(widthLevelActualIntegerSizeNextLevel), heightLevelActualIntegerSizeNextLevel, CHECKERBOARD_PART_2);

	( cudaDeviceSynchronize() );

	gpuErrchk( cudaPeekAtLastError() );

#ifdef RUN_DETAILED_TIMING

	auto timeCopyDataKernelEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = timeCopyDataKernelEnd-timeCopyDataKernelStart;

	timeCopyDataKernelTotalTime += diff.count();

#endif
}

//due to the checkerboard indexing, half2 must be converted to half with the half function used for copying to the next level
template<>
void ProcessCUDABP<half2>::copyMessageValuesToNextLevelDown(int widthLevelActualIntegerSizePrevLevel, int heightLevelActualIntegerSizePrevLevel,
	int widthLevelActualIntegerSizeNextLevel, int heightLevelActualIntegerSizeNextLevel,
	half2* messageUDeviceCheckerboard1CopyFrom, half2* messageDDeviceCheckerboard1CopyFrom, half2* messageLDeviceCheckerboard1CopyFrom,
	half2* messageRDeviceCheckerboard1CopyFrom, half2* messageUDeviceCheckerboard2CopyFrom, half2* messageDDeviceCheckerboard2CopyFrom,
	half2* messageLDeviceCheckerboard2CopyFrom, half2* messageRDeviceCheckerboard2CopyFrom, half2* messageUDeviceCheckerboard1CopyTo,
	half2* messageDDeviceCheckerboard1CopyTo, half2* messageLDeviceCheckerboard1CopyTo, half2* messageRDeviceCheckerboard1CopyTo,
	half2* messageUDeviceCheckerboard2CopyTo, half2* messageDDeviceCheckerboard2CopyTo, half2* messageLDeviceCheckerboard2CopyTo,
	half2* messageRDeviceCheckerboard2CopyTo)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.copyMessageValuesToNextLevelDown(widthLevelActualIntegerSizePrevLevel, heightLevelActualIntegerSizePrevLevel,
		widthLevelActualIntegerSizeNextLevel, heightLevelActualIntegerSizeNextLevel,
		(half*)messageUDeviceCheckerboard1CopyFrom, (half*)messageDDeviceCheckerboard1CopyFrom, (half*)messageLDeviceCheckerboard1CopyFrom,
		(half*)messageRDeviceCheckerboard1CopyFrom, (half*)messageUDeviceCheckerboard2CopyFrom, (half*)messageDDeviceCheckerboard2CopyFrom,
		(half*)messageLDeviceCheckerboard2CopyFrom, (half*)messageRDeviceCheckerboard2CopyFrom, (half*)messageUDeviceCheckerboard1CopyTo,
		(half*)messageDDeviceCheckerboard1CopyTo, (half*)messageLDeviceCheckerboard1CopyTo, (half*)messageRDeviceCheckerboard1CopyTo,
		(half*)messageUDeviceCheckerboard2CopyTo, (half*)messageDDeviceCheckerboard2CopyTo, (half*)messageLDeviceCheckerboard2CopyTo,
		(half*)messageRDeviceCheckerboard2CopyTo);
}


//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T>
void ProcessCUDABP<T>::initializeDataCosts(float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceCheckerboard1, T* dataCostDeviceCheckerboard2, BPsettings& algSettings)
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
			algSettings.heightImages, algSettings.lambda_bp, algSettings.data_k_bp);

	( cudaDeviceSynchronize() );
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T>
void ProcessCUDABP<T>::initializeMessageValsToDefault(T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
												  T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
												  int widthLevel, int heightLevel, int numPossibleMovements)
{
	int widthOfCheckerboard = getCheckerboardWidth<T>(widthLevel);

	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightLevel / (float)threads.y));

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	initializeMessageValsToDefaultKernel<T> <<< grid, threads >>> (messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1,
												messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
												messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, widthOfCheckerboard, heightLevel);

	cudaDeviceSynchronize();
}


template<typename T>
void ProcessCUDABP<T>::initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteToCheckerboard1,
		T* dataCostDeviceToWriteToCheckerboard2,
		int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
		int prevWidthLevelActualIntegerSize,
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

	gpuErrchk( cudaPeekAtLastError() );

	size_t offsetNum = 0;

	initializeCurrentLevelDataStereoNoTextures<T> <<<grid, threads>>>(
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard1,
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_1, ((int) offsetNum / sizeof(float)));

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	initializeCurrentLevelDataStereoNoTextures<T> <<<grid, threads>>>(
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard2,
			widthLevelActualIntegerSize, heightLevelActualIntegerSize,
			prevWidthLevelActualIntegerSize, prevHeightLevelActualIntegerSize,
			CHECKERBOARD_PART_2, ((int) offsetNum / sizeof(float)));

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
}

//due to indexing, need to convert to half* and use half arrays for this function
template<>
void ProcessCUDABP<half2>::initializeDataCurrentLevel(half2* dataCostStereoCheckerboard1,
		half2* dataCostStereoCheckerboard2, half2* dataCostDeviceToWriteToCheckerboard1,
		half2* dataCostDeviceToWriteToCheckerboard2, int widthLevelActualIntegerSize,
		int heightLevelActualIntegerSize, int prevWidthLevelActualIntegerSize,
		int prevHeightLevelActualIntegerSize)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.initializeDataCurrentLevel((half*)dataCostStereoCheckerboard1,
			(half*)dataCostStereoCheckerboard2, (half*)dataCostDeviceToWriteToCheckerboard1,
			(half*)dataCostDeviceToWriteToCheckerboard2, widthLevelActualIntegerSize,
			heightLevelActualIntegerSize, prevWidthLevelActualIntegerSize,
			prevHeightLevelActualIntegerSize);
}

template<typename T>
void ProcessCUDABP<T>::retrieveOutputDisparity(T* dataCostDeviceCurrentLevelCheckerboard1, T* dataCostDeviceCurrentLevelCheckerboard2,
		T* messageUDeviceSet0Checkerboard1, T* messageDDeviceSet0Checkerboard1, T* messageLDeviceSet0Checkerboard1, T* messageRDeviceSet0Checkerboard1,
		T* messageUDeviceSet0Checkerboard2, T* messageDDeviceSet0Checkerboard2, T* messageLDeviceSet0Checkerboard2, T* messageRDeviceSet0Checkerboard2,
		T* messageUDeviceSet1Checkerboard1, T* messageDDeviceSet1Checkerboard1, T* messageLDeviceSet1Checkerboard1, T* messageRDeviceSet1Checkerboard1,
		T* messageUDeviceSet1Checkerboard2, T* messageDDeviceSet1Checkerboard2, T* messageLDeviceSet1Checkerboard2, T* messageRDeviceSet1Checkerboard2,
		float* resultingDisparityMapDevice, int widthLevel, int heightLevel, int currentCheckerboardSet)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	int widthCheckerboard = getCheckerboardWidth<T>(widthLevel);
	grid.x = (unsigned int) ceil((float) widthCheckerboard / (float) threads.x);
	grid.y = (unsigned int) ceil((float) heightLevel / (float) threads.y);

	if (currentCheckerboardSet == 0)
	{
		retrieveOutputDisparityCheckerboardStereoOptimized<T> <<<grid, threads>>>(
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
		retrieveOutputDisparityCheckerboardStereoOptimized<T> <<<grid, threads>>>(
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

	/*if (currentCheckerboardSet == 0)
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
		}*/

	(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class ProcessCUDABP<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class ProcessCUDABP<double>;

#elif ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

template class ProcessCUDABP<float>;
template class ProcessCUDABP<half>;
template class ProcessCUDABP<half2>;

#endif
