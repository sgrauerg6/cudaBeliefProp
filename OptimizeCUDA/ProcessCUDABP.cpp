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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* May be needed if using half2
#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

template<>
int ProcessCUDABP<half2>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
			return (int)ceil(((ceil(((float)widthLevelActualIntegerSize) / 2.0)) / 2.0));
}

template<>
int ProcessCUDABP<half>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
	ProcessCUDABP<half2> processCUDABPHalf;
	return processCUDABPHalf.getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize) * 2;
}

#endif
*/


template<typename T>
void ProcessCUDABP<T>::printDataAndMessageValsAtPoint(int xVal, int yVal,
		const levelProperties& currentLevelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
		const Checkerboard_Parts currentCheckerboardSet)
{
	dim3 threads(1, 1);
	dim3 grid;

	grid.x = 1;
	grid.y = 1;

	if (currentCheckerboardSet == 0) {
		printDataAndMessageValsAtPointKernel<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel,
				currentLevelProperties.heightLevel);
	} else {
		printDataAndMessageValsAtPointKernel<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel,
				currentLevelProperties.heightLevel);
	}
}

template<typename T>
void ProcessCUDABP<T>::printDataAndMessageValsToPoint(int xVal, int yVal,
		const levelProperties& currentLevelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
		const Checkerboard_Parts currentCheckerboardSet)
{
	dim3 threads(1, 1);
	dim3 grid;

	grid.x = 1;
	grid.y = 1;

	if (currentCheckerboardSet == 0) {
		printDataAndMessageValsToPointKernel<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel,
				currentLevelProperties.heightLevel);
	} else {
		printDataAndMessageValsToPointKernel<T> <<<grid, threads>>>(xVal, yVal,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel,
				currentLevelProperties.heightLevel);
	}
}

//functions directed related to running BP to retrieve the movement between the images

//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T>
void ProcessCUDABP<T>::runBPAtCurrentLevel(const BPsettings& algSettings,
		const levelProperties& currentLevelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<T>& messagesDeviceCheckerboard0,
		const checkerboardMessages<T>& messagesDeviceCheckerboard1)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int) ceil(
			(float) (currentLevelProperties.widthCheckerboardLevel) / (float) threads.x); //only updating half at a time
	grid.y = (unsigned int) ceil((float) currentLevelProperties.heightLevel / (float) threads.y);

	//in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
	//still is a check if start of row is aligned
	bool dataAligned = MemoryAlignedAtDataStart(0, 1);

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		Checkerboard_Parts checkboardPartUpdate = CHECKERBOARD_PART_1;

		if ((iterationNum % 2) == 0)
		{
			checkboardPartUpdate = CHECKERBOARD_PART_1;
		}
		else
		{
			checkboardPartUpdate = CHECKERBOARD_PART_0;
		}

		(cudaDeviceSynchronize());

#if (((USE_SHARED_MEMORY == 3) || (USE_SHARED_MEMORY == 4))  && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
		int numDataSharedMemory = BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
					* (DISP_INDEX_START_REG_LOCAL_MEM);
		int numBytesSharedMemory = numDataSharedMemory * sizeof(T);

#if (USE_SHARED_MEMORY == 4)

		numBytesSharedMemory *= 5;

#endif //(USE_SHARED_MEMORY == 4)

		int maxbytes = numBytesSharedMemory; // 96 KB
		cudaFuncSetAttribute(runBPIterationUsingCheckerboardUpdates<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

		//std::cout << "numDataSharedMemory: " << numDataSharedMemory << std::endl;
		runBPIterationUsingCheckerboardUpdates<T><<<grid, threads, maxbytes>>>(checkboardPartUpdate, currentLevelProperties,
						dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
						messagesDeviceCheckerboard0.messagesU, messagesDeviceCheckerboard0.messagesD,
						messagesDeviceCheckerboard0.messagesL, messagesDeviceCheckerboard0.messagesR,
						messagesDeviceCheckerboard1.messagesU, messagesDeviceCheckerboard1.messagesD,
						messagesDeviceCheckerboard1.messagesL, messagesDeviceCheckerboard1.messagesR,
						algSettings.disc_k_bp, dataAligned);

#else
		runBPIterationUsingCheckerboardUpdates<T><<<grid, threads>>>(checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceCheckerboard0.messagesU, messagesDeviceCheckerboard0.messagesD,
				messagesDeviceCheckerboard0.messagesL, messagesDeviceCheckerboard0.messagesR,
				messagesDeviceCheckerboard1.messagesU, messagesDeviceCheckerboard1.messagesD,
				messagesDeviceCheckerboard1.messagesL, messagesDeviceCheckerboard1.messagesR,
				algSettings.disc_k_bp, dataAligned);
#endif

		(cudaDeviceSynchronize());
		gpuErrchk( cudaPeekAtLastError() );

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T>
void ProcessCUDABP<T>::copyMessageValuesToNextLevelDown(
		const levelProperties& currentLevelProperties,
		const levelProperties& nextlevelProperties,
		const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyFrom,
		const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyFrom,
		const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyTo,
		const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyTo)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel) / (float)threads.x);
	grid.y = (unsigned int)ceil((float)(currentLevelProperties.heightLevel) / (float)threads.y);

	( cudaDeviceSynchronize() );

	gpuErrchk( cudaPeekAtLastError() );

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	copyPrevLevelToNextLevelBPCheckerboardStereo<T> <<< grid, threads >>> (CHECKERBOARD_PART_0, currentLevelProperties, nextlevelProperties, messagesDeviceCheckerboard0CopyFrom.messagesU, messagesDeviceCheckerboard0CopyFrom.messagesD,
		messagesDeviceCheckerboard0CopyFrom.messagesL, messagesDeviceCheckerboard0CopyFrom.messagesR, messagesDeviceCheckerboard1CopyFrom.messagesU, messagesDeviceCheckerboard1CopyFrom.messagesD,
		messagesDeviceCheckerboard1CopyFrom.messagesL, messagesDeviceCheckerboard1CopyFrom.messagesR, messagesDeviceCheckerboard0CopyTo.messagesU, messagesDeviceCheckerboard0CopyTo.messagesD,
		messagesDeviceCheckerboard0CopyTo.messagesL, messagesDeviceCheckerboard0CopyTo.messagesR, messagesDeviceCheckerboard1CopyTo.messagesU, messagesDeviceCheckerboard1CopyTo.messagesD,
		messagesDeviceCheckerboard1CopyTo.messagesL, messagesDeviceCheckerboard1CopyTo.messagesR);

(cudaDeviceSynchronize());
gpuErrchk(cudaPeekAtLastError());

	copyPrevLevelToNextLevelBPCheckerboardStereo<T> <<< grid, threads >>> (CHECKERBOARD_PART_1, currentLevelProperties, nextlevelProperties, messagesDeviceCheckerboard0CopyFrom.messagesU, messagesDeviceCheckerboard0CopyFrom.messagesD,
			messagesDeviceCheckerboard0CopyFrom.messagesL, messagesDeviceCheckerboard0CopyFrom.messagesR, messagesDeviceCheckerboard1CopyFrom.messagesU, messagesDeviceCheckerboard1CopyFrom.messagesD,
			messagesDeviceCheckerboard1CopyFrom.messagesL, messagesDeviceCheckerboard1CopyFrom.messagesR, messagesDeviceCheckerboard0CopyTo.messagesU, messagesDeviceCheckerboard0CopyTo.messagesD,
			messagesDeviceCheckerboard0CopyTo.messagesL, messagesDeviceCheckerboard0CopyTo.messagesR, messagesDeviceCheckerboard1CopyTo.messagesU, messagesDeviceCheckerboard1CopyTo.messagesD,
			messagesDeviceCheckerboard1CopyTo.messagesL, messagesDeviceCheckerboard1CopyTo.messagesR);

	( cudaDeviceSynchronize() );

	gpuErrchk( cudaPeekAtLastError() );
}




//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T>
void ProcessCUDABP<T>::initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties, float* image1PixelsCompDevice,
		float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
		T* dataCostDeviceCheckerboard2)
{
	//since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//setup execution parameters
	//the thread size remains constant throughout but the grid size is adjusted based on the current level/kernal to run
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//kernal run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
	grid.x = (unsigned int)ceil((float)currentLevelProperties.widthLevel / (float)threads.x);
	grid.y = (unsigned int)ceil((float)currentLevelProperties.heightLevel / (float)threads.y);

	//initialize the data the the "bottom" of the image pyramid
	initializeBottomLevelDataStereo<T><<<grid, threads>>>(currentLevelProperties, image1PixelsCompDevice,
			image2PixelsCompDevice, dataCostDeviceCheckerboard1,
			dataCostDeviceCheckerboard2, algSettings.lambda_bp, algSettings.data_k_bp);

	( cudaDeviceSynchronize() );
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T>
void ProcessCUDABP<T>::initializeMessageValsToDefault(
		const levelProperties& currentLevelProperties,
		const checkerboardMessages<T>& messagesDeviceCheckerboard0,
		const checkerboardMessages<T>& messagesDeviceCheckerboard1)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel / (float)threads.x), (unsigned int)ceil((float)currentLevelProperties.heightLevel / (float)threads.y));

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	initializeMessageValsToDefaultKernel<T> <<< grid, threads >>> (currentLevelProperties, messagesDeviceCheckerboard0.messagesU, messagesDeviceCheckerboard0.messagesD, messagesDeviceCheckerboard0.messagesL,
			messagesDeviceCheckerboard0.messagesR, messagesDeviceCheckerboard1.messagesU, messagesDeviceCheckerboard1.messagesD, messagesDeviceCheckerboard1.messagesL,
			messagesDeviceCheckerboard1.messagesR);

	cudaDeviceSynchronize();
}


template<typename T>
void ProcessCUDABP<T>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2,
		T* dataCostDeviceToWriteToCheckerboard1,
		T* dataCostDeviceToWriteToCheckerboard2)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
	//the four-connected neighbors are in the other checkerboard
	grid.x = (unsigned int) ceil(
			((float) currentLevelProperties.widthCheckerboardLevel) / (float) threads.x);
	grid.y = (unsigned int) ceil(
			(float) currentLevelProperties.heightLevel / (float) threads.y);

	gpuErrchk( cudaPeekAtLastError() );

	size_t offsetNum = 0;

	initializeCurrentLevelDataStereo<T> <<<grid, threads>>>(CHECKERBOARD_PART_0,
			currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard1,
			((int) offsetNum / sizeof(float)));

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );

	initializeCurrentLevelDataStereo<T> <<<grid, threads>>>(CHECKERBOARD_PART_1,
			currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2,
			dataCostDeviceToWriteToCheckerboard2,
			((int) offsetNum / sizeof(float)));

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO)

template<>
void ProcessCUDABP<half2>::printDataAndMessageValsAtPoint(int xVal, int yVal,
		const levelProperties& levelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<half2>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<half2>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<half2>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<half2>& messagesDeviceSet1Checkerboard1,
		const Checkerboard_Parts currentCheckerboardSet)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.printDataAndMessageValsAtPoint(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
			(half*) messagesDeviceSet0Checkerboard0.messagesU,
			(half*) messagesDeviceSet0Checkerboard0.messagesD,
			(half*) messagesDeviceSet0Checkerboard0.messagesL,
			(half*) messagesDeviceSet0Checkerboard0.messagesR,
			(half*) messagesDeviceSet0Checkerboard1.messagesU,
			(half*) messagesDeviceSet0Checkerboard1.messagesD,
			(half*) messagesDeviceSet0Checkerboard1.messagesL,
			(half*) messagesDeviceSet0Checkerboard1.messagesR,
			(half*) messagesDeviceSet1Checkerboard0.messagesU,
			(half*) messagesDeviceSet1Checkerboard0.messagesD,
			(half*) messagesDeviceSet1Checkerboard0.messagesL,
			(half*) messagesDeviceSet1Checkerboard0.messagesR,
			(half*) messagesDeviceSet1Checkerboard1.messagesU,
			(half*) messagesDeviceSet1Checkerboard1.messagesD,
			(half*) messagesDeviceSet1Checkerboard1.messagesL,
			(half*) messagesDeviceSet1Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel * 2,
			heightLevel, currentCheckerboardSet);
}

template<>
void ProcessCUDABP<half2>::printDataAndMessageValsToPoint(int xVal, int yVal,
		const levelProperties& levelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<half2>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<half2>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<half2>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<half2>& messagesDeviceSet1Checkerboard1,
		const Checkerboard_Parts currentCheckerboardSet)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.printDataAndMessageValsToPoint(xVal, yVal, (half*)dataCostDeviceCurrentLevelCheckerboard1, (half*)dataCostDeviceCurrentLevelCheckerboard2,
			(half*) messagesDeviceSet0Checkerboard0.messagesU,
			(half*) messagesDeviceSet0Checkerboard0.messagesD,
			(half*) messagesDeviceSet0Checkerboard0.messagesL,
			(half*) messagesDeviceSet0Checkerboard0.messagesR,
			(half*) messagesDeviceSet0Checkerboard1.messagesU,
			(half*) messagesDeviceSet0Checkerboard1.messagesD,
			(half*) messagesDeviceSet0Checkerboard1.messagesL,
			(half*) messagesDeviceSet0Checkerboard1.messagesR,
			(half*) messagesDeviceSet1Checkerboard0.messagesU,
			(half*) messagesDeviceSet1Checkerboard0.messagesD,
			(half*) messagesDeviceSet1Checkerboard0.messagesL,
			(half*) messagesDeviceSet1Checkerboard0.messagesR,
			(half*) messagesDeviceSet1Checkerboard1.messagesU,
			(half*) messagesDeviceSet1Checkerboard1.messagesD,
			(half*) messagesDeviceSet1Checkerboard1.messagesL,
			(half*) messagesDeviceSet1Checkerboard1.messagesR, currentLevelProperties.widthCheckerboardLevel * 2,
			levelProperties.heightLevel, currentCheckerboardSet);
}

//due to the checkerboard indexing, half2 must be converted to half with the half function used for copying to the next level
template<>
void ProcessCUDABP<half2>::copyMessageValuesToNextLevelDown(
		const levelProperties& currentLevelProperties,
		const levelProperties& nextlevelProperties,
		const checkerboardMessages<half2>& messagesDeviceCheckerboard0CopyFrom,
		const checkerboardMessages<half2>& messagesDeviceCheckerboard1CopyFrom,
		const checkerboardMessages<half2>& messagesDeviceCheckerboard0CopyTo,
		const checkerboardMessages<half2>& messagesDeviceCheckerboard1CopyTo)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.copyMessageValuesToNextLevelDown(
			prevlevelProperties,
			currentLevelProperties,
			(half*)messagesDeviceCheckerboard0CopyFrom.messagesU,
			(half*)messagesDeviceCheckerboard0CopyFrom.messagesD,
			(half*)messagesDeviceCheckerboard0CopyFrom.messagesL,
			(half*)messagesDeviceCheckerboard0CopyFrom.messagesR,
			(half*)messagesDeviceCheckerboard1CopyFrom.messagesU,
			(half*)messagesDeviceCheckerboard1CopyFrom.messagesD,
			(half*)messagesDeviceCheckerboard1CopyFrom.messagesL,
			(half*)messagesDeviceCheckerboard1CopyFrom.messagesR,
			(half*)messagesDeviceCheckerboard0CopyTo.messagesU,
			(half*)messagesDeviceCheckerboard0CopyTo.messagesD,
			(half*)messagesDeviceCheckerboard0CopyTo.messagesL,
			(half*)messagesDeviceCheckerboard0CopyTo.messagesR,
			(half*)messagesDeviceCheckerboard1CopyTo.messagesU,
			(half*)messagesDeviceCheckerboard1CopyTo.messagesD,
			(half*)messagesDeviceCheckerboard1CopyTo.messagesL,
			(half*)messagesDeviceCheckerboard1CopyTo.messagesR);
}

//due to indexing, need to convert to half* and use half arrays for this function
template<>
void ProcessCUDABP<half2>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		half2* dataCostStereoCheckerboard1,
		half2* dataCostStereoCheckerboard2,
		half2* dataCostDeviceToWriteToCheckerboard1,
		half2* dataCostDeviceToWriteToCheckerboard2)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.initializeDataCurrentLevel(currentLevelProperties,
			prevLevelProperties,
			(half*)dataCostStereoCheckerboard1,
			(half*)dataCostStereoCheckerboard2,
			(half*)dataCostDeviceToWriteToCheckerboard1,
			(half*)dataCostDeviceToWriteToCheckerboard2);
}

#endif

template<typename T>
void ProcessCUDABP<T>::retrieveOutputDisparity(const Checkerboard_Parts currentCheckerboardSet,
		const levelProperties& levelProperties,
		T* dataCostDeviceCurrentLevelCheckerboard1,
		T* dataCostDeviceCurrentLevelCheckerboard2,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
		const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
		float* resultingDisparityMapCompDevice)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int) ceil((float) levelProperties.widthCheckerboardLevel / (float) threads.x);
	grid.y = (unsigned int) ceil((float) levelProperties.heightLevel / (float) threads.y);

	if (currentCheckerboardSet == Checkerboard_Parts::CHECKERBOARD_PART_0)
	{
		retrieveOutputDisparityCheckerboardStereoOptimized<T> <<<grid, threads>>>(levelProperties,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet0Checkerboard0.messagesU,
				messagesDeviceSet0Checkerboard0.messagesD,
				messagesDeviceSet0Checkerboard0.messagesL,
				messagesDeviceSet0Checkerboard0.messagesR,
				messagesDeviceSet0Checkerboard1.messagesU,
				messagesDeviceSet0Checkerboard1.messagesD,
				messagesDeviceSet0Checkerboard1.messagesL,
				messagesDeviceSet0Checkerboard1.messagesR, resultingDisparityMapCompDevice);
	}
	else
	{
		retrieveOutputDisparityCheckerboardStereoOptimized<T> <<<grid, threads>>>(levelProperties,
				dataCostDeviceCurrentLevelCheckerboard1,
				dataCostDeviceCurrentLevelCheckerboard2,
				messagesDeviceSet1Checkerboard0.messagesU,
				messagesDeviceSet1Checkerboard0.messagesD,
				messagesDeviceSet1Checkerboard0.messagesL,
				messagesDeviceSet1Checkerboard0.messagesR,
				messagesDeviceSet1Checkerboard1.messagesU,
				messagesDeviceSet1Checkerboard1.messagesD,
				messagesDeviceSet1Checkerboard1.messagesL,
				messagesDeviceSet1Checkerboard1.messagesR, resultingDisparityMapCompDevice);
	}

	(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());
}



#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
template class ProcessCUDABP<float>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
template class ProcessCUDABP<double>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF
template class ProcessCUDABP<half>;
#endif //CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT

//not currently supporting half2 data type
//template class ProcessCUDABP<half2>;
